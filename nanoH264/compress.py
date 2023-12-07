import io
import pickle
import numpy as np

from utils.numpy_utils import serialize_array, deserialize_array

from .intra_prediction import IntraPredictionModel
from .macroblocks import get_macroblocks
from .metrics import compute_residual_energy


class Compressor:
  def compress(self, video):
    """
    video: np.ndarray with video of shape (n_frames, h, w)
    Output: Byte representation of the encoded video
    """
    raise NotImplementedError

  def decompress(self, byte_encoding):
    """
    byte_encoding: Byte representation of the encoded video
    Output: np.ndarray with video reconstruction of shape (n_frames, h, w)
    """
    raise NotImplementedError


prediction_type_enum = {
  "store": 0,
  "intra": 1,
  # TODO: "inter": 2,
}


class VCmp(Compressor):
  def __init__(self, macroblock_size=16, energy_threshold=float('inf')):
    self.macroblock_size = macroblock_size
    self.energy_threshold = energy_threshold

    self.last_encode_shape = None

    self.intra_model = IntraPredictionModel()

  def encode(self, video):
    '''
    For now this will autoregressively in raster-scan order go over the macroblocks and try to 
    predict them based on past encoded macroblocsk
    '''
    macroblocks = get_macroblocks(video, self.macroblock_size)
    T, H, W = macroblocks.shape

    encoding = []

    for t in range(T):  # TODO: implement inter prediction
      for h in range(H):
        for w in range(W):
          curr_mb = macroblocks.grid[t][h][w]

          # TODO: maybe make this work at multiple macroblock size levels (check standard)
          params, prediction, residual_energy = self.intra_model.best_prediction(curr_mb, macroblocks)
          if not isinstance(prediction, np.ndarray):  # models weren't able to find a reference block
            params, prediction = {"type": "store", "code": curr_mb.content}, curr_mb.content

          # Select prediction method 
          if params["type"] == "intra" and residual_energy > self.energy_threshold:
            params, prediction = {"type": "store", "code": curr_mb.content}, curr_mb.content

          curr_mb.residual = curr_mb.content - prediction
          curr_mb.params = params

          # encoding.append([prediction_type_enum[params['type']], params['code'], curr_mb.residual])
          # TODO: lets see if no-residual can work fine
          encoding.append([prediction_type_enum[params['type']], params['code']])

    # TODO: this needs to be compressed in some type of header
    self.last_encode_shape = macroblocks.shape  # save for decode
    return encoding

  def decode(self, encodings):
    T, H, W = self.last_encode_shape
    reco_video = np.zeros((T, H * self.macroblock_size, W * self.macroblock_size))
    reco_mbs = get_macroblocks(reco_video, self.macroblock_size)

    for i, enc in enumerate(encodings):
      t, h, w = i // (H * W), (i % (H * W)) // W, i % W
      curr_mb = reco_mbs.grid[t][h][w]

      # TODO: lets see if no-residual can work fine
      prediction_type_code, code = enc
      # prediction_type_code, code, residual = enc

      if prediction_type_code == 0:  # store
        curr_mb.content = code
      elif prediction_type_code == 1:  # intra
        intra_prediction = self.intra_model.decode_prediction(curr_mb, reco_mbs, code)
        # prediction = intra_prediction + residual
        # TODO: lets see if no-residual can work fine
        prediction = intra_prediction
        curr_mb.content = prediction
      else:
        raise ValueError("Corrupted prediction type")

    # Put together video
    reco = reco_mbs.reassemble_video()
    return reco

  def compress(self, video):
    encoding = self.encode(video)
    # TODO: BE MORE SMART WHEN SERIALIZING INTRA CODES
    return serialize_array(encoding)

  def decompress(self, byte_encoding):
    encoding = deserialize_array(byte_encoding)
    reconstruction = self.decode(encoding)
    return reconstruction
