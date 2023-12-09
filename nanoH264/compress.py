import io
import pickle
import numpy as np

from utils.metrics import compute_residual_energy
from utils.numpy_utils import serialize_array, deserialize_array

from .intra_prediction import IntraPredictionModel
from .inter_prediction import InterPredictionModel
from .macroblocks import get_macroblocks


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

    # Stuff for decoding/decompression
    self.last_encode_shape = None
    self.mb_array_byte_size = None

    self.intra_model = IntraPredictionModel()
    self.inter_model = InterPredictionModel()

  def encode(self, video, use_residual=True):
    '''
    For now this will autoregressively in raster-scan order go over the macroblocks and try to 
    predict them based on past encoded macroblocsk
    '''
    macroblocks = get_macroblocks(video, self.macroblock_size)
    T, H, W = macroblocks.shape

    encoding = {
      "header": (T, H, W, self.macroblock_size),
      "body": [],
    }

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

          encoding['body'].append({
            "pred_type": prediction_type_enum[params['type']],
            "code": params['code'],
            "residual": curr_mb.residual if use_residual else 0,
          })

    return encoding

  def decode(self, encoding, use_residual=True):
    T, H, W, mb_size = encoding['header']
    reco_video = np.zeros((T, H * self.macroblock_size, W * mb_size))
    reco_mbs = get_macroblocks(reco_video, self.macroblock_size)

    mb_encodings = encoding['body']
    for i, enc in enumerate(mb_encodings):
      t, h, w = i // (H * W), (i % (H * W)) // W, i % W
      curr_mb = reco_mbs.grid[t][h][w]

      prediction_type_code, code, residual = enc['pred_type'], enc['code'], enc['residual']

      if prediction_type_code == 0:  # store
        curr_mb.content = code
      elif prediction_type_code == 1:  # intra
        intra_prediction = self.intra_model.decode_prediction(curr_mb, reco_mbs, code)
        prediction = intra_prediction + (residual if use_residual else 0)
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
