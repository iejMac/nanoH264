import io
import pickle
import numpy as np

from utils import serialize_array, deserialize_array


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
    

class IdentityCompressor(Compressor):
  def __init__(self):
    pass

  def compress(self, video):
    return serialize_array(video)

  def decompress(self, byte_encoding):
    return deserialize_array(byte_encoding)
