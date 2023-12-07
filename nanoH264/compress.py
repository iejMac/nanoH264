import numpy as np


class Compressor:
  def compress(self, video):
    """
    video: np.ndarray with video of shape (n_frames, h, w)
    Output: Binary representation of the encoded video
    """
    raise NotImplementedError

  def decompress(self, bin_encoding):
    """
    bin_encoding: Binary representation of the encoded video
    Output: np.ndarray with video reconstruction of shape (n_frames, h, w)
    """
    raise NotImplementedError
    

class IdentityCompressor(Compressor):
  def __init__(self):
    self.encoding_dtype = np.uint8

  def encode(self, video):
    return video

  def decode(self, encoding):
    return encoding

  def compress(self, video):
    encoding = self.encode(video)
    bin_encoding = encoding.tobytes()
    return bin_encoding

  def decompress(self, bin_encoding):
    encoding = np.frombuffer(bin_encoding, dtype=self.encoding_dtype)
    reconstruction = self.decode(encoding)
    return reconstruction
