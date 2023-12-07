import pprint
import numpy as np

from utils import serialize_array


class Criterion:
  def __init__(self, compression_rate=2.0, acceptable_error_rate=0.2, epsilon=20):
    """
    More interpretable criterion for successful compression

    compression_rate: threshold for len(x)/len(code) 
    acceptable_error_rate: proportion of pixels that can be different in reconstruction
    epsilon: by how much on average

    These values can be tuned by looking at adding various amount of random noise
    """
    self.compression_rate = compression_rate
    self.acceptable_error_rate = acceptable_error_rate
    self.epsilon = epsilon
 
  def __call__(self, vid, byte_code, rec):
    """
    vid - np.ndarray of shape (n_frames, h, w) with original video
    byte_code - byte representation of the encoded video
    rec - np.ndarray of shape (n_frames, h, w) with reconstructed video
    """
    real_compression_rate = len(serialize_array(vid)) / len(byte_code)
    residual = np.abs(vid - rec)
    num_errors = np.sum(residual > self.epsilon)

    report = {
      "compresion_rate": real_compression_rate,
      "residual_energy": compute_residual_energy(vid, rec),
      "grades": {
        "passed_compression": (real_compression_rate > self.compression_rate),
        "passed_quality": (num_errors / np.prod(vid.shape) < self.acceptable_error_rate),
      },
    }
    return report


def compute_residual_energy(x1, x2):
  return np.sum((x1-x2)**2)
