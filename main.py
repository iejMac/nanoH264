import numpy as np
from nanoH264 import IdentityCompressor
from utils import play_video


if __name__ == "__main__":
  # Load test video
  vid = np.load("data/moving-mnist-1.npy")
  print(vid.shape)
  # play_video(vid, 256)

  cmp = IdentityCompressor()

  bin_encoding = cmp.compress(vid)

  print(type(bin_encoding))
  print(len(bin_encoding))


