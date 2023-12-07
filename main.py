import numpy as np

from nanoH264 import VCmp
from utils.visualize import play_video, visualize_residual


if __name__ == "__main__":
  vid = np.load("data/moving-mnist-1.npy")
  cmp = VCmp()


  # first one frame
  vid = vid[:1]
  print(vid.shape)

  encoding = cmp.encode(vid)
  reconstruction = cmp.decode(encoding)
  residual = visualize_residual(vid, reconstruction)

  play_video([vid, reconstruction, residual])
