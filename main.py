import numpy as np
from nanoH264 import VCmp
from utils import play_video


if __name__ == "__main__":
  vid = np.load("data/moving-mnist-1.npy")
  cmp = VCmp()

  encoding = cmp.encode(vid)

  reconstruction = cmp.decode(encoding)

  play_video(reconstruction)
