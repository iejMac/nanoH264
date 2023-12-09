import numpy as np

from nanoH264 import VCmp
from utils.visualize import play_video, visualize_residual
from utils.numpy_utils import serialize_array


if __name__ == "__main__":
  vid = np.load("data/moving-mnist-1.npy")

  cmp = VCmp(
    macroblock_size=4,
    energy_threshold=1000,
  )

  # first one frame
  vid = vid[:1]

  encoding = cmp.encode(vid)
  code_types = [c['pred_type'] for c in encoding['body']]
  print(code_types)
  byte_stream = cmp.compress(vid)
  print(len(serialize_array(vid))/len(byte_stream))

  reconstruction = cmp.decode(encoding)

  viz_vids = [vid, reconstruction]
  viz_vids.append(visualize_residual(vid, reconstruction))

  play_video(viz_vids)
