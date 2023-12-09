import numpy as np

from nanoH264 import VCmp
from utils.visualize import play_video, visualize_residual
from utils.numpy_utils import serialize_array


if __name__ == "__main__":
  vid = np.load("data/moving-mnist-1.npy")

  cmp = VCmp(
    macroblock_size=4,
    energy_threshold=400,
  )

  # first one frame
  vid = vid[:2]
  use_residual = False

  encoding = cmp.encode(vid, use_residual)

  # TODO: Maybe add some code for this type of viz
  code_types = np.array([c['pred_type'] for c in encoding['body']]).reshape(vid.shape[0], 16, 16)

  print(code_types)
  reconstruction = cmp.decode(encoding, use_residual)

  viz_vids = [vid, reconstruction]
  viz_vids.append(visualize_residual(vid, reconstruction))

  play_video(viz_vids)
