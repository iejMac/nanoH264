import numpy as np

from nanoH264 import VCmp
from utils.visualize import play_video, visualize_residual
from utils.numpy_utils import serialize_array
from utils.metrics import compute_residual_energy


if __name__ == "__main__":
  vid = np.load("data/moving-mnist-1.npy")

  cmp = VCmp(
    macroblock_size=8,
    temporal_range=1,
    spatial_range=2,
  )
  grid_size = vid.shape[-1] // cmp.macroblock_size

  # first one frame
  vid = vid[:2]

  viz_params = False
  viz_rec = False
  viz_comp = True

  if viz_params:
    # Interpretable codes for visualization
    # =====================================
    encoding = cmp.encode(vid)

    code_types = np.array([c['pred_type'] for c in encoding['body']])

    # TODO: Maybe add some code for this type of viz
    print(code_types.reshape(vid.shape[0], grid_size, grid_size))
    reconstruction = cmp.decode(encoding, use_residual=True)

    res = np.array([np.sum(c['residual']**2) for c in encoding['body']])
    # TODO: Maybe add some code for this type of viz
    # print(res.reshape(vid.shape[0], grid_size, grid_size))
    total_res_per_p = np.sum(res) / np.prod(vid.shape)
    print(total_res_per_p)
    # =====================================
  if viz_comp:
    # Compression:
    # =====================================
    ser_vid = serialize_array(vid)
    ser_enc = cmp.compress(vid)

    print(f"Vid size: {len(ser_vid)} [{type(ser_vid)}]")
    print(f"Enc size: {len(ser_enc)} [{type(ser_enc)}]")
    print(f"Compression rate: {len(ser_vid)/len(ser_enc)}")

    rec = cmp.decompress(ser_enc)
    # =====================================

    assert compute_residual_energy(vid, rec) == 0.0
  if viz_rec:
    viz_vids = [vid, reconstruction]
    viz_vids.append(visualize_residual(vid, reconstruction))

    play_video(viz_vids)
