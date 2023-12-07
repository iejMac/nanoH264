import numpy as np


class MacroBlock:
  def __init__(self, t, y, x, content):
    self.position = (t, y, x)
    self.size = content.shape[-1]
    self.content = content
    self.residual = None  # content - best_prediction
    self.params = {}
  @property
  def encoded(self):
    return self.prediction != None


class MacroBlockGrid:
  def __init__(self, blocks):
    T, H, W = blocks.shape[:3]
    self.shape = (T, H, W)

    self.grid = []

    for t in range(T):
      self.grid.append([])
      for h in range(H):
        self.grid[t].append([])
        for w in range(W):
          block = MacroBlock(
            t, h, w,
            blocks[t, h, w]
          )
          self.grid[t][h].append(block)

  def reassemble_video(self):
    T, H, W = self.shape
    block_size = self.grid[0][0][0].size  # Assuming all blocks are the same size
    reassembled_video = np.zeros((T, H * block_size, W * block_size), dtype=self.grid[0][0][0].content.dtype)
    for t in range(T):
      for h in range(H):
        for w in range(W):
          block = self.grid[t][h][w]
          y, x = h * block_size, w * block_size
          reassembled_video[t, y:y+block_size, x:x+block_size] = block.content
    return reassembled_video


# TODO: try to this using stride tricks
def get_mb_array(video, block_size):
  n_frames, h, w = video.shape
  macroblocks = np.empty((n_frames, h // block_size, w // block_size, block_size, block_size), dtype=video.dtype)
  for i in range(n_frames):
    for h_ind in range(h // block_size):
      for w_ind in range(w // block_size):
        macroblocks[i, h_ind, w_ind] = video[i, h_ind * block_size:(h_ind + 1) * block_size,
                                            w_ind * block_size:(w_ind + 1) * block_size]
  return macroblocks


def get_macroblocks(video, block_size):
  macroblocks = get_mb_array(video, block_size)
  return MacroBlockGrid(macroblocks)
