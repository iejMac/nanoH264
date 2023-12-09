import numpy as np

from utils.metrics import compute_residual_energy


def prediction_mode(func):
    """Decorator to mark and register 'prediction_mode' methods in a class-level dictionary."""
    if not hasattr(func, '_is_prediction_mode'):
        func._is_prediction_mode = True
    return func

def generate_motion_vectors(distance):
  vs = [(x, y) for x in range(-distance, distance + 1) for y in range(-distance, distance + 1)]
  vs.sort(key=lambda v: abs(v[0]) + abs(v[1]))
  return vs


class InterPredictionModel:
  """Produces best inter-frame prediction for the current macroblock"""
  _prediction_modes = {}

  def __init__(self, t_search_range=1, s_search_range=2):
    """
    t_search_range - how many frames back can we search
    s_search_range - how many blocks to the sides can we search
    """
    self.t_search_range, self.s_search_range = t_search_range, s_search_range
    self.space_motion_vectors = generate_motion_vectors(s_search_range)

    # Register all decorated mode methods in the instance.
    for name, method in type(self).__dict__.items():
      if callable(method) and hasattr(method, "_is_prediction_mode"):
        self._prediction_modes[name] = method.__get__(self, type(self))

  def get_prediction(self, curr_mb, mbs):
    """Returns best prediction for current macroblock."""
    min_residual_energy, min_params, min_prediction = float("inf"), {}, None

    for k, mode in self._prediction_modes.items():
      params, prediction = mode(curr_mb, mbs)
      if prediction is not None:  # valid prediction mode
        residual_energy = compute_residual_energy(curr_mb.content, prediction)
        if residual_energy < min_residual_energy:
          min_residual_energy, min_params, min_prediction = residual_energy, params, prediction
  
    min_params["type"] = "inter"  # for logging/visualization
    return min_params, min_prediction, min_residual_energy

  def decode_prediction(self, curr_mb, mbs, code):
    t, h, w = curr_mb.position
    dt, dh, dw = code
    return mbs.grid[t + dt][h + dh][w + dw].content


  # NOTE: order is important here, decode_prediction relies on this
  @prediction_mode
  def motion_vectors(self, curr_mb, mbs):
    """Searches over search range to find a good reference block."""
    # TODO: each search gets a budget instead of range
    # condition search on past coded motion vectors of previous blocks
    params, prediction = {"prediction_mode": "motion_vectors", "code": None}, None
    T, H, W = mbs.shape
    t, h, w = curr_mb.position
    if t == 0:
      return params, prediction

    best_v, best_re = (0, 0, 0), float("inf")
    for dt in range(1, self.t_search_range + 1):
      if t - dt >= 0:
        for dh, dw in self.space_motion_vectors:
          if (0 <= h + dh < H) and (0 <= w + dw < W):
            ref_block, ref_v = mbs.grid[t - dt][h + dh][w + dw], (-dt, dh, dw)
            re = compute_residual_energy(curr_mb.content, ref_block.content)
            if re < best_re:
              best_v, best_re = ref_v, re

    dt, dh, dw = best_v
    params["code"] = best_v
    prediction = mbs.grid[t + dt][h + dh][w + dw].content
    return params, prediction
