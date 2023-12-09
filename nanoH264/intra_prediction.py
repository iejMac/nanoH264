import numpy as np

from utils.metrics import compute_residual_energy


def prediction_mode(func):
    """Decorator to mark and register 'prediction_mode' methods in a class-level dictionary."""
    if not hasattr(func, '_is_prediction_mode'):
        func._is_prediction_mode = True
    return func


class IntraPredictionModel:
  """Produces best intra-frame prediction for the current macroblock"""
  _prediction_modes = {}

  def __init__(self):
    # Register all decorated mode methods in the instance.
    for name, method in type(self).__dict__.items():
      if callable(method) and hasattr(method, '_is_prediction_mode'):
        self._prediction_modes[name] = method.__get__(self, type(self))

  def get_prediction(self, curr_mb, mbs):
    """Returns best prediction for current macroblock."""
    min_residual_energy, min_params, min_prediction = float('inf'), {}, None

    for k, mode in self._prediction_modes.items():
      params, prediction = mode(curr_mb, mbs)
      if prediction is not None:  # valid prediction mode
        residual_energy = compute_residual_energy(curr_mb.content, prediction)
        if residual_energy < min_residual_energy:
          min_residual_energy, min_params, min_prediction = residual_energy, params, prediction
   
    min_params["type"] = "intra"  # for logging/visualization
    return min_params, min_prediction, min_residual_energy

  def decode_prediction(self, curr_mb, mbs, code):
    params, prediction = list(self._prediction_modes.values())[code](curr_mb, mbs)
    return prediction

  # NOTE: order is important here, decode_prediction relies on this
  @prediction_mode
  def vertical(self, curr_mb, mbs):
    params, prediction = {"prediction_mode": "vertical", "code": 0}, None
    t, h, w = curr_mb.position
    if h != 0:  # Make sure there is a block above curr_mb
      prediction = mbs.grid[t][h-1][w].content
    return params, prediction

  @prediction_mode
  def horizontal(self, curr_mb, mbs):
    params, prediction = {"prediction_mode": "horizontal", "code": 1}, None
    t, h, w = curr_mb.position
    if w != 0:  # Make sure there is a block above curr_mb
      prediction = mbs.grid[t][h][w-1].content
    return params, prediction

  @prediction_mode
  def DC(self, curr_mb, mbs):
    params, prediction = {"prediction_mode": "DC", "code": 2}, None
    t, h, w = curr_mb.position
    if h != 0 and w != 0:  # Make sure there is a block above curr_mb
      prediction = (mbs.grid[t][h-1][w].content + mbs.grid[t][h][w-1].content) // 2
    return params, prediction
