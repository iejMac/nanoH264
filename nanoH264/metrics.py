import numpy as np


def compute_residual_energy(x, y):
  return np.sum((x-y)**2)
