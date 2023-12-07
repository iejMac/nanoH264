"""nanoH264"""
from .compress import VCmp

from .metrics import Criterion, compute_residual_energy


# For tests
BestCompressor = VCmp
best_config = {
  "macroblock_size": 4,
  "energy_threshold": 400,
}
