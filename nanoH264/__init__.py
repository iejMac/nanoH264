"""nanoH264"""
from .compress import VCmp

# For tests
BestCompressor = VCmp
best_config = {
  "macroblock_size": 4,
  "energy_threshold": 400,
}
