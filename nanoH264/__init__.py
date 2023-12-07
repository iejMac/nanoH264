"""nanoH264"""


from .compress import IdentityCompressor, VCmp

from .metrics import Criterion, compute_residual_energy


# For tests
BestCompressor = IdentityCompressor
