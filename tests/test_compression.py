import json
import numpy as np

from nanoH264 import BestCompressor, best_config

from utils.logging import log_compression_report
from utils.metrics import Criterion


def test_compressor():
  with open("current_criterion.json", "r") as file:
    crit_config = json.load(file)
  crit = Criterion(**crit_config)

  vid = np.load("data/moving-mnist-1.npy")

  cmp = BestCompressor(**best_config)

  byte_enc = cmp.compress(vid)
  rec = cmp.decompress(byte_enc)

  compression_report = crit(vid, byte_enc, rec)
  log_compression_report(compression_report)

  for grade, p in compression_report['grades'].items():
    assert grade and p
