"""MOE-DROPPED in pre_train: per-layer dropped and attempted (routed) counts on one line per update (H269 gate 4)."""
import re
from types import SimpleNamespace

import numpy as np

from maxtext.trainers.pre_train.train import required_rbf_log_lines


def test_dropped_and_routed_per_layer():
  # three layers of [factor, dropped rows, routed rows], layer-major, as moe_required_rbf carries them
  probe = np.array([1.5, 2, 100, 2.25, 0, 120, 3.0, 7, 90], dtype=np.float32)
  lines = required_rbf_log_lines(SimpleNamespace(log_moe_dropped_tokens=True), 12, probe, " program=train")
  assert lines[0].startswith("REQUIRED_RBF step=12 max=3.0000 per_layer=[1.5000,2.2500,3.0000]")
  m = re.fullmatch(r"MOE-DROPPED step=12 dropped=9 routed=310 frac=(\S+) per_layer=\[2,0,7\] "
                   r"routed_per_layer=\[100,120,90\] program=train", lines[1])
  assert m and abs(float(m.group(1)) - 9 / 310) < 1e-6, lines[1]


def test_ring_of_experts_logs_minus_one():
  probe = np.array([1.0, -1, -1, 1.0, -1, -1], dtype=np.float32)
  line = required_rbf_log_lines(SimpleNamespace(log_moe_dropped_tokens=True), 3, probe)[1]
  assert "per_layer=[-1,-1]" in line and "routed_per_layer=[-1,-1]" in line


def test_without_drop_logging_only_the_factor_line():
  assert len(required_rbf_log_lines(SimpleNamespace(log_moe_dropped_tokens=False), 1, np.array([1.0, 2.0]))) == 1
