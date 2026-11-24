# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""olmo_mm_batch_hash_steps: MM-BATCH-HASH records match OLMo-core's BatchHashCallback hashes."""

import hashlib
import json
import unittest
from unittest import mock

import numpy as np

from maxtext.input_pipeline import olmo_mm_data_processing as mm

try:
  import torch
except ImportError:  # pragma: no cover
  torch = None


def _batch(seed=0, rows=2, seq=16):
  g = torch.Generator().manual_seed(seed)
  return {
      "input_ids": torch.randint(0, 100352, (rows, seq), generator=g),
      "labels": torch.randint(-100, 100352, (rows, seq), generator=g),
      "loss_masks": torch.rand((rows, seq), generator=g),
      "position_ids": torch.arange(seq).repeat(rows, 1),
      "document_ids": torch.zeros((rows, seq), dtype=torch.int64),
      "pooled_patches_idx": torch.randint(-1, 729, (rows, 5, 4), generator=g, dtype=torch.int32),
      "image_crop_counts": torch.tensor([3, 0]),
      "images": torch.rand((rows, 3, 729, 588), generator=g),  # not hashed
      "index": torch.tensor([7, 11]),
  }


@unittest.skipIf(torch is None, "needs torch (OLMo-core batches are torch tensors)")
class OlmoCoreBatchHashTest(unittest.TestCase):

  def test_definition(self):
    b = _batch()
    h = mm.olmo_core_batch_hash(b)
    keys = [k for k in mm.OLMO_CORE_BATCH_HASH_KEYS if k in b]
    self.assertEqual(sorted(h), sorted(keys + ["all"]))
    total = hashlib.sha256()
    for k in keys:
      data = np.ascontiguousarray(b[k].numpy()).tobytes()
      self.assertEqual(h[k], hashlib.sha256(data).hexdigest())
      total.update(k.encode())
      total.update(data)
    self.assertEqual(h["all"], total.hexdigest())
    self.assertNotIn("images", h)

  def test_matches_olmo_core(self):
    try:
      from olmo_core.train.callbacks.batch_hash import hash_batch  # pylint: disable=import-outside-toplevel
    except ImportError:
      self.skipTest("OLMo-core with train/callbacks/batch_hash.py (allenai/OLMo-core#942) not installed")
    for seed in range(3):
      b = _batch(seed)
      self.assertEqual(mm.olmo_core_batch_hash(b), hash_batch(b))
    # A changed row changes its key and "all", and nothing else.
    b, c = _batch(0), _batch(0)
    c["labels"][1, 3] += 1
    hb, hc = mm.olmo_core_batch_hash(b), mm.olmo_core_batch_hash(c)
    self.assertEqual({k for k in hb if hb[k] != hc[k]}, {"labels", "all"})

  def test_log_record_and_step_window(self):
    b = _batch()
    with mock.patch.object(mm.max_logging, "log") as log:
      rec = mm.maybe_log_olmo_core_batch_hash(b, step=3, rank=5, max_step=20)
      self.assertIsNone(mm.maybe_log_olmo_core_batch_hash(b, step=21, rank=5, max_step=20))
      self.assertIsNone(mm.maybe_log_olmo_core_batch_hash(b, step=1, rank=5, max_step=0))
    self.assertEqual(log.call_count, 1)
    line = log.call_args[0][0]
    self.assertTrue(line.startswith("MM-BATCH-HASH "))
    parsed = json.loads(line[len("MM-BATCH-HASH ") :])
    self.assertEqual(parsed, rec)
    self.assertEqual((parsed["step"], parsed["rank"], parsed["index"]), (3, 5, [7, 11]))
    self.assertEqual(parsed["hashes"], mm.olmo_core_batch_hash(b))

  def test_live_loader_counts_steps_from_resume(self):
    """_LiveMixtureHostLoader numbers its batches from initial_batch_step + 1, per OLMo-core rank."""
    batches = [_batch(s) for s in range(4)]
    loader = object.__new__(mm._LiveMixtureHostLoader)  # pylint: disable=protected-access
    # pylint: disable=protected-access
    loader._iter, loader._rank, loader._batch_step, loader._batch_hash_steps = iter(batches), 9, 10, 12
    loader._row_slice, loader._max_target_length, loader._max_crops, loader._max_pooled = None, 16, 3, 5
    loader._n_patches, loader._patch_dim, loader._pool_size, loader._dtype_mm = 729, 588, 4, np.float32
    loader._reset_positions, loader._padding_input_token_id, loader._use_loss_weights = True, -1, False
    # pylint: enable=protected-access
    with mock.patch.object(mm, "olmo_core_batch_to_maxtext_batch", lambda raw, **kw: {"inputs": raw["input_ids"]}):
      with mock.patch.object(mm.max_logging, "log") as log:
        for _ in range(3):
          next(loader)
    records = [json.loads(c[0][0][len("MM-BATCH-HASH ") :]) for c in log.call_args_list]
    self.assertEqual([(r["step"], r["rank"]) for r in records], [(11, 9), (12, 9)])  # step 13 is past the window
    self.assertEqual(records[0]["hashes"], mm.olmo_core_batch_hash(batches[0]))


if __name__ == "__main__":
  unittest.main()
