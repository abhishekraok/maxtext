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

"""olmo_mm with per_device_batch_size < 1: trained rows carry the OLMo-core ranks' rows, the others zeros."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import unittest

import jax
import numpy as np
from jax.sharding import Mesh

from maxtext.input_pipeline import olmo_mm_data_processing as mm


class _FakeRankLoader:
  """Rank r's batches: rows_per_rank rows whose inputs encode (rank, row, batch number)."""

  def __init__(self, rank, rows_per_rank):
    self.rank, self.rows, self.n = rank, rows_per_rank, 0

  def __iter__(self):
    return self

  def __next__(self):
    self.n += 1
    inputs = np.array([[1000 * self.rank + 10 * j + self.n] * 4 for j in range(self.rows)], dtype=np.int32)
    return {
        "inputs": inputs,
        "pooled_patches_idx": np.zeros((self.rows, 3, 4), dtype=np.int32),
        "images": np.ones((self.rows, 2, 3, 5), dtype=np.float32),
    }


class TrainedRowsTest(unittest.TestCase):

  def test_rows_map_to_ranks_and_untrained_rows_are_blank(self):
    local_rows = [5, 40, 2, 3, 63, 0]  # a host's rows in local order; trained: rows < 8
    cutoff, rows_per_rank = 8, 2
    ranks = sorted({r // rows_per_rank for r in local_rows if r < cutoff})
    loader = mm._TrainedRowsHostLoader(  # pylint: disable=protected-access
        local_rows, cutoff, rows_per_rank, {r: _FakeRankLoader(r, rows_per_rank) for r in ranks}
    )
    for n in (1, 2):
      batch = next(loader)
      for i, row in enumerate(local_rows):
        if row < cutoff:
          rank, offset = divmod(row, rows_per_rank)
          self.assertEqual(batch["inputs"][i, 0], 1000 * rank + 10 * offset + n)
          self.assertTrue((batch["images"][i] == 1).all())
          self.assertTrue((batch["pooled_patches_idx"][i] == 0).all())
        else:
          self.assertTrue((batch["inputs"][i] == 0).all())
          self.assertTrue((batch["images"][i] == 0).all())
          self.assertTrue((batch["pooled_patches_idx"][i] == -1).all())

  def test_host_rows_cover_the_loaded_batch_once(self):
    if len(jax.devices()) < 8:
      self.skipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    devices = np.array(jax.devices()[:8]).reshape(1, 1, 4, 2)
    mesh = Mesh(devices, ("data", "stage", "fsdp", "expert"))
    rows = mm.host_loaded_global_rows(mesh, 16, 32)
    self.assertEqual(sorted(rows), list(range(16)))
    # Local row i lands on mesh.local_devices[i // 2] (multihost_dataloading._form_global_array): check against the
    # sharding's own index map.
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(mesh.axis_names))
    index = sharding.devices_indices_map((16, 32))
    for i, row in enumerate(rows):
      d = mesh.local_devices[i // 2]
      self.assertIn(row, range(*index[d][0].indices(16)))


if __name__ == "__main__":
  unittest.main()
