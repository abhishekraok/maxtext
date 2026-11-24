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

"""log_nonfinite_probe: per (row, CP rank) and per ViT stage probes localize injected non-finite values.

Fake 16-device mesh data2 x fsdp2 x ep4 on olmo35-ep-as-cp (crops CP-sharded over the expert axis, 4 crops per rank).
Injects (a) NaN pixels into one valid crop (one (row, rank) block, appears at stage 0) and (b) NaN into the block-1
weights as used (test hook ``_vit_weight_as_used``, appears at stage 2 in every block).

Run with 16 host devices: XLA_FLAGS=--xla_force_host_platform_device_count=16 JAX_PLATFORMS=cpu.
"""

from __future__ import annotations

import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import encoders
from maxtext.models import olmoe3_vision
from maxtext.utils import maxtext_utils

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASE_YML = os.path.join(_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

_ROWS, _CROPS, _P, _CP, _LAYERS = 4, 16, 4, 4, 3
_USED = (5, 0, 16, 1)


def _cfg(chunk: int, cp_shard: bool = True, *extra: str):
  return pyconfig.initialize(
      [
          "",
          _BASE_YML,
          "run_name=ut_vit_block_probe",
          "base_output_directory=/var/tmp/ut_vit_block_probe",
          "dataset_type=synthetic",
          "model_name=olmo35-tiny",
          "override_model_config=True",
          "base_emb_dim=64",
          "dtype=float32",
          "dtype_mm=float32",
          "weight_dtype=float32",
          "matmul_precision=highest",
          "use_multimodal=True",
          "vision_encoder_block=olmo35",
          "image_size_for_vit=28",
          "patch_size_for_vit=14",
          "hidden_size_for_vit=32",
          "intermediate_size_for_vit=64",
          "num_attention_heads_for_vit=4",
          f"num_hidden_layers_for_vit={_LAYERS}",
          "olmo_vit_feature_layers=[2,1]",
          "olmo_connector_mlp_hidden_size=96",
          "olmo_vit_remat_blocks=True",
          f"olmo_vit_crop_chunk_size={chunk}",
          f"olmo_vit_cp_shard_crops={cp_shard}",
          "olmo_connector_gather_once=False",
          "olmo_vit_balance_crops=False",
          f"olmo_max_crops_per_seq={_CROPS}",
          "custom_mesh_and_rule=olmo35-ep-as-cp",
          "context_parallel_load_balance=False",
          "ici_data_parallelism=2",
          "ici_fsdp_parallelism=2",
          f"ici_expert_parallelism={_CP}",
          "enable_checkpointing=False",
          "skip_jax_distributed_system=True",
          "per_device_batch_size=0.25",
          "max_target_length=64",
          *extra,
      ]
  )


def _inputs():
  rng = np.random.default_rng(0)
  imgs = np.zeros((_ROWS, _CROPS, _P, 588), np.float32)
  idx = np.full((_ROWS, _CROPS * _P // 4, 4), -1, dtype=np.int32)
  for b, n in enumerate(_USED):
    imgs[b, :n] = np.clip(rng.standard_normal((n, _P, 588)), -1, 1)
    if n:
      flat = rng.permutation(n * _P)
      k = len(flat) // 4
      idx[b, :k] = flat[: 4 * k].reshape(k, 4)
  return imgs, jnp.asarray(idx)


def _poison_stage(stage):
  def hook(w, layer):
    if layer != stage:
      return w
    return w.reshape(-1).at[0].set(jnp.nan).reshape(w.shape)

  return hook


class VitBlockProbeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 16:
      self.skipTest("needs 16 devices")

  def _run(self, cfg, imgs, idx, hook=None):
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    enc = olmoe3_vision.OLMo35VisionEncoder(cfg, mesh, rngs=nnx.Rngs(params=0))
    patcher = (
        mock.patch.object(olmoe3_vision, "_vit_weight_as_used", hook)
        if hook
        else mock.patch.object(
            olmoe3_vision, "_vit_weight_as_used", olmoe3_vision._vit_weight_as_used  # pylint: disable=protected-access
        )
    )
    with patcher, jax.set_mesh(mesh):
      _, probes = jax.jit(lambda e: e(jnp.asarray(imgs), pooled_patches_idx=idx, return_probes=True))(enc)
    return {k: np.asarray(v) for k, v in probes.items()}, mesh

  @parameterized.parameters((4, True), (2, True), (4, False))  # flat CP path (prod), chunked CP scan, non-CP scan
  def test_localizes_pixel_block_and_weight_stage(self, chunk, cp_shard):
    cfg = _cfg(chunk, cp_shard)
    imgs, idx = _inputs()
    clean, _ = self._run(cfg, imgs, idx)
    self.assertEqual(clean["nonfinite_probe_vit_feats_blocks"].shape, (_ROWS, _CP))
    self.assertEqual(clean["nonfinite_probe_vit_layer_blocks"].shape, (_LAYERS + 1, _ROWS, _CP))
    self.assertEqual(clean["nonfinite_probe_vit_wused_blocks"].shape, (_LAYERS + 1, _ROWS, _CP))
    for k in ("nonfinite_probe_vit_feats_blocks", "nonfinite_probe_vit_layer_blocks", "nonfinite_probe_vit_wused_blocks"):
      np.testing.assert_array_equal(clean[k], 0, err_msg=k)

    # (a) One NaN pixel in valid crop 9 of row 2 -> rank 9 // 4 = 2, from stage 0 on; weights clean.
    bad = imgs.copy()
    bad[2, 9, 1, 5] = np.nan
    got, _ = self._run(cfg, bad, idx)
    lay = got["nonfinite_probe_vit_layer_blocks"]
    expect = np.zeros((_ROWS, _CP), bool)
    expect[2, 2] = True
    for s in range(_LAYERS + 1):
      np.testing.assert_array_equal(lay[s] > 0, expect, err_msg=f"stage {s}")
    np.testing.assert_array_equal(got["nonfinite_probe_vit_feats_blocks"] > 0, expect)
    np.testing.assert_array_equal(got["nonfinite_probe_vit_wused_blocks"], 0)

    # (b) NaN in block 1's weights as used (stage 2): visible in the weight probe at stage 2 only, in every
    # (row, rank) block; activations clean through stage 1, non-finite everywhere from stage 2 on.
    got, _ = self._run(cfg, imgs, idx, hook=_poison_stage(2))
    w = got["nonfinite_probe_vit_wused_blocks"]
    n_weights = 4 + 4 * 2 + 2 * 2  # LN scale/bias x2, attention 4 x (kernel, bias), MLP 2 x (kernel, bias)
    np.testing.assert_array_equal(w[2], n_weights)
    np.testing.assert_array_equal(np.delete(w, 2, axis=0), 0)
    lay = got["nonfinite_probe_vit_layer_blocks"]
    np.testing.assert_array_equal(lay[:2], 0)
    self.assertTrue(np.all(lay[2:] > 0))
    self.assertTrue(np.all(got["nonfinite_probe_vit_feats_blocks"] > 0))

    # Stage-0 weights (patch embedding / positional embedding) are probed as well.
    got, _ = self._run(cfg, imgs, idx, hook=_poison_stage(0))
    np.testing.assert_array_equal(got["nonfinite_probe_vit_wused_blocks"][0], 3)
    np.testing.assert_array_equal(got["nonfinite_probe_vit_wused_blocks"][1:], 0)

  def test_wrapper_sows_and_log_lines(self):
    from maxtext.trainers.pre_train import train  # pylint: disable=import-outside-toplevel

    cfg = _cfg(4, True, "log_nonfinite_probe=True")
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    imgs, idx = _inputs()
    imgs[0, 3, 0, 0] = np.nan  # row 0, rank 0
    vis = encoders.VisionEncoder(cfg, mesh, rngs=nnx.Rngs(params=0))

    @nnx.jit
    def fwd(m):
      out, _ = m(jnp.asarray(imgs), idx)
      return out, nnx.pop(m, nnx.Intermediate)

    with mock.patch.object(olmoe3_vision, "_vit_weight_as_used", _poison_stage(3)), jax.set_mesh(mesh):
      _, inter = fwd(vis)
    probe = train.nonfinite_probe_leaves(inter)
    names = {k.split("/")[-1] for k in probe}
    for n in ("nonfinite_probe_vit_feats_blocks", "nonfinite_probe_vit_layer_blocks", "nonfinite_probe_vit_wused_blocks"):
      self.assertIn(n, names)
    lines = train.nonfinite_probe_log_lines(7, jax.device_get(probe))
    by = {re.search(r"(nonfinite_probe_\w+)", l).group(1): l for l in lines}
    fb = by["nonfinite_probe_vit_feats_blocks"]
    self.assertIn(f"shape=[{_ROWS}, {_CP}] rows_x_ranks=[[", fb)
    lb = by["nonfinite_probe_vit_layer_blocks"]
    self.assertRegex(
        lb, r"first_stage=0 per_stage=\[\d+, \d+, \d+, \d+\] nonzero_cells=\d+ cells\(stage,row,rank,count\)="
    )
    self.assertIn("cells(stage,row,rank,count)=[[0, 0, 0, ", lb)
    wb = by["nonfinite_probe_vit_wused_blocks"]
    self.assertIn(f"first_stage=3 per_stage=[0, 0, 0, {16 * _ROWS * _CP}] nonzero_cells={_ROWS * _CP}", wb)
    self.assertLessEqual(wb.count("["), 2 + 64 + 1)
    for l in lines:
      print(l)

    mlines = train.mesh_device_log_lines(cfg, mesh)
    dev_lines = [l for l in mlines if l.startswith("MESH-DEVICE ")]
    blk_lines = [l for l in mlines if l.startswith("MESH-BLOCK ")]
    self.assertLen(dev_lines, 16)
    self.assertLen(blk_lines, 4 * _CP)  # (data x fsdp) batch shards x CP ranks, one device each
    self.assertTrue(all(l.count("id=") == 1 for l in blk_lines))
    print("\n".join(mlines[:3] + [l for l in mlines if l.startswith("MESH-BLOCK")][:3]))


if __name__ == "__main__":
  absltest.main()
