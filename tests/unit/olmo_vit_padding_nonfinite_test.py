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

"""OLMo 3.5 vision path: garbage (NaN / inf / huge) in padding crops never reaches valid outputs or weight gradients.

Padding crops (crop >= n_used = max(pooled_patches_idx) // P + 1; every crop of a text row) are uninitialized-memory
candidates on TPU. The test fills them with NaN / inf / 1e30 (pixels) and NaN / inf (ViT output features) and checks the
image embeddings and the vision weight gradients are finite and identical to the clean run.

Run with 8 host devices: XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu.
"""

from __future__ import annotations

import os
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

_ROWS, _CROPS, _P = 4, 16, 4  # (28 / 14) ** 2 patches per crop
_USED = (5, 0, 16, 1)  # crops the connector reads per row (row 1: a text row, row 2: no padding)


def _cfg(cp_shard: bool, gather_once: bool, chunk: int, *extra: str):
  return pyconfig.initialize([
      "",
      _BASE_YML,
      "run_name=ut_vit_padding",
      "base_output_directory=/tmp/ut_vit_padding",
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
      "num_hidden_layers_for_vit=3",
      "olmo_vit_feature_layers=[2,1]",
      "olmo_connector_mlp_hidden_size=96",
      "olmo_vit_remat_blocks=True",
      f"olmo_vit_crop_chunk_size={chunk}",
      f"olmo_vit_cp_shard_crops={cp_shard}",
      f"olmo_connector_gather_once={gather_once}",
      "olmo_vit_balance_crops=False",
      f"olmo_max_crops_per_seq={_CROPS}",
      "custom_mesh_and_rule=olmo35-ep-as-cp",
      "context_parallel_load_balance=False",
      "ici_data_parallelism=1",
      "ici_fsdp_parallelism=2",
      "enable_checkpointing=False",
      "skip_jax_distributed_system=True",
      "ici_expert_parallelism=4",
      "per_device_batch_size=0.5",
      "max_target_length=64",
      *extra,
  ])


def _inputs():
  rng = np.random.default_rng(0)
  imgs = np.zeros((_ROWS, _CROPS, _P, 588), np.float32)
  n_pooled = _CROPS * _P // 4 * 4
  idx = np.full((_ROWS, n_pooled, 4), -1, dtype=np.int32)
  for b, n in enumerate(_USED):
    imgs[b, :n] = np.clip(rng.standard_normal((n, _P, 588)), -1, 1)
    if n:
      flat = rng.permutation(n * _P)
      k = len(flat) // 4
      idx[b, :k] = flat[: 4 * k].reshape(k, 4)
  garbage = imgs.copy()
  for b, n in enumerate(_USED):
    pad = garbage[b, n:]
    pad[..., 0::3] = np.nan
    pad[..., 1::3] = np.inf
    pad[..., 2::3] = -1e30
  return jnp.asarray(imgs), jnp.asarray(garbage), jnp.asarray(idx)


def _poison_padding_features(orig):
  """Wraps ``_encode_crops_maybe_chunked`` so the padding crops' ViT output is NaN / inf (garbage feature buffers)."""
  used = jnp.asarray(_USED)

  def wrapped(self, crops, **kw):
    out = orig(self, crops, **kw)
    b, n, p = crops.shape[0], crops.shape[1], crops.shape[2]
    pad = (jnp.arange(n)[None, :] >= used[:b, None])[:, :, None, None]
    bad = jnp.where(jnp.arange(out.shape[-1]) % 2 == 0, jnp.nan, jnp.inf).astype(out.dtype)
    return jnp.where(pad, bad, out.reshape(b, n, p, -1)).reshape(out.shape)

  return wrapped


class VitPaddingNonfiniteTest(parameterized.TestCase):

  @parameterized.parameters(
      # (olmo_vit_cp_shard_crops, olmo_connector_gather_once, crop chunk)
      (True, False, 2),  # Small mm SFT path
      (True, True, 2),
      (False, False, 4),
  )
  def test_padding_garbage_does_not_leak(self, cp_shard, gather_once, chunk):
    if jax.device_count() < 8:
      self.skipTest("needs 8 devices")
    cfg = _cfg(cp_shard, gather_once, chunk)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    clean, garbage, idx = _inputs()

    def run(imgs, use_idx=True):
      enc = olmoe3_vision.OLMo35VisionEncoder(cfg, mesh, rngs=nnx.Rngs(params=0))
      con = olmoe3_vision.OLMo35VisionConnector(cfg, mesh, rngs=nnx.Rngs(params=1))

      def loss(enc_m, con_m):
        out = con_m(enc_m(imgs, pooled_patches_idx=idx if use_idx else None), idx)
        return jnp.sum(jnp.sin(out)), out

      with jax.set_mesh(mesh):
        (l, out), grads = jax.jit(lambda e, c: nnx.value_and_grad(loss, argnums=(0, 1), has_aux=True)(e, c))(enc, con)
      return float(l), np.asarray(out), [np.asarray(g) for g in jax.tree.leaves(nnx.state(grads))]

    l_ref, o_ref, g_ref = run(clean)
    self.assertTrue(np.isfinite(l_ref))
    self.assertTrue(np.all(np.isfinite(o_ref)))
    self.assertTrue(all(np.all(np.isfinite(g)) for g in g_ref))
    self.assertTrue(np.any(o_ref != 0))
    np.testing.assert_array_equal(o_ref[1], 0.0)  # text row

    # Valid numerics unchanged vs the unmasked encoder (clean pixels: zero padding crops).
    l_nm, o_nm, g_nm = run(clean, use_idx=False)
    np.testing.assert_allclose(o_ref, o_nm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(l_ref, l_nm, rtol=1e-6, atol=1e-6)
    for a, b in zip(g_ref, g_nm):
      np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    cases = {"pixels": (garbage, None), "features": (clean, "features"), "both": (garbage, "features")}
    for name, (imgs, poison) in cases.items():
      with self.subTest(name):
        if poison:
          orig = olmoe3_vision.OLMo35VisionEncoder._encode_crops_maybe_chunked  # pylint: disable=protected-access
          with mock.patch.object(
              olmoe3_vision.OLMo35VisionEncoder, "_encode_crops_maybe_chunked", _poison_padding_features(orig)
          ):
            l_x, o_x, g_x = run(imgs)
        else:
          l_x, o_x, g_x = run(imgs)
        self.assertTrue(np.all(np.isfinite(o_x)), f"{name}: non-finite image embeddings")
        np.testing.assert_array_equal(o_x, o_ref)
        self.assertEqual(l_x, l_ref)
        for i, (a, b) in enumerate(zip(g_x, g_ref)):
          self.assertTrue(np.all(np.isfinite(a)), f"{name}: non-finite gradient leaf {i}")
          np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

  def test_probes(self):
    if jax.device_count() < 8:
      self.skipTest("needs 8 devices")
    cfg = _cfg(True, False, 2)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    _, garbage, idx = _inputs()
    enc = olmoe3_vision.OLMo35VisionEncoder(cfg, mesh, rngs=nnx.Rngs(params=0))
    with jax.set_mesh(mesh):
      feats, probes = jax.jit(lambda e: e(garbage, pooled_patches_idx=idx, return_probes=True))(enc)
    pad = np.array([_CROPS - n for n in _USED])
    np.testing.assert_array_equal(np.asarray(probes["nonfinite_probe_pixels_out_of_range"]), pad * _P * 588)
    # Padding pixels are zeroed before the ViT, so its output (probed before the feature mask) is finite too.
    np.testing.assert_array_equal(np.asarray(probes["nonfinite_probe_vit_feats"]), 0)
    np.testing.assert_array_equal(np.asarray(probes["nonfinite_probe_vit_feats_valid"]), 0)
    np.testing.assert_array_equal(np.asarray(probes["nonfinite_probe_vit_params"]), 0)
    self.assertTrue(np.all(np.isfinite(np.asarray(feats))))

  def test_wrapper_sows_probes(self):
    # The production path: VisionEncoder's outer-remat branch (CP > 1) returns the probes out of jax.checkpoint.
    if jax.device_count() < 8:
      self.skipTest("needs 8 devices")
    cfg = _cfg(True, False, 2, "log_nonfinite_probe=True")
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    _, garbage, idx = _inputs()
    vis = encoders.VisionEncoder(cfg, mesh, rngs=nnx.Rngs(params=0))

    @nnx.jit
    def fwd(m):
      out, _ = m(garbage, idx)
      return out, nnx.pop(m, nnx.Intermediate)

    with jax.set_mesh(mesh):
      out, inter = fwd(vis)
    got = {}
    for path, val in jax.tree_util.tree_leaves_with_path(inter):
      keys = [str(k.key) for k in path if hasattr(k, "key")]
      got[[k for k in keys if k.startswith("nonfinite_probe_")][0]] = np.asarray(val)
    self.assertEqual(
        sorted(got),
        [
            "nonfinite_probe_connector_params",
            "nonfinite_probe_pixels_out_of_range",
            "nonfinite_probe_vit_feats",
            "nonfinite_probe_vit_feats_blocks",
            "nonfinite_probe_vit_feats_valid",
            "nonfinite_probe_vit_layer_blocks",
            "nonfinite_probe_vit_params",
            "nonfinite_probe_vit_wused_blocks",
        ],
    )
    pad = np.array([_CROPS - n for n in _USED])
    np.testing.assert_array_equal(got["nonfinite_probe_pixels_out_of_range"], pad * _P * 588)
    for k in ("nonfinite_probe_connector_params", "nonfinite_probe_vit_feats", "nonfinite_probe_vit_params"):
      np.testing.assert_array_equal(got[k], 0)
    self.assertTrue(np.all(np.isfinite(np.asarray(out))))

  def test_few_key_attention_selects_invalid_keys(self):
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((3, 2, 4)), jnp.float32)
    k = rng.standard_normal((3, 4, 2, 4)).astype(np.float32)
    v = rng.standard_normal((3, 4, 2, 4)).astype(np.float32)
    valid = np.array([[True, True, False, False], [False] * 4, [True] * 4])
    k_bad, v_bad = k.copy(), v.copy()
    k_bad[~valid] = np.nan
    v_bad[~valid] = np.inf

    def f(q_, k_, v_):
      return jnp.sum(jnp.sin(olmoe3_vision._few_key_attention(q_, k_, v_, jnp.asarray(valid))))  # pylint: disable=protected-access

    ref = jax.value_and_grad(f, argnums=(0, 1, 2))(q, jnp.asarray(k), jnp.asarray(v))
    bad = jax.value_and_grad(f, argnums=(0, 1, 2))(q, jnp.asarray(k_bad), jnp.asarray(v_bad))
    self.assertEqual(float(bad[0]), float(ref[0]))
    for a, b in zip(bad[1], ref[1]):
      self.assertTrue(np.all(np.isfinite(np.asarray(a))))
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

  def test_crop_valid(self):
    idx = jnp.asarray(
        [
            [[0, 1, 2, 3], [9, -1, -1, -1]],  # max 9 -> crop 2 -> 3 used
            [[-1, -1, -1, -1], [-1, -1, -1, -1]],  # text row
        ],
        jnp.int32,
    )
    np.testing.assert_array_equal(
        np.asarray(olmoe3_vision.olmo35_crop_valid(idx, 4, 4)),
        [[True, True, True, False], [False, False, False, False]],
    )


if __name__ == "__main__":
  absltest.main()
