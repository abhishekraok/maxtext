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

"""olmo_connector_gather_once / olmo_connector_skip_empty_chunks / olmo_vit_chunk_remat: same output and gradients.

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
from maxtext.models import olmoe3_vision
from maxtext.utils import maxtext_utils

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASE_YML = os.path.join(_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


def _cfg(balance: bool, n_crops: int, chunk: int, extra=()):
  return pyconfig.initialize([
      "",
      _BASE_YML,
      "run_name=ut_con_gather_once",
      "base_output_directory=/tmp/ut_con_gather_once",
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
      "olmo_vit_cp_shard_crops=True",
      f"olmo_vit_balance_crops={balance}",
      f"olmo_max_crops_per_seq={n_crops}",
      "custom_mesh_and_rule=olmo35-ep-as-cp",
      "context_parallel_load_balance=False",
      "ici_data_parallelism=1",
      "ici_fsdp_parallelism=2",
      "ici_expert_parallelism=4",
      "per_device_batch_size=0.5",
      "max_target_length=64",
      "skip_jax_distributed_system=True",
      *extra,
  ])


_NEW = (
    "olmo_connector_gather_once=True",
    "olmo_connector_skip_empty_chunks=True",
    "olmo_vit_chunk_remat=False",
)


class ConnectorGatherOnceTest(parameterized.TestCase):

  @parameterized.parameters(
      # (rows, crops per row, chunk, crops the connector reads per row, balance, extra flags)
      (4, 16, 2, (5, 0, 16, 1), True, _NEW),
      (4, 16, 2, (5, 0, 16, 1), True, _NEW[:1]),
      (4, 16, 4, (0, 0, 0, 0), True, _NEW),
      (2, 32, 2, (9, 30), False, _NEW),
      (2, 32, 2, (9, 30), True, _NEW[2:]),
  )
  def test_matches_reference(self, rows, n_crops, chunk, used, balance, extra):
    if jax.device_count() < 8:
      self.skipTest("needs 8 devices")
    ref_cfg = _cfg(balance, n_crops, chunk)
    new_cfg = _cfg(balance, n_crops, chunk, extra)
    mesh = Mesh(maxtext_utils.create_device_mesh(ref_cfg), ref_cfg.mesh_axes)
    p = 4  # (28 / 14) ** 2 patches per crop
    rng = np.random.default_rng(0)
    imgs = jnp.asarray(rng.standard_normal((rows, n_crops, p, 588)).astype(np.float32))
    n_pooled = n_crops * p // 4 * 4
    idx = np.full((rows, n_pooled, 4), -1, dtype=np.int32)
    for b, n in enumerate(used):
      if n:
        flat = rng.permutation(n * p)
        k = len(flat) // 4
        # spread the valid pooled tokens over the row so several chunks and CP ranks hold some
        pos = np.sort(rng.choice(n_pooled, size=k, replace=False))
        idx[b, pos] = flat[: 4 * k].reshape(k, 4)
    idx = jnp.asarray(idx)

    def run(cfg):
      enc = olmoe3_vision.OLMo35VisionEncoder(cfg, mesh, rngs=nnx.Rngs(params=0))
      con = olmoe3_vision.OLMo35VisionConnector(cfg, mesh, rngs=nnx.Rngs(params=1))

      def loss(enc_m, con_m):
        feats = enc_m(imgs, pooled_patches_idx=idx)
        out = con_m(feats, idx)
        return jnp.sum(jnp.sin(out)), out

      with jax.set_mesh(mesh):
        (l, out), grads = jax.jit(
            lambda e, c: nnx.value_and_grad(loss, argnums=(0, 1), has_aux=True)(e, c)
        )(enc, con)
      return l, out, grads

    l_ref, o_ref, g_ref = run(ref_cfg)
    if "olmo_connector_gather_once=True" in extra:
      with mock.patch.object(
          olmoe3_vision.OLMo35VisionConnector,
          "_forward_connector_maybe_chunked",
          side_effect=AssertionError("fell back"),
      ):
        l_new, o_new, g_new = run(new_cfg)
    else:
      l_new, o_new, g_new = run(new_cfg)
    np.testing.assert_allclose(np.asarray(o_new), np.asarray(o_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(l_new), float(l_ref), rtol=1e-6, atol=1e-6)
    leaves_new = jax.tree.leaves(nnx.state(g_new))
    leaves_ref = jax.tree.leaves(nnx.state(g_ref))
    self.assertEqual(len(leaves_new), len(leaves_ref))
    for a, b in zip(leaves_new, leaves_ref):
      np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
