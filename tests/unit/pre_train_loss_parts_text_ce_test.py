# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LOSS-PARTS text / image CE split (log_loss_parts on multimodal batches) of the pre-training loss."""

from dataclasses import dataclass
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import vocabulary_tiling
from tests.unit.pre_train_loss_mask_test import _Config, _UniformNnxDecoder


@dataclass
class _MMConfig(_Config):
  micro_batch_size_to_train_on: int = 3  # per_device_batch_size < 1: row 3 is loaded but not trained
  micro_batch_size_to_eval_on: int = 3
  z_loss_multiplier: float = 1e-4
  use_multimodal: bool = True
  log_loss_parts: bool = True


_B, _S, _P, _K = 4, 5, 3, 2


def _batch(image_rows):
  rng = np.random.default_rng(0)
  pooled = np.full((_B, _P, _K), -1, np.int32)
  for r in image_rows:
    pooled[r, 0, 0] = 7
  seg = np.ones((_B, _S), np.int32)
  seg[0, -1] = 0
  lw = rng.uniform(0.0, 1.5, (_B, _S)).astype(np.float32)
  lw[1, :2] = 0.0
  return {
      "inputs": jnp.zeros((_B, _S), jnp.int32),
      "inputs_position": jnp.broadcast_to(jnp.arange(_S), (_B, _S)),
      "inputs_segmentation": jnp.ones((_B, _S), jnp.int32),
      "targets": jnp.zeros((_B, _S), jnp.int32),
      "targets_segmentation": jnp.asarray(seg),
      "loss_weights": jnp.asarray(lw),
      "pooled_patches_idx": jnp.asarray(pooled),
  }


class LossPartsTextCeTest(unittest.TestCase):
  """ce_text / ce_img from loss_fn's row split match hand-computed CE over the trained text / image rows."""

  def setUp(self):
    super().setUp()
    self.xent = jnp.asarray(np.random.default_rng(1).uniform(0.5, 3.0, (3, _S)).astype(np.float32))
    self.z = self.xent * 1e-3  # cross_entropy_with_logits adds the z-loss to each token's xent

  def _loss(self, config, data):
    model = _UniformNnxDecoder(config.vocab_size)
    with mock.patch.object(pre_train.max_utils, "cross_entropy_with_logits", return_value=(self.xent, self.z)):
      return pre_train.loss_fn(model, config, dict(data), None, None, is_train=True)

  def _hand_ce(self, data, rows):
    seg = np.asarray(data["targets_segmentation"])[:3] != 0
    lw = np.asarray(data["loss_weights"])[:3]
    ce = np.asarray(self.xent - self.z) * lw * seg
    return float(ce[rows].sum() / np.maximum(lw[rows], 0).sum()), float(np.maximum(lw[rows], 0).sum())

  def test_mixed_rows(self):
    data = _batch(image_rows=(1, 3))  # row 3 is an image row but untrained: it must not count
    loss, aux = self._loss(_MMConfig(), data)
    del loss
    ce = float(aux["xent_sum"] / aux["total_weights"] - aux["z_loss"])
    ce_text, w_text, ce_img, w_img = (float(x) for x in pre_train.text_image_ce(aux, ce))
    want_text, want_wt = self._hand_ce(data, [0, 2])
    want_img, want_wi = self._hand_ce(data, [1])
    self.assertAlmostEqual(ce_text, want_text, places=5)
    self.assertAlmostEqual(ce_img, want_img, places=5)
    self.assertAlmostEqual(w_text, want_wt, places=5)
    self.assertAlmostEqual(w_img, want_wi, places=5)
    self.assertAlmostEqual(w_text + w_img, float(aux["total_weights"]), places=5)
    self.assertAlmostEqual((ce_text * w_text + ce_img * w_img) / (w_text + w_img), ce, places=5)
    self.assertNotAlmostEqual(ce_text, ce, places=3)

  def test_text_only_batch(self):
    data = _batch(image_rows=(3,))  # the only image row is untrained
    _, aux = self._loss(_MMConfig(), data)
    ce = aux["xent_sum"] / aux["total_weights"] - aux["z_loss"]
    self.assertAlmostEqual(float(aux["ce_text_sum"] / aux["weights_text"]), float(ce), places=5)
    ce_text, w_text, ce_img, w_img = pre_train.text_image_ce(aux, ce)
    self.assertEqual(float(ce_text), float(ce))  # exactly the step's ce
    self.assertEqual(float(w_text), float(aux["total_weights"]))
    self.assertEqual(float(w_img), 0.0)
    self.assertEqual(float(ce_img), 0.0)

  def test_grad_accumulation_sums(self):
    """Microbatch sums add (gradient_accumulation sums aux over microbatches)."""
    data = _batch(image_rows=(0,))
    _, a1 = self._loss(_MMConfig(), data)
    data2 = _batch(image_rows=(2,))
    _, a2 = self._loss(_MMConfig(), data2)
    summed = jax.tree.map(lambda x, y: x + y, {k: a1[k] for k in ("ce_text_sum", "weights_text", "ce_img_sum",
                                                                    "weights_img")},
                          {k: a2[k] for k in ("ce_text_sum", "weights_text", "ce_img_sum", "weights_img")})
    ce_text, *_ = pre_train.text_image_ce(summed, 0.0)
    t1, w1 = self._hand_ce(data, [1, 2])
    t2, w2 = self._hand_ce(data2, [0, 1])
    self.assertAlmostEqual(float(ce_text), (t1 * w1 + t2 * w2) / (w1 + w2), places=5)

  def test_off_without_log_loss_parts_or_images(self):
    data = _batch(image_rows=(1,))
    _, aux = self._loss(_MMConfig(log_loss_parts=False), data)
    self.assertNotIn("ce_text_sum", aux)
    no_pooled = {k: v for k, v in data.items() if k != "pooled_patches_idx"}
    _, aux = self._loss(_MMConfig(), no_pooled)
    self.assertNotIn("ce_text_sum", aux)
    _, aux = self._loss(_MMConfig(use_multimodal=False), data)
    self.assertNotIn("ce_text_sum", aux)

  def test_loss_unchanged(self):
    data = _batch(image_rows=(1,))
    loss_on, aux_on = self._loss(_MMConfig(), data)
    loss_off, aux_off = self._loss(_MMConfig(log_loss_parts=False), data)
    self.assertEqual(float(loss_on), float(loss_off))
    self.assertEqual(float(aux_on["xent_sum"]), float(aux_off["xent_sum"]))

  def test_fused_per_row(self):
    """fused_lm_head_ce_nnx_loss(per_row=True): row sums of (loss - z) of the kernel's per-token outputs."""
    loss_tok = jnp.arange(1.0, 1.0 + _B * _S).reshape(_B, _S)
    z_tok = loss_tok / 100.0
    decoder = types.SimpleNamespace(
        decoder_norm=lambda h: h,
        dropout=lambda y, deterministic: y,
        logits_dense=types.SimpleNamespace(kernel=jnp.zeros((2, 8))),
    )
    model = types.SimpleNamespace(decoder=decoder, mesh=jax.make_mesh((1,), ("data",)))
    config = types.SimpleNamespace(logits_via_embedding=False, final_logits_soft_cap=None, logits_dot_in_fp32=False,
                                   quantize_logits_proj=False, enable_dropout=False, dtype=jnp.float32,
                                   z_loss_multiplier=1e-2, shard_mode=0, debug_sharding=False)
    data = _batch(image_rows=())
    with mock.patch.object(vocabulary_tiling, "create_sharding",
                           return_value=jax.sharding.NamedSharding(model.mesh, jax.sharding.PartitionSpec())), \
        mock.patch.object(vocabulary_tiling, "_fused_ce_config_from", return_value=None), \
        mock.patch.object(vocabulary_tiling.fused_ce, "sharded_fused_linear_cross_entropy",
                          return_value=(loss_tok, z_tok)):
      total, z, rows = vocabulary_tiling.fused_lm_head_ce_nnx_loss(
          model, jnp.zeros((_B, _S, 2)), data, config, True, per_row=True
      )
      total2, z2 = vocabulary_tiling.fused_lm_head_ce_nnx_loss(model, jnp.zeros((_B, _S, 2)), data, config, True)
    np.testing.assert_allclose(np.asarray(rows), np.asarray(jnp.sum(loss_tok - z_tok, axis=-1)), rtol=1e-6)
    self.assertEqual(float(total), float(total2))
    self.assertEqual(float(z), float(z2))

  def test_log_line(self):
    scalars = {"learning/lm_loss": 1.5, "learning/z_loss": 0.5, "learning/moe_lb_loss": 0.01,
               "learning/total_weights": 100.0, "learning/ce_text": 0.9, "learning/tokens_text": 84.0,
               "learning/ce_img": 1.5, "learning/tokens_img": 16.0}
    with mock.patch.object(pre_train.max_logging, "log") as log:
      pre_train._log_loss_parts_line(0, scalars)  # pylint: disable=protected-access
    self.assertEqual(
        log.call_args[0][0],
        "LOSS-PARTS step=1 ce=1.000000 moe_lb=0.010000 z=0.500000 tokens=100"
        " ce_text=0.900000 tokens_text=84 ce_img=1.500000 tokens_img=16",
    )
    with mock.patch.object(pre_train.max_logging, "log") as log:
      pre_train._log_loss_parts_line(0, {k: v for k, v in scalars.items() if "_text" not in k and "_img" not in k})  # pylint: disable=protected-access
    self.assertEqual(log.call_args[0][0], "LOSS-PARTS step=1 ce=1.000000 moe_lb=0.010000 z=0.500000 tokens=100")


if __name__ == "__main__":
  unittest.main()
