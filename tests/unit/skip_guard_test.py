"""``skip_guard``: skipped-step counts in the skip-step optimizer, the step's guard vector, and the host check.

    JAX_PLATFORMS=cpu pytest tests/unit/skip_guard_test.py
"""

import os
import tempfile
import types
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np
import optax

from maxtext.configs import types as config_types
from maxtext.optimizers.optimizers import skip_step_on_spikes
from maxtext.trainers.pre_train import train
from maxtext.trainers.pre_train.train import (
    SKIP_GUARD_FIELDS,
    image_emb_nonfinite_flag,
    maybe_skip_guard,
    skip_guard_vector,
)


def _cfg(**kw):
  return types.SimpleNamespace(**{"skip_guard": True, "use_multimodal": True, "skip_guard_max_skips": 3,
                                  "base_output_directory": _cfg.out, "run_name": "r", **kw})


_cfg.out = "/tmp/skip_guard_test_unset"


def _vec(**kw):
  d = {k: 0.0 for k in SKIP_GUARD_FIELDS}
  d.update(loss=1.0, raw_grad_norm=0.5)
  d.update(kw)
  return {"skip_guard": jnp.asarray([d[k] for k in SKIP_GUARD_FIELDS], jnp.float32)}


class OptimizerCountsTest(unittest.TestCase):
  """The counters live in the saved optimizer state and skipped steps leave params and Adam state untouched."""

  def setUp(self):
    self.opt = skip_step_on_spikes(optax.adam(0.1), interval=4, scaling_factor=6.0, track_skips=True)
    self.params = {"w": jnp.ones(3)}
    self.state = self.opt.init(self.params)
    self.g = {"w": jnp.full(3, 0.1)}

  def _step(self, loss, gn=0.1):
    upd, self.state = self.opt.update(self.g, self.state, self.params, loss=jnp.float32(loss), grad_norm=jnp.float32(gn))
    return upd

  def test_counts_by_kind_and_consecutive(self):
    self._step(1.0)
    self.assertEqual(int(self.state["consecutive_skips"]), 0)
    inner_before = self.state["inner_state"]
    upd = self._step(np.nan)
    np.testing.assert_array_equal(np.asarray(upd["w"]), 0.0)  # nothing applied
    for a, b in zip(jax_leaves(inner_before), jax_leaves(self.state["inner_state"])):
      if a.dtype != jnp.int32:  # schedule counters follow the step; Adam moments do not move
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    self.assertEqual((int(self.state["skips_nonfinite"]), int(self.state["consecutive_skips"])), (1, 1))
    self._step(1.0, gn=np.inf)
    self.assertEqual((int(self.state["skips_nonfinite"]), int(self.state["consecutive_skips"])), (2, 2))
    self._step(1.0)
    self.assertEqual((int(self.state["skips_nonfinite"]), int(self.state["consecutive_skips"])), (2, 0))
    self.assertEqual(int(self.state["skips_spike"]), 0)

  def test_spike_counted_as_spike(self):
    for loss in [1.0, 1.01, 0.99, 1.0, 1.0]:
      self._step(loss)
    self._step(1e6)
    self.assertEqual((int(self.state["skips_spike"]), int(self.state["skips_nonfinite"])), (1, 0))
    self.assertTrue(bool(self.state["is_skipped"]))

  def test_untracked_state_unchanged(self):
    st = skip_step_on_spikes(optax.adam(0.1), interval=4, scaling_factor=6.0).init(self.params)
    self.assertNotIn("skips_nonfinite", st)


def jax_leaves(t):
  import jax  # pylint: disable=import-outside-toplevel
  return jax.tree_util.tree_leaves(t)


class VectorTest(unittest.TestCase):

  def test_fields(self):
    st = {"is_skipped": jnp.bool_(True), "skips_nonfinite": jnp.int32(2), "skips_spike": jnp.int32(1),
          "consecutive_skips": jnp.int32(1)}
    v = dict(zip(SKIP_GUARD_FIELDS, np.asarray(skip_guard_vector(st, jnp.int32(1), jnp.float32(np.nan), jnp.float32(3)))))
    self.assertEqual((v["skipped"], v["image_emb_nonfinite"], v["loss_nonfinite"], v["grad_norm_nonfinite"]),
                     (1, 1, 1, 0))
    self.assertEqual((v["skips_nonfinite"], v["skips_spike"], v["consecutive_skips"], v["raw_grad_norm"]), (2, 1, 1, 3))

  def test_image_flag_any_row_and_no_wrap(self):
    inter = lambda rows: {"nonfinite_probe_image_emb": (jnp.asarray(rows, jnp.int32),)}
    self.assertEqual(int(image_emb_nonfinite_flag(_cfg(), inter([0, 0, 2]))), 1)
    self.assertEqual(int(image_emb_nonfinite_flag(_cfg(), inter([2**31 - 1, 1]))), 1)
    self.assertEqual(int(image_emb_nonfinite_flag(_cfg(), inter([0, 0]))), 0)
    # A path sown twice keeps both sows: a bad first sow is not hidden by a clean second one.
    two = {"nonfinite_probe_image_emb": (jnp.asarray([1], jnp.int32), jnp.asarray([0], jnp.int32))}
    self.assertEqual(int(image_emb_nonfinite_flag(_cfg(), two)), 1)
    with self.assertRaisesRegex(ValueError, "vision encoder did not run"):
      image_emb_nonfinite_flag(_cfg(), {})
    self.assertEqual(int(image_emb_nonfinite_flag(_cfg(use_multimodal=False), {})), 0)


class HostCheckTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    _cfg.out = self._tmp.name
    train._SKIP_JOURNAL = {}

  def tearDown(self):
    self._tmp.cleanup()

  def _run(self, step=7, **kw):
    with mock.patch.object(train.max_logging, "log") as log:
      maybe_skip_guard(_cfg(), step, _vec(**kw))
    return [c.args[0] for c in log.call_args_list]

  def test_journal_survives_restart_and_counts(self):
    self._run(step=501, skipped=1, loss_nonfinite=1, skips_nonfinite=1, consecutive_skips=1)
    self.assertTrue(os.path.exists(os.path.join(_cfg.out, "r", "skip_guard", "step_0000501_nonfinite")))
    for st in (601, 701):  # restart: restored state has no skips, but the journal does
      self._run(step=st, skipped=1, skips_spike=1, consecutive_skips=1)
    train._SKIP_JOURNAL = None
    with mock.patch.object(train.max_logging, "log"):
      train.skip_guard_startup(_cfg())
    self.assertEqual(train._SKIP_JOURNAL, {501: "nonfinite", 601: "spike", 701: "spike"})
    with self.assertRaises(SystemExit):  # state says 1 skip; the journal holds 4 distinct steps > cap 3
      self._run(step=801, skipped=1, skips_spike=1, consecutive_skips=1)
    self.assertTrue(os.path.exists(os.path.join(_cfg.out, "r", "skip_guard", "HALT")))
    with self.assertRaises(SystemExit):  # every later start stops at the marker
      with mock.patch.object(train.max_logging, "log"):
        train.skip_guard_startup(_cfg())

  def test_replayed_skip_counted_once(self):
    self._run(step=9, skipped=1, skips_spike=1, consecutive_skips=1)
    self._run(step=9, skipped=1, skips_spike=1, consecutive_skips=1)
    self.assertEqual(train._SKIP_JOURNAL, {9: "spike"})

  def test_clean_step_silent(self):
    self.assertEqual(self._run(), [])

  def test_nonfinite_skip_logged_not_halted(self):
    lines = self._run(skipped=1, image_emb_nonfinite=1, skips_nonfinite=1, consecutive_skips=1)
    self.assertEqual(len(lines), 1)
    self.assertIn("SKIP-STEP step=7 kind=nonfinite", lines[0])
    self.assertIn("image_emb_nonfinite=1", lines[0])

  def test_spike_skip_kind(self):
    self.assertIn("kind=spike", self._run(skipped=1, skips_spike=1, consecutive_skips=1)[0])

  def test_second_consecutive_skip_halts(self):
    with self.assertRaises(SystemExit) as cm:
      self._run(skipped=1, loss_nonfinite=1, skips_nonfinite=2, consecutive_skips=2)
    self.assertEqual(cm.exception.code, 1)

  def test_cap_counts_both_kinds(self):
    self._run(skipped=1, skips_nonfinite=2, skips_spike=1, consecutive_skips=1)  # 3 = cap: runs on
    with self.assertRaises(SystemExit):
      self._run(skipped=1, skips_nonfinite=2, skips_spike=2, consecutive_skips=1)  # 4 > cap 3

  def test_cap_without_spikes(self):
    cfg = _cfg(skip_guard_cap_counts_spikes=False)
    with mock.patch.object(train.max_logging, "log"):
      for st in range(10, 20, 2):  # five spike skips, none back to back: under cap 3 when spikes do not count
        maybe_skip_guard(cfg, st, _vec(skipped=1, skips_spike=(st - 8) // 2, consecutive_skips=1))
      maybe_skip_guard(cfg, 30, _vec(skipped=1, loss_nonfinite=1, skips_nonfinite=3, skips_spike=5, consecutive_skips=1))
      with self.assertRaises(SystemExit):  # 4 non-finite > 3
        maybe_skip_guard(cfg, 40, _vec(skipped=1, loss_nonfinite=1, skips_nonfinite=4, skips_spike=5,
                                       consecutive_skips=1))
      train._SKIP_JOURNAL = {}
      with self.assertRaises(SystemExit):  # back-to-back spikes still halt
        maybe_skip_guard(cfg, 50, _vec(skipped=1, skips_spike=2, consecutive_skips=2))

  def test_missing_vector_fails_closed(self):
    with self.assertRaisesRegex(RuntimeError, "no guard vector"):
      maybe_skip_guard(_cfg(), 7, {})

  def test_off_is_a_no_op(self):
    maybe_skip_guard(_cfg(skip_guard=False), 7, {})


class ValidationTest(unittest.TestCase):

  def _ok(self, **kw):
    d = dict(skip_guard=True, skip_step_on_spikes=True, skip_guard_max_skips=32, abort_on_nan_loss=False,
             abort_on_inf_loss=False, enable_diloco=False, enable_checkpointing=True, async_checkpointing=False,
             moe_dropless_fallback=None, enable_emergency_checkpoint=False, enable_multi_tier_checkpointing=False,
             trainable_parameters_mask=[], frozen_parameters_mask=[], use_multimodal=True,
             freeze_vision_encoder_params=False)
    d.update(kw)
    return config_types.MaxTextConfig.validate_skip_guard(types.SimpleNamespace(**d))

  def test_good(self):
    self._ok()

  def test_refusals(self):
    for kw, msg in [
        ({"skip_step_on_spikes": False}, "skip_step_on_spikes"),
        ({"skip_guard_max_skips": -1}, "skip_guard_max_skips"),
        ({"abort_on_nan_loss": True}, "abort_on_nan_loss"),
        ({"abort_on_inf_loss": True}, "abort_on_inf_loss"),
        ({"enable_diloco": True}, "enable_diloco"),
        ({"async_checkpointing": True}, "async_checkpointing"),
        ({"moe_dropless_fallback": "step"}, "moe_dropless_fallback"),
        ({"enable_multi_tier_checkpointing": True}, "multi-tier"),
        ({"enable_emergency_checkpoint": True}, "emergency"),
        ({"freeze_vision_encoder_params": True}, "freezing"),
        ({"frozen_parameters_mask": ["x"]}, "freezing"),
    ]:
      with self.assertRaisesRegex(ValueError, msg):
        self._ok(**kw)

  def test_default_off(self):
    f = config_types.MaxTextConfig.model_fields
    self.assertIs(f["skip_guard"].default, False)
    self.assertEqual(f["skip_guard_max_skips"].default, -1)


if __name__ == "__main__":
  unittest.main()
