# Copyright 2023-2026 Google LLC
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

"""olmo35_small_mm_sft_64k.yml resolves to what it says, and olmo_mm_keep_sources (its images-off arm) cuts the
OLMo-core data config to the named sources.

    JAX_PLATFORMS=cpu pytest tests/unit/olmo35_small_mm_sft_config_test.py
"""

import json
import os
import tempfile
import unittest

import yaml

from maxtext.configs import pyconfig
from maxtext.input_pipeline import olmo_mm_data_processing as omm
from maxtext.utils.globals import MAXTEXT_PKG_DIR

CONFIG = os.path.join(MAXTEXT_PKG_DIR, "configs", "olmo", "olmo35_small_mm_sft_64k.yml")


def _norm(value):
  value = getattr(value, "value", value)  # enums
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return float(value)
  if isinstance(value, (list, tuple)):
    return [_norm(v) for v in value]
  return value


class OLMo35SmallMmSftConfigTest(unittest.TestCase):

  def _resolve(self, *extra):
    return pyconfig.initialize(
        ["", CONFIG, "run_name=olmo35_small_mm_sft_config_test", "skip_jax_distributed_system=True", *extra]
    )

  def test_every_setting_resolves_to_the_files_value(self):
    with open(CONFIG, encoding="utf-8") as f:
      want = yaml.safe_load(f)
    cfg = self._resolve()
    shadowed = {
        k: (v, getattr(cfg, k)) for k, v in want.items() if k != "base_config" and _norm(getattr(cfg, k)) != _norm(v)
    }
    self.assertEqual(shadowed, {}, "settings replaced at resolution (by the model config?): {key: (file, resolved)}")

  def test_schedule_and_batch(self):
    cfg = self._resolve()
    steps = cfg.learning_rate_schedule_steps
    self.assertEqual(steps, 9632)
    self.assertEqual(int(steps * cfg.warmup_steps_fraction), int(9632 * 0.03))  # 288
    self.assertEqual(int(steps * cfg.warmup_steps_fraction) + int(steps * cfg.wsd_decay_steps_fraction), steps)
    # 16 rows of 65,536 on 512 chips (data 4 x fsdp 4 x expert 32).
    self.assertEqual(cfg.ici_data_parallelism * cfg.ici_fsdp_parallelism * cfg.ici_expert_parallelism, 512)
    self.assertEqual(round(cfg.per_device_batch_size * 512), 16)
    self.assertTrue(cfg.scan_layers)
    self.assertTrue(cfg.use_multimodal)

  def test_text_arm_schedule(self):
    cfg = self._resolve(
        "steps=8076",
        "learning_rate_schedule_steps=8076",
        "warmup_steps_fraction=0.02996532937097573",
        "wsd_decay_steps_fraction=0.9700346706290243",
    )
    self.assertEqual(int(8076 * cfg.warmup_steps_fraction), 242)
    self.assertEqual(int(8076 * cfg.warmup_steps_fraction) + int(8076 * cfg.wsd_decay_steps_fraction), 8076)

  def test_images_off_arm_is_the_same_config(self):
    on = self._resolve()
    off = self._resolve("olmo_mm_keep_sources=[text_sft]")
    self.assertEqual(list(off.olmo_mm_keep_sources), ["text_sft"])
    keys_on, keys_off = on.get_keys(), off.get_keys()
    differ = sorted(k for k in keys_on if k not in ("olmo_mm_keep_sources", "run_name") and keys_on[k] != keys_off[k])
    self.assertEqual(differ, [])


class KeepSourcesTest(unittest.TestCase):

  def test_write_source_subset_config(self):
    cfg = {
        "dataset": {
            "sources": {"text_sft": {"a": 1}, "pixmo_cap": {"b": 2}, "doc_qa": {"c": 3}},
            "target_loss_mass": {"text_sft": 0.98, "pixmo_cap": 0.01, "doc_qa": 0.01},
            "mean_loss_weight": {"text_sft": 5000.0, "pixmo_cap": 200.0, "doc_qa": 6.0},
            "tokenizer": {"x": 1},
        },
        "data_loader": {"seed": 7, "pack_max_crops": 512},
    }
    with tempfile.TemporaryDirectory() as tmp:
      src = os.path.join(tmp, "config.json")
      with open(src, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
      out = omm.write_source_subset_config(src, ["text_sft"], os.path.join(tmp, "work"))
      with open(out, encoding="utf-8") as f:
        got = json.load(f)
      self.assertEqual(list(got["dataset"]["sources"]), ["text_sft"])
      self.assertEqual(got["dataset"]["target_loss_mass"], {"text_sft": 0.98})
      self.assertEqual(got["dataset"]["mean_loss_weight"], {"text_sft": 5000.0})
      self.assertEqual(got["dataset"]["tokenizer"], cfg["dataset"]["tokenizer"])
      self.assertEqual(got["data_loader"], cfg["data_loader"])
      # Order is the config's, not the argument's.
      out2 = omm.write_source_subset_config(src, ["doc_qa", "text_sft"], os.path.join(tmp, "work"))
      with open(out2, encoding="utf-8") as f:
        self.assertEqual(list(json.load(f)["dataset"]["sources"]), ["text_sft", "doc_qa"])
      with self.assertRaises(ValueError):
        omm.write_source_subset_config(src, ["nope"], os.path.join(tmp, "work"))


if __name__ == "__main__":
  unittest.main()
