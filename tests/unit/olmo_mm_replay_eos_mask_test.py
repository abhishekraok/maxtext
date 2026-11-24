"""olmo_mm_text_replay_eos_mask: H270's replay end-token fix on prepacked TEXT rows, never on image rows.

    JAX_PLATFORMS=cpu pytest tests/unit/olmo_mm_replay_eos_mask_test.py
"""

import os
import tempfile
import unittest

import numpy as np

from maxtext.input_pipeline import olmo_mm_data_processing as mm
from tests.unit.olmo_mm_prepacked_mixed_test import CONVERT, L, reference_convert_rows

EOS, IM = mm.REPLAY_EOS_ID, mm.REPLAY_IM_START_ID


def h270_keep(inputs, inputs_segmentation, targets):
  """H270's replay_eos_keep (experiments/h270/replay_eos_mask.patch), in numpy, on MaxText-coordinate arrays."""
  seg = inputs_segmentation
  s = seg.shape[-1]
  idx = np.broadcast_to(np.arange(s), seg.shape)
  is_start = np.concatenate([np.ones_like(seg[:, :1], bool), seg[:, 1:] != seg[:, :-1]], axis=1)
  start = np.maximum.accumulate(np.where(is_start, idx, 0), axis=1)
  first = np.take_along_axis(inputs, start, axis=1)
  seg_ext = np.concatenate([seg, np.full_like(seg[:, :2], -1)], axis=1)
  nxt, nxt2 = seg_ext[:, 1 : s + 1], seg_ext[:, 2 : s + 2]
  last_target = (seg > 0) & (nxt == seg) & (nxt2 != seg)
  return ~((targets == EOS) & last_target & (first != IM))


def stream_file(rng, rows, step):
  """Rows of whole documents: chat (starts <|im_start|>), replay (plain start, trained final <|endoftext|>) and others."""
  ids = rng.integers(1, 1000, size=(rows, L)).astype(np.int32)
  mask = np.zeros((rows, L), bool)
  flat, off = [], [0]
  for r in range(rows):
    used = int(rng.integers(L // 2, L + 1))
    lens, s = [], 0
    while s < used:
      n = int(min(used - s, rng.integers(2, 16)))
      kind = rng.integers(0, 4)
      if kind == 0:  # chat
        ids[r, s] = IM
      if kind in (0, 1):  # ends in <|endoftext|>
        ids[r, s + n - 1] = EOS
      if kind == 2 and n > 2:  # <|endoftext|> inside, not at the end
        ids[r, s + 1] = EOS
      mask[r, s + 1 : s + n] = rng.random(n - 1) < 0.8
      lens.append(n)
      s += n
    ids[r, used:] = 0
    flat += lens
    off.append(len(flat))
  return {"input_ids": ids, "label_mask": mask, "doc_lens_flat": np.array(flat, np.int32),
          "doc_lens_offsets": np.array(off, np.int32)}


class ReplayEosMaskTest(unittest.TestCase):

  def test_rule_cases(self):
    ids = np.array([5, 6, EOS, IM, 7, EOS, 8, EOS, 9, 3, EOS, 0, 0])
    mask = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0], bool)
    lens = np.array([3, 3, 3, 2])  # replay (trained end), chat, end-token inside, untrained end
    new, dropped = mm.replay_eos_label_mask(ids, mask, lens)
    self.assertEqual(dropped, 1)
    np.testing.assert_array_equal(np.nonzero(mask & ~new)[0], [2])
    np.testing.assert_array_equal(mask, np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0], bool))  # input untouched

  def test_matches_h270_rule_on_random_rows(self):
    rng = np.random.default_rng(1)
    total = 0
    for step in range(1, 41):
      f = stream_file(rng, 4, step)
      ref = reference_convert_rows(f, 4, L)
      keep = h270_keep(ref["inputs"], ref["inputs_segmentation"], ref["targets"])
      want = (ref["targets_segmentation"] != 0) & keep
      for r in range(4):
        lens = f["doc_lens_flat"][f["doc_lens_offsets"][r] : f["doc_lens_offsets"][r + 1]]
        m, d = mm.replay_eos_label_mask(f["input_ids"][r], f["label_mask"][r], lens)
        out = mm.olmo_core_batch_to_maxtext_batch(mm.prepacked_row_to_olmo_core_batch(f["input_ids"][r], m, lens),
                                                  **CONVERT)
        np.testing.assert_array_equal(out["targets_segmentation"][0] != 0, want[r])
        self.assertEqual(d, int(((ref["targets_segmentation"][r] != 0) & ~keep[r]).sum()))
        total += d
    self.assertGreater(total, 20)


class FakeEosImageLoader:
  """Image rows that do NOT start with <|im_start|> and end in a trained <|endoftext|>: the rule would drop it."""

  def __init__(self, rank, consumed):
    self.rank, self.k = rank, consumed

  def __iter__(self):
    return self

  def __next__(self):
    self.k += 1
    ids = np.full((1, L), 7, np.int64)
    ids[0, 0] = 900_000 + 1000 * self.rank + self.k
    ids[0, L - 1] = EOS
    labels = np.full((1, L), -100, np.int64)
    labels[0, :-1] = ids[0, 1:]
    batch = {"input_ids": ids, "labels": labels, "loss_masks": (labels >= 0).astype(np.float32),
             "document_ids": np.zeros((1, L), np.int64), "images": np.ones((1, 1, 4, 5), np.float32),
             "pooled_patches_idx": np.zeros((1, 1, 4), np.int64)}
    return mm.olmo_core_batch_to_maxtext_batch(batch, **CONVERT)


class LoaderTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    rng = np.random.default_rng(2)
    for step in range(1, 31):
      np.savez(os.path.join(self.tmp.name, f"step{step:05d}.npz"), **stream_file(rng, 4, step))

  def tearDown(self):
    self.tmp.cleanup()

  def _batches(self, on, steps=30, n_img=24):
    loader = mm._PrepackedMixedHostLoader(  # pylint: disable=protected-access
        local_rows=[0, 1, 2, 3], cutoff=4, text=mm.PrepackedTextRows(self.tmp.name),
        schedule=mm.ImageSlotSchedule(4, steps, n_img), image_loader_for_rank=FakeEosImageLoader, next_step=1,
        convert_kwargs=CONVERT, host=0, hash_steps=-1, replay_eos_mask=on)
    return [next(loader) for _ in range(steps)]

  def test_text_rows_change_image_rows_never(self):
    off, on = self._batches(False), self._batches(True)
    text_drops, image_rows = 0, 0
    for b0, b1 in zip(off, on):
      for i in range(4):
        for key in b0:
          if key not in ("targets", "targets_segmentation", "loss_weights"):
            np.testing.assert_array_equal(b0[key][i], b1[key][i])  # inputs, positions, images: unchanged
        kept = b1["targets_segmentation"][i] != 0
        np.testing.assert_array_equal(b0["targets"][i][kept], b1["targets"][i][kept])  # an untrained target reads 0
        diff = int(((b0["targets_segmentation"][i] != 0) & (b1["targets_segmentation"][i] == 0)).sum())
        self.assertFalse(((b0["targets_segmentation"][i] == 0) & (b1["targets_segmentation"][i] != 0)).any())
        if int(b0["inputs"][i, 0]) >= 900_000:
          image_rows += 1
          self.assertEqual(diff, 0)  # the image row's final <|endoftext|> target stays trained
          self.assertNotEqual(int(b1["targets_segmentation"][i, L - 2]), 0)
        else:
          text_drops += diff
    self.assertEqual(image_rows, 24)
    self.assertGreater(text_drops, 10)

  def test_off_by_default(self):
    self.assertIs(mm.prepacked_row_to_olmo_core_batch.__module__, mm.__name__)
    from maxtext.configs import types as config_types  # pylint: disable=import-outside-toplevel
    self.assertIs(config_types.MaxTextConfig.model_fields["olmo_mm_text_replay_eos_mask"].default, False)


if __name__ == "__main__":
  unittest.main()
