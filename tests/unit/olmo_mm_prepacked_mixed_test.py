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

"""olmo_mm OLMO_MM_TEXT_NPZ_DIR: prepacked text rows in file order, image rows in scheduled slots.

Set PREPACKED_REAL_DIR to a local directory of real step#####.npz files to also check them against the text
team's prepacked reader (``prepacked.convert_rows`` of their SFT port, copied below).
"""

import io
import json
import os
import tempfile
import unittest

import numpy as np

from maxtext.input_pipeline import olmo_mm_data_processing as mm

L = 64
CONVERT = dict(
    max_target_length=L,
    max_crops_per_seq=2,
    max_pooled_patches_per_seq=3,
    n_patches_per_crop=4,
    patch_dim=5,
    pool_size=4,
    dtype_mm=np.float32,
    reset_positions_on_segment=True,
    padding_input_token_id=-1,
    use_loss_weights=True,
    include_router_token_mask=True,
)


def reference_convert_rows(z, rows_total, seq_len):
  """The text team's reader (sft_port.patch, src/maxtext/input_pipeline/prepacked.py convert_rows), verbatim logic."""
  input_ids = np.asarray(z["input_ids"]).astype(np.int64)
  label_mask = np.asarray(z["label_mask"]).astype(bool)
  n_rows, _ = input_ids.shape
  if "doc_lens" in z:
    doc_lens = [json.loads(str(s)) for s in z["doc_lens"]]
  else:
    flat, off = np.asarray(z["doc_lens_flat"]), np.asarray(z["doc_lens_offsets"])
    doc_lens = [flat[off[r] : off[r + 1]].tolist() for r in range(n_rows)]
  keys = ("inputs", "inputs_position", "inputs_segmentation", "targets", "targets_position", "targets_segmentation")
  out = {k: np.zeros((rows_total, seq_len), np.int32) for k in keys}
  for r in range(n_rows):
    ids = input_ids[r]
    seg = np.zeros(seq_len, np.int32)
    pos = np.zeros(seq_len, np.int32)
    s = 0
    for k, n in enumerate(doc_lens[r], start=1):
      seg[s : s + n] = k
      pos[s : s + n] = np.arange(n)
      s += n
    trained = np.zeros(seq_len, bool)
    trained[:-1] = label_mask[r, 1:]
    out["inputs"][r] = ids
    out["targets"][r, :-1] = ids[1:]
    out["inputs_segmentation"][r] = seg
    out["inputs_position"][r] = pos
    out["targets_position"][r] = pos
    out["targets_segmentation"][r] = np.where(trained, np.maximum(seg, 1), 0)
  return out


def assert_same_training_signal(test, ours, ref, used):
  """Same tokens, labels, trained positions, positions and document partition on the row's real tokens."""
  sl = slice(0, used)
  np.testing.assert_array_equal(ours["inputs"][sl], ref["inputs"][sl])
  trained = ref["targets_segmentation"] != 0
  np.testing.assert_array_equal(ours["targets_segmentation"] != 0, trained)
  np.testing.assert_array_equal(ours["targets"][trained], ref["targets"][trained])
  np.testing.assert_array_equal(ours["inputs_position"][sl], ref["inputs_position"][sl])
  seg_o, seg_r = ours["inputs_segmentation"][sl], ref["inputs_segmentation"][sl]
  np.testing.assert_array_equal(np.diff(seg_o) != 0, np.diff(seg_r) != 0)
  np.testing.assert_array_equal(ours["loss_weights"], trained.astype(np.float32))
  np.testing.assert_array_equal(ours["router_token_mask"][:used], 1)
  test.assertFalse(ours["router_token_mask"][used:].any())


def synthetic_file(rng, rows, step):
  ids = rng.integers(1, 1000, size=(rows, L)).astype(np.int32)
  mask = np.zeros((rows, L), bool)
  flat, off = [], [0]
  for r in range(rows):
    ids[r, 0] = 10_000 * step + r  # identifies (file, row)
    used = int(rng.integers(L // 2, L + 1))
    lens, s = [], 0
    while s < used:
      n = int(min(used - s, rng.integers(3, 20)))
      lens.append(n)
      mask[r, s + 1 : s + n] = rng.random(n - 1) < 0.7  # never a document's first token
      s += n
    ids[r, used:] = 0
    flat += lens
    off.append(len(flat))
  return {"input_ids": ids, "label_mask": mask, "doc_lens_flat": np.array(flat, np.int32),
          "doc_lens_offsets": np.array(off, np.int32)}


class FakeImageLoader:
  """Rank r's image sequences: 1-row MaxText batches whose first input token is 900000 + 1000 r + k."""

  def __init__(self, rank, consumed):
    self.rank, self.k = rank, consumed

  def __iter__(self):
    return self

  def __next__(self):
    self.k += 1
    batch = {
        "input_ids": np.full((1, L), 7, np.int64),
        "labels": np.full((1, L), 7, np.int64),
        "loss_masks": np.ones((1, L), np.float32),
        "document_ids": np.zeros((1, L), np.int64),
        "images": np.ones((1, 1, 4, 5), np.float32),
        "pooled_patches_idx": np.zeros((1, 1, 4), np.int64),
    }
    batch["input_ids"][0, 0] = 900_000 + 1000 * self.rank + self.k
    return mm.olmo_core_batch_to_maxtext_batch(batch, **CONVERT)


class PrepackedMixedTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    rng = np.random.default_rng(0)
    self.rows = 4
    self.files = {}
    for step in range(1, 31):
      f = synthetic_file(rng, self.rows, step)
      self.files[step] = f
      np.savez(os.path.join(self.tmp.name, f"step{step:05d}.npz"), **f)

  def tearDown(self):
    self.tmp.cleanup()

  def test_schedule(self):
    s = mm.ImageSlotSchedule(16, 9632, 24896)
    self.assertEqual(s.text_rows, 129216)
    flags = [s.is_image(g) for g in range(s.total)]
    self.assertEqual(sum(flags), 24896)
    text = [s.text_index(g) for g in range(s.total) if not flags[g]]
    self.assertEqual(text, list(range(129216)))  # every text row once, in order
    per_row = [sum(flags[r::16]) for r in range(16)]
    self.assertLessEqual(max(per_row) - min(per_row), 0.05 * np.mean(per_row))  # ranks share the image load
    for row, step in ((0, 1), (3, 100), (15, 9633)):
      self.assertEqual(s.image_count_before_step(row, step), sum(flags[row : (step - 1) * 16 : 16]))
    z = mm.ImageSlotSchedule(16, 8076, 0)
    self.assertFalse(any(z.is_image(g) for g in range(z.total)))

  def test_text_rows_match_text_team_reader(self):
    for step in (1, 2, 3):
      f = self.files[step]
      ref = reference_convert_rows(f, self.rows, L)
      for r in range(self.rows):
        used = int(f["doc_lens_flat"][f["doc_lens_offsets"][r] : f["doc_lens_offsets"][r + 1]].sum())
        b = mm.prepacked_row_to_olmo_core_batch(f["input_ids"][r], f["label_mask"][r],
                                                f["doc_lens_flat"][f["doc_lens_offsets"][r]:f["doc_lens_offsets"][r + 1]])
        ours = {k: v[0] for k, v in mm.olmo_core_batch_to_maxtext_batch(b, **CONVERT).items()}
        assert_same_training_signal(self, ours, {k: v[r] for k, v in ref.items()}, used)

  def _loader(self, image_rows, steps, local_rows, next_step=1):
    text = mm.PrepackedTextRows(self.tmp.name)
    return mm._PrepackedMixedHostLoader(  # pylint: disable=protected-access
        local_rows=local_rows, cutoff=self.rows, text=text,
        schedule=mm.ImageSlotSchedule(self.rows, steps, image_rows),
        image_loader_for_rank=FakeImageLoader, next_step=next_step, convert_kwargs=CONVERT, host=0, hash_steps=-1)

  def test_text_only_arm_is_the_file_stream(self):
    local_rows = [2, 0, 9, 3]  # row 9 is not trained
    loader = self._loader(0, 30, local_rows)
    for step in range(1, 31):
      batch = next(loader)
      for i, row in enumerate(local_rows):
        if row >= self.rows:
          self.assertFalse(batch["inputs"][i].any())
          self.assertTrue((batch["pooled_patches_idx"][i] == -1).all())
        else:
          self.assertEqual(batch["inputs"][i, 0], 10_000 * step + row)
    with self.assertRaises(StopIteration):
      next(loader)

  def test_image_arm_interleaves_and_resumes(self):
    steps, n_img = 30, 24  # 120 slots: 96 text rows (files 1..24), 24 image rows
    local_rows = [0, 1, 2, 3]
    full = [next(self._loader(n_img, steps, local_rows)) for _ in range(1)]
    loader = self._loader(n_img, steps, local_rows)
    seen_text, seen_img, batches = [], 0, []
    for _ in range(steps):
      b = next(loader)
      batches.append(b)
      for i in range(4):
        tok = int(b["inputs"][i, 0])
        if tok >= 900_000:
          seen_img += 1
        else:
          seen_text.append(tok)
    self.assertEqual(seen_img, n_img)
    expected = [10_000 * (t // 4 + 1) + t % 4 for t in range(96)]
    self.assertEqual(seen_text, expected)
    np.testing.assert_array_equal(full[0]["inputs"], batches[0]["inputs"])
    # Resume at step 11: the same batches as the uninterrupted run, image ranks fast-forwarded.
    resumed = self._loader(n_img, steps, local_rows, next_step=11)
    for k in range(10, steps):
      b = next(resumed)
      for key in b:
        np.testing.assert_array_equal(b[key], batches[k][key])

  def test_saved_state_resume_equals_uninterrupted_and_checks_fall_back(self):
    """H269 saved-state resume: a checkpoint of update u records each image rank's loader state as of u (not the
    host's read-ahead); restoring it gives the uninterrupted batches; a bad or missing file falls back (None)."""
    import json as _json  # pylint: disable=import-outside-toplevel

    steps, n_img, local_rows = 30, 24, [0, 1, 2, 3]

    class StatefulImageLoader(FakeImageLoader):

      def __init__(self, rank, consumed, loader_state=None):
        super().__init__(rank, consumed)
        if loader_state is not None:
          self.k = loader_state["k"]  # a real loader restores its OLMo-core state instead of redrawing

      def loader_state(self):
        return {"k": self.k, "tuple": (self.rank, self.k)}

    def loader(next_step, rank_states=None):
      return mm._PrepackedMixedHostLoader(  # pylint: disable=protected-access
          local_rows=local_rows, cutoff=self.rows, text=mm.PrepackedTextRows(self.tmp.name),
          schedule=mm.ImageSlotSchedule(self.rows, steps, n_img), image_loader_for_rank=StatefulImageLoader,
          next_step=next_step, convert_kwargs=CONVERT, host=0, hash_steps=-1, rank_states=rank_states)

    sch = mm.ImageSlotSchedule(self.rows, steps, n_img)
    reference = loader(1)
    batches = [next(reference) for _ in range(steps)]
    with tempfile.TemporaryDirectory() as base:
      for u in (5, 12, 20):
        run = loader(1)
        for _ in range(u + 2):  # the host has read two steps past the checkpoint's update
          next(run)
        path = mm.save_active_loader_state(base, u, host=0)
        binding = {"rows": self.rows, "steps": steps, "image_rows": n_img, "image_config_sha256": "",
                   "local_rows": local_rows}
        consumed = {r: sch.image_count_before_step(r, u + 1) for r in local_rows}
        states, why = mm.load_loader_state(base, u, 0, binding, consumed)
        self.assertEqual(why, "")
        self.assertEqual(states[0]["tuple"], (0, consumed[0]))  # pickled: the tuple survives exactly
        resumed = loader(u + 1, rank_states=states)
        for k in range(u, steps):
          b = next(resumed)
          for key in b:
            np.testing.assert_array_equal(b[key], batches[k][key])
        # Checks: wrong update, binding, counts, tampered payload, missing file -> fallback with a reason.
        self.assertIsNone(mm.load_loader_state(base, u + 1, 0, binding, consumed)[0])
        self.assertIsNone(mm.load_loader_state(base, u, 0, dict(binding, image_rows=n_img + 1), consumed)[0])
        self.assertIsNone(mm.load_loader_state(base, u, 0, binding, dict(consumed, **{0: consumed[0] + 1}))[0])
        doc = _json.loads(open(path).read())
        doc["payload"]["ranks"]["0"]["consumed"] += 0
        doc["payload"]["update"] = u + 100
        open(path, "w").write(_json.dumps(doc))
        self.assertIn("sha256", mm.load_loader_state(base, u, 0, binding, consumed)[1])

  @unittest.skipUnless(os.environ.get("PREPACKED_REAL_DIR"), "set PREPACKED_REAL_DIR")
  def test_real_files_match_text_team_reader(self):
    root = os.environ["PREPACKED_REAL_DIR"]
    conv = dict(CONVERT, max_target_length=65536)
    for name in sorted(os.listdir(root)):
      if not (name.startswith("step") and name.endswith(".npz")):
        continue
      with open(os.path.join(root, name), "rb") as fh:
        z = dict(np.load(io.BytesIO(fh.read())))
      rows = z["input_ids"].shape[0]
      ref = reference_convert_rows(z, rows, 65536)
      for r in range(rows):
        dl = z["doc_lens_flat"][z["doc_lens_offsets"][r] : z["doc_lens_offsets"][r + 1]]
        b = mm.prepacked_row_to_olmo_core_batch(z["input_ids"][r], z["label_mask"][r], dl)
        ours = {k: v[0] for k, v in mm.olmo_core_batch_to_maxtext_batch(b, **conv).items()}
        assert_same_training_signal(self, ours, {k: v[r] for k, v in ref.items()}, int(dl.sum()))


if __name__ == "__main__":
  unittest.main()
