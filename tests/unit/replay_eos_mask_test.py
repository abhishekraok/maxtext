# Copyright 2023–2026 Google LLC
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
"""Tests for the replay end-token fix (`replay_eos_keep` in trainers/pre_train/train.py).

The function is read from train.py's source and run on its own, so the test needs only jax and numpy, not the
trainer's imports. Batches use the SFT loaders' layout: targets[i] is input[i+1], and target i is trained iff
targets_segmentation[i] != 0.
"""

import ast
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

EOS, IM_START, PAD = 100257, 100264, 100277
TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / "src" / "maxtext" / "trainers" / "pre_train" / "train.py"


def _load_keep():
  """replay_eos_keep and the two token ids it reads, taken from train.py's source."""
  source = TRAIN_PY.read_text()
  wanted = {"replay_eos_keep", "_EOS_ID", "_IM_START_ID"}
  namespace = {"jax": jax, "jnp": jnp}
  for node in ast.parse(source).body:
    names = set()
    if isinstance(node, ast.FunctionDef):
      names = {node.name}
    elif isinstance(node, ast.Assign):
      for target in node.targets:
        names |= {elt.id for elt in ast.walk(target) if isinstance(elt, ast.Name)}
    if names & wanted:
      exec(ast.get_source_segment(source, node), namespace)  # pylint: disable=exec-used
  return jax.jit(namespace["replay_eos_keep"])


KEEP = _load_keep()


def _batch(rows, seq):
  """rows: per row, a list of documents (lists of token ids). Every token after a document's first is trained."""
  inputs = np.full((len(rows), seq), PAD, np.int32)
  targets = np.full((len(rows), seq), PAD, np.int32)
  segmentation = np.zeros((len(rows), seq), np.int32)
  trained = np.zeros((len(rows), seq), np.int32)
  for r, documents in enumerate(rows):
    at = 0
    for number, tokens in enumerate(documents, 1):
      n = len(tokens)
      inputs[r, at : at + n] = tokens
      segmentation[r, at : at + n] = number
      targets[r, at : at + n - 1] = tokens[1:]
      trained[r, at : at + n - 1] = number
      at += n
  return inputs, segmentation, targets, trained


def _dropped(rows, seq):
  inputs, segmentation, targets, trained = _batch(rows, seq)
  keep = np.asarray(KEEP(jnp.asarray(inputs), jnp.asarray(segmentation), jnp.asarray(targets)))
  return sorted((int(r), int(i)) for r, i in zip(*np.nonzero(~keep & (trained != 0))))


class ReplayEosMaskTest(unittest.TestCase):

  def test_chat_stop_is_kept(self):
    chat = [IM_START, 5, 6, EOS, IM_START, 7, 8, EOS]  # two turns, both stops trained
    self.assertEqual(_dropped([[chat]], 16), [])

  def test_replay_document_loses_only_its_final_stop(self):
    document = [11, 12, EOS, 13, 14, EOS]  # a stop inside the document stays trained
    self.assertEqual(_dropped([[document]], 16), [(0, 4)])

  def test_one_drop_per_replay_document_in_a_packed_row(self):
    chat = [IM_START, 5, EOS]
    first, second = [11, 12, 13, EOS], [21, 22, EOS]
    rows = [[chat, first, chat, second], [second, chat]]
    # row 0: chat 0-2, first 3-6 (final target at 5), chat 7-9, second 10-12 (final target at 11)
    self.assertEqual(_dropped(rows, 16), [(0, 5), (0, 11), (1, 1)])

  def test_document_that_fills_the_row(self):
    document = [11, 12, 13, 14, 15, 16, 17, EOS]
    self.assertEqual(_dropped([[document]], 8), [(0, 6)])

  def test_document_without_a_final_stop_drops_nothing(self):
    self.assertEqual(_dropped([[[11, 12, 13, 14]]], 8), [])

  def test_padding_only_row(self):
    self.assertEqual(_dropped([[]], 8), [])

  def test_one_token_document(self):
    self.assertEqual(_dropped([[[EOS], [IM_START, 5, EOS]]], 8), [])


if __name__ == "__main__":
  unittest.main()
