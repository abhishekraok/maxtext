"""Pre-packed batches from another trainer (MAXTEXT_PREPACKED_DIR), for same-batch parity runs.

One file per step, `<dir>/step{k:05d}.npz` (or `step{k:04d}.npz`), holding the global batch's rows in order (row r = OLMo-core rank r):
  input_ids  [R, L] int   token ids (padding included)
  label_mask [R, L] bool  OLMo-core convention: label_mask[i] = token i is a trained label (its loss is
                          predicted at position i - 1); position 0 is never used as a label.
  doc_lens   [R] JSON-string array of per-row document lengths, or doc_lens_flat + doc_lens_offsets [R + 1].
Converted to MaxText's SFT fields: inputs = input_ids; targets[i] = input_ids[i + 1]; target i trained iff
label_mask[i + 1]; segment ids 1..k and positions 0..n-1 per document from doc_lens (0 on padding);
targets_segmentation = the source position's segment id where trained, else 0 (the loss mask is != 0, so a label
crossing a document boundary is trained exactly when OLMo-core trains it).

The R rows fill global rows 0..R-1 of the loaded batch (config.global_batch_size_to_load rows, laid out on
P(mesh.axis_names) like MultiHostDataLoadIterator); the remaining rows are padding. The loss trains rows
[:micro_batch_size_to_train_on], which must equal R. Every host reads the whole (small) file and serves its own shards.
"""

import io
import json
import os

import numpy as np
import jax
from jax.sharding import NamedSharding, PartitionSpec
from etils import epath

from maxtext.utils import max_logging


def convert_rows(z, rows_total, seq_len):
  """OLMo-core packed rows -> dict of [rows_total, seq_len] int32 MaxText SFT arrays (numpy)."""
  input_ids = np.asarray(z["input_ids"]).astype(np.int64)
  label_mask = np.asarray(z["label_mask"]).astype(bool)
  n_rows, length = input_ids.shape
  if length != seq_len:
    raise ValueError(f"prepacked rows have length {length}, config max_target_length is {seq_len}")
  if n_rows > rows_total:
    raise ValueError(f"prepacked file has {n_rows} rows, the loaded batch has {rows_total}")
  if "doc_lens" in z.files:
    doc_lens = [json.loads(str(s)) for s in z["doc_lens"]]
  else:
    flat, off = np.asarray(z["doc_lens_flat"]), np.asarray(z["doc_lens_offsets"])
    doc_lens = [flat[off[r] : off[r + 1]].tolist() for r in range(n_rows)]
  out = {
      k: np.zeros((rows_total, seq_len), np.int32)
      for k in ("inputs", "inputs_position", "inputs_segmentation", "targets", "targets_position", "targets_segmentation")
  }
  for r in range(n_rows):
    ids = input_ids[r]
    seg = np.zeros(seq_len, np.int32)
    pos = np.zeros(seq_len, np.int32)
    s = 0
    for k, n in enumerate(doc_lens[r], start=1):
      n = int(n)
      if s + n > seq_len:
        raise ValueError(f"row {r}: doc_lens sum past the row length")
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
  return out, int(label_mask[:, 1:].sum())


class PrepackedIterator:
  """Yields global batches (dict of jax.Array) read from MAXTEXT_PREPACKED_DIR, one file per step."""

  def __init__(self, config, mesh):
    self.dir = epath.Path(os.environ["MAXTEXT_PREPACKED_DIR"])
    self.step = int(os.environ.get("MAXTEXT_PREPACKED_START", "1"))
    self.seq_len = int(config.max_target_length)
    self.rows_total = int(config.global_batch_size_to_load)
    self.rows_trained = int(config.micro_batch_size_to_train_on)
    self.sharding = NamedSharding(mesh, PartitionSpec(mesh.axis_names))
    self._pool = None
    self._pending = None
    max_logging.log(
        f"PREPACKED: reading {self.dir}/step####.npz from step {self.step}; loaded rows {self.rows_total}, "
        f"trained rows {self.rows_trained}, length {self.seq_len}"
    )

  def __iter__(self):
    return self

  def _load(self, step):
    """Reads and converts one step's file (numpy only; runs in the prefetch thread). None at end of data."""
    f = self.dir / f"step{step:05d}.npz"  # full export: 5-digit names
    if not f.exists():
      f = self.dir / f"step{step:04d}.npz"  # first export (steps 1-100): 4-digit names
    if not f.exists():
      return None
    z = np.load(io.BytesIO(f.read_bytes()), allow_pickle=False)
    arrays, n_labels = convert_rows(z, self.rows_total, self.seq_len)
    return arrays, n_labels, int(np.asarray(z["input_ids"]).shape[0]), str(f)

  def _prefetch(self, step):
    if self._pool is None:
      import concurrent.futures  # pylint: disable=import-outside-toplevel
      self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    self._pending = (step, self._pool.submit(self._load, step))

  def __next__(self):
    # The next step's file is read and converted in a background thread while the current step trains
    # (reading it synchronously cost ~0.2 s/step on 8x8x8).
    if getattr(self, "_pending", None) is None or self._pending[0] != self.step:
      self._prefetch(self.step)
    loaded = self._pending[1].result()
    self._pending = None
    if loaded is None:
      max_logging.log(f"PREPACKED: no file for step {self.step}; end of data")
      raise StopIteration
    arrays, n_labels, n_rows, fname = loaded
    if n_rows != self.rows_trained:
      raise ValueError(f"prepacked file has {n_rows} rows but the loss trains {self.rows_trained}")
    if jax.process_index() == 0:
      max_logging.log(
          f"PREPACKED step={self.step} rows={n_rows} labels={n_labels} "
          f"trained_targets={int((arrays['targets_segmentation'] != 0).sum())}"
      )
    self.step += 1
    self._prefetch(self.step)
    shape = (self.rows_total, self.seq_len)
    return {
        k: jax.make_array_from_callback(shape, self.sharding, lambda idx, a=a: a[idx]) for k, a in arrays.items()
    }

  def skip_steps(self, n: int) -> None:
    """Resume: advance past n already-trained steps without reading their files."""
    self.step += int(n)
    max_logging.log(f"PREPACKED: resume skipped {n} steps; next file step{self.step:04d}.npz")

  def close(self):
    pass
