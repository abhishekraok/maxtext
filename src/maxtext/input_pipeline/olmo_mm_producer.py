"""Batch producer process for ``dataset_type=olmo_mm`` (see ``_ProducerNpzHostLoader``).

Runs the live ``OLMo-core`` ``MixtureDataLoader`` for one host's loader ranks in a process of its own, with
process workers, and writes each host batch as a compact ``batch_NNNNNNN.npz`` into a local directory that the
training process reads. Started by ``_ProducerNpzHostLoader`` with ``python -m`` (fork and exec), so it never
inherits the TPU runtime, which is what rules process workers out inside the training process: OLMo-core's
example pool forks the calling process, and a fork after JAX has initialized the TPU deadlocks.

Why: at 256K each loading host builds four sequences a step from raw sources (image decode and resize, PDF
rendering, tokenization), work that threads cannot spread over the host's cores (the GIL). One host took 33 to
44 s a batch with 8 threads and about 2 s with 16 process workers a rank (2026-11-21).

The loader runs one producer per rank (see _ProducerNpzHostLoader), so a child inherits one rank's state only.
The ranks are started one at a time on this thread: each start forks OLMo-core's batch producer process, and
``filelock`` refuses a fork while another thread is inside a lock (the other ranks starting, in a thread pool).
After that the ranks are read in parallel threads; no further forks happen in this process.

Usage: python -m maxtext.input_pipeline.olmo_mm_producer --spec /path/spec.json
The spec (written by the loader) holds ``out_dir``, ``ranks``, ``ahead``, ``start_step`` and ``live_kwargs``,
the keyword arguments of ``_LiveMixtureHostLoader`` without ``shard_index``.
"""

import argparse
import concurrent.futures
import gc
import json
import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position

from maxtext.input_pipeline import olmo_mm_data_processing as mm  # noqa: E402  pylint: disable=wrong-import-position

BATCH_FILE = "batch_{step:07d}.npz"


def _log(msg: str) -> None:
  print(f"[olmo_mm_producer pid={os.getpid()}] {msg}", flush=True)


def _concat(parts):
  if any(set(p) != set(parts[0]) for p in parts):
    raise ValueError("rank batches have different fields")
  return {name: np.concatenate([p[name] for p in parts], axis=0) for name in parts[0]}


def _pending_files(out_dir: str) -> int:
  return sum(1 for name in os.listdir(out_dir) if name.startswith("batch_") and name.endswith(".npz"))


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--spec", required=True, help="JSON file written by _ProducerNpzHostLoader.")
  args = parser.parse_args(argv)
  with open(args.spec, encoding="utf-8") as f:
    spec = json.load(f)
  out_dir = spec["out_dir"]
  ahead = max(1, int(spec["ahead"]))
  os.makedirs(out_dir, exist_ok=True)
  kwargs = dict(spec["live_kwargs"])
  row_slice = kwargs.pop("row_slice", None)
  kwargs["row_slice"] = slice(int(row_slice[0]), int(row_slice[1])) if row_slice else None
  kwargs["dtype_mm"] = jnp.dtype(kwargs["dtype_mm"])
  ranks = [int(r) for r in spec["ranks"]]
  _log(f"ranks {ranks} of {kwargs['shard_count']}, {kwargs['num_workers']} process workers a rank, out_dir {out_dir}, "
       f"{ahead} batches ahead")

  # The training process that started us. When it dies we are reparented (to PID 1 or a subreaper); compare
  # against the PID we started under rather than against 1: in the training container the launcher execs
  # python, so the training process itself is PID 1 and "ppid == 1" was always true (2026-11-21).
  parent_pid = os.getppid()
  loaders, firsts = [], []
  t_start = time.time()
  # The workers are forked from this process. Freeze the heap first so the children's garbage collector does not
  # touch the inherited objects' headers, which would make every shared page private in every child (CPython
  # copy-on-write; 64 children of a parent holding four ranks' state reached 370 GB on one host, 2026-11-21).
  gc.collect()
  gc.freeze()
  for rank in ranks:
    t0 = time.time()
    loader = mm._LiveMixtureHostLoader(shard_index=rank, **kwargs)  # pylint: disable=protected-access
    firsts.append(next(loader))
    loaders.append(loader)
    _log(f"rank {rank}: loader built and first batch ready in {time.time() - t0:.1f} s")
  _log(f"all ranks ready in {time.time() - t_start:.1f} s")

  pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(loaders))
  step = int(spec.get("start_step", 0))
  batch = _concat(firsts)
  produced = 0
  t_window = time.time()
  while True:
    while _pending_files(out_dir) >= ahead:
      if os.getppid() != parent_pid:
        _log(f"parent process {parent_pid} gone; exiting")
        return
      time.sleep(0.05)
    tmp = os.path.join(out_dir, f".{BATCH_FILE.format(step=step)}.tmp")
    final = os.path.join(out_dir, BATCH_FILE.format(step=step))
    mm.save_maxtext_batch_npz(tmp, batch)
    os.replace(tmp, final)
    step += 1
    produced += 1
    if produced % 20 == 0:
      dt = time.time() - t_window
      _log(f"wrote batch {step - 1}; last 20 batches {dt:.1f} s ({dt / 20:.2f} s a batch, wall, includes waits)")
      t_window = time.time()
    batch = _concat([f.result() for f in [pool.submit(next, loader) for loader in loaders]])


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
