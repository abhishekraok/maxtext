"""Fast data-cursor sidecars for MaxText SFT saves (H269; owner's approval, #1437 comment 7109262245).

Writer. When Tunix initiates the save of update S, the save hook gives it a checkpoint identity, "<nonce>:<S>", in the
checkpoint's own custom metadata (METADATA_KEY). The nonce is drawn once per attempt and is the same on every host.
Each host then writes one object, <state_dir>/step{S:07d}/host{p:04d}.json, holding the cursor and a sha256 of it. The
cursor records the step, the batches pulled (S x gradient accumulation), the host, the raw rows consumed, the identity
and the host's input shard (index, count). It is written only when eligible(), which refuses:
- any batch past MARGIN batches before this host's last real batch. The ends come from the stream's CPU replay
  receipt (RECEIPT.json `hosts[].last_step`), read from MAXTEXT_FAST_CURSOR_RECEIPT and accepted only when its sha256
  equals MAXTEXT_FAST_CURSOR_RECEIPT_SHA256, which the launcher pins (pin_from_env). The sidecar records that receipt
  sha and the loader config (rows per batch, gradient accumulation; shard index and count), and a restore refuses
  unless both equal the run's own pin and config. A run with no valid pin publishes nothing, and restores nothing. Near the end the packer's final flush
  can advance the cursor and still be wrong (several packed rows per batch). With a loading worker, the source's own
  end-of-stream flag lives in the worker's copy and never reaches the parent, so no end-of-stream signal is
  trusted: everything near or past the pinned end falls back.
- the padding tail;
- a batch that consumed no new raw row;
- a source that says it is exhausted (when the parent can see it).

Reader, in two stages, each falling back to the registered uniform skip with its reason logged on every host:
- candidate() runs before the loader is built. It takes the newest committed save only, never an older one. It
  checks the object's sha and fields, and that batches == step x gradient accumulation. Any I/O or parse error is a
  fallback.
- verify_restored() runs after Tunix restores. Tunix must have restored that step, at that batch count, with that
  identity in its custom metadata, and this host's shard must be the recorded one. Otherwise the loader is rebuilt
  without the skip.

Replay-derived cursors (H267 Amendment 6). A save written by a run without this patch has no sidecar and no identity
in its metadata, so the reader above can never use it. For such a save a cursor can be computed offline from the
stream's pinned replay (derive_cursor.py: the host's cumulative documents consumed at step S). It is a second,
explicit kind: its identity is "replay:<S>", and it carries one more field, the digest of the replay pieces it was
computed from. It is accepted only when all of these hold; otherwise the host falls back as above:
- the launcher pins this host's cursor sha256 (MAXTEXT_FAST_CURSOR_REPLAY_SHA256, a JSON object of process index to
  sha256), and the object read has that sha. The pin is the sidecar's own "sha256" field: the hash of the cursor as
  canonical JSON (its fields, keys sorted), which the reader recomputes from the fields it uses. It is not a hash of
  the file's bytes, so formatting outside the cursor is not covered and does not matter;
- its receipt sha and replay-pieces digest equal the pinned receipt's own, for this host's input shard;
- after the restore, the save's metadata has NO fast-cursor identity key, whatever its value, null included (a save
  a patched run wrote uses its own sidecar, never a replay-derived one), and the restored step, batch count and
  shard equal the cursor's.
The identity check for saves written by a patched run is unchanged.

Installed into MaxText as src/maxtext/input_pipeline/fast_cursor.py by experiments/fast_cursor/fast_cursor.patch;
tests/test_fast_cursor.py pins that copy to this file.
"""
import hashlib
import json

METADATA_KEY = "fast_cursor_id"
MARGIN = 4  # batches: covers the packer's final flush, which holds at most one batch of rows; the minimum allowed
FIELDS = ("step", "batches", "host", "raw_rows_consumed", "identity", "shard_index", "shard_count", "receipt_sha256",
          "rows_per_batch", "grad_accum")
REPLAY_FIELDS = FIELDS + ("replay_pieces_sha256",)  # a replay-derived cursor's fields
REPLAY_PREFIX = "replay:"


def is_replay(cursor):
  ident = cursor.get("identity") if isinstance(cursor, dict) else None
  return isinstance(ident, str) and ident.startswith(REPLAY_PREFIX)


def _fields(cursor):
  return REPLAY_FIELDS if is_replay(cursor) else FIELDS


def replay_pieces_digest(pieces):
  """sha256 over a receipt's list of one host's replay pieces (file name and sha256 of each, in order)."""
  body = json.dumps([[p["file"], p["sha256"]] for p in pieces])
  return hashlib.sha256(body.encode()).hexdigest()


def _int(v):
  return isinstance(v, int) and not isinstance(v, bool)


def identity(nonce, step):
  return f"{int(nonce)}:{int(step)}"


def pin_from_env(env, read_bytes, log=print):
  """{"receipt_sha256", "ends": {data host: last real batch}, "margin"} from the pinned receipt, or None (no
  publishing, no fast restore). Never raises."""
  try:
    path, want = env.get("MAXTEXT_FAST_CURSOR_RECEIPT", ""), env.get("MAXTEXT_FAST_CURSOR_RECEIPT_SHA256", "").strip()
    if not path or len(want) != 64:
      log("FAST-CURSOR: no pinned receipt (MAXTEXT_FAST_CURSOR_RECEIPT and _SHA256)")
      return None
    data = read_bytes(path)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
      log(f"FAST-CURSOR: receipt {path} sha256 {got} != pinned {want}")
      return None
    receipt = json.loads(data)
    ends = {h["host"]: h["last_step"] for h in receipt["hosts"]}
    margin_text = env.get("MAXTEXT_FAST_CURSOR_MARGIN", "") or str(MARGIN)
    margin = int(margin_text) if margin_text.strip().isdigit() else None
    if receipt.get("ok") is not True or not ends or not all(_int(k) and k >= 0 and _int(v) and v > 0
                                                           for k, v in ends.items()):
      log(f"FAST-CURSOR: receipt {path} is not ok or has malformed host ends")
      return None
    if margin is None or margin < MARGIN:
      log(f"FAST-CURSOR: margin {margin_text!r} is not an integer >= {MARGIN}")
      return None
    # Replay-derived cursors: the receipt's replay pieces per host, and the sidecar shas the launcher pins.
    replay_pieces = {h["host"]: replay_pieces_digest(h["replay_pieces"]) for h in receipt["hosts"]
                     if h.get("replay_pieces")}
    replay_pins = {}
    text = env.get("MAXTEXT_FAST_CURSOR_REPLAY_SHA256", "").strip()
    if text:
      try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and all(isinstance(v, str) and len(v) == 64 for v in parsed.values()):
          replay_pins = {int(k): v for k, v in parsed.items()}
        else:
          log("FAST-CURSOR: MAXTEXT_FAST_CURSOR_REPLAY_SHA256 is not an object of process index to sha256; ignored")
      except ValueError as e:
        log(f"FAST-CURSOR: MAXTEXT_FAST_CURSOR_REPLAY_SHA256 unreadable ({e!r}); ignored")
    return {"receipt_sha256": got, "ends": ends, "margin": margin, "replay_pieces": replay_pieces,
            "replay_pins": replay_pins}
  except Exception as e:  # pylint: disable=broad-except
    log(f"FAST-CURSOR: no usable pinned receipt ({e!r})")
    return None


def loader_config(global_batch, process_count, grad_accum):
  """`global_batch` is the batch the loader LOADS per update (config.global_batch_size_to_load), not the batch trained
  on: each process loads global_batch // process_count rows. With per_device_batch_size < 1 the two differ (H267:
  512 loaded and 16 trained on 128 processes, so 4 rows per process; 16 // 128 would be 0)."""
  return {"rows_per_batch": int(global_batch) // int(process_count), "grad_accum": max(int(grad_accum or 1), 1)}


def eligible(raw_at, batches, padding, source_exhausted, host_end, margin=MARGIN):
  """None if the cursor after `batches` pulled batches can be resumed from, else the reason it cannot.
  `host_end` is this data host's pinned last real batch (None: not pinned); `source_exhausted` the source's own flag
  where the parent can see it (True, False, or None)."""
  if not (_int(host_end) and host_end > 0):
    return "this host's last real batch is not pinned"
  if not (_int(margin) and margin >= MARGIN):
    return f"margin {margin!r} is not an integer >= {MARGIN}"
  if batches > host_end - margin:
    return f"batch {batches} is within {margin} of this host's last real batch {host_end}"
  if padding:
    return "the host is in its padding tail"
  if source_exhausted:
    return "the raw source is exhausted"
  if batches not in raw_at or batches - 1 not in raw_at:
    return "no recorded position for this batch and the one before it"
  if raw_at[batches] <= raw_at[batches - 1]:
    return "the batch consumed no new raw row (final packer flush)"
  return None


def _path(state_dir, step, host):
  return state_dir / f"step{int(step):07d}" / f"host{int(host):04d}.json"


def encode(cursor):
  body = json.dumps({k: cursor[k] for k in _fields(cursor)}, sort_keys=True)
  return json.dumps({"cursor": json.loads(body), "sha256": hashlib.sha256(body.encode()).hexdigest()},
                    sort_keys=True).encode()


def write(state_dir, cursor):
  """Publishes the cursor as one object (written to a temporary name, then renamed). Returns its sha."""
  path = _path(state_dir, cursor["step"], cursor["host"])
  path.parent.mkdir(parents=True, exist_ok=True)
  data = encode(cursor)
  tmp = path.parent / (path.name + ".tmp")
  tmp.write_bytes(data)
  try:
    tmp.replace(path)  # overwrites a cursor left by an earlier attempt at this step
  except (AttributeError, NotImplementedError):
    if path.exists():
      path.unlink()  # a reader in between finds no cursor and falls back
    tmp.rename(path)
  return json.loads(data)["sha256"]


def _check(obj, step, host, grad_accum, pin, config):
  c, sha = obj["cursor"], obj["sha256"]
  fields = _fields(c)
  if set(c) != set(fields):
    return "sha or fields do not match"
  body = json.dumps({k: c[k] for k in fields}, sort_keys=True)
  if hashlib.sha256(body.encode()).hexdigest() != sha:
    return "sha or fields do not match"
  texts = ("identity", "receipt_sha256", "replay_pieces_sha256")
  if not all(_int(c[k]) and c[k] >= 0 for k in fields if k not in texts) or not all(
      isinstance(c[k], str) for k in fields if k in texts):
    return "malformed field"
  if c["step"] != step or c["host"] != host:
    return f"names step {c['step']} host {c['host']}"
  if c["batches"] != step * grad_accum:
    return f"batches {c['batches']} != step {step} x gradient accumulation {grad_accum}"
  if c["shard_count"] < 1 or c["shard_index"] >= c["shard_count"]:
    return "malformed shard"
  if c["receipt_sha256"] != pin["receipt_sha256"]:
    return f"was written under receipt {c['receipt_sha256'][:12]}, this run pins {pin['receipt_sha256'][:12]}"
  if (c["rows_per_batch"], c["grad_accum"]) != (config["rows_per_batch"], config["grad_accum"]):
    return f"loader config {c['rows_per_batch']} rows x GA {c['grad_accum']} != this run's {config}"
  if is_replay(c):
    if c["identity"] != f"{REPLAY_PREFIX}{step}":
      return f"replay-derived cursor names {c['identity']!r}, not {REPLAY_PREFIX}{step}"
    if pin.get("replay_pins", {}).get(host) != sha:
      return "replay-derived cursor is not the one the launcher pins for this host"
    if c["replay_pieces_sha256"] != pin.get("replay_pieces", {}).get(c["shard_index"]):
      return "replay-derived cursor was not computed from the pinned receipt's replay pieces for this shard"
  return None


def candidate(checkpoint_dir, state_dir, host, grad_accum, pin, config, log=print):
  """The newest committed save's verified cursor (a dict), or None: the uniform-skip fallback. `pin` and `config` are
  this run's (pin_from_env, loader_config); without a pin there is no fast restore."""
  if pin is None:
    log(f"FAST-CURSOR host {host}: fallback: this run pins no receipt")
    return None
  try:
    steps = sorted((int(p.name) for p in checkpoint_dir.iterdir() if p.name.isdigit()), reverse=True)
    step = next((s for s in steps if (checkpoint_dir / str(s) / "commit_success.txt").exists()), None)
    if step is None:
      log(f"FAST-CURSOR host {host}: fallback: no committed save")
      return None
    obj = json.loads(_path(state_dir, step, host).read_bytes())  # read once; this object is the one returned
    why = _check(obj, step, host, grad_accum, pin, config)
  except Exception as e:  # pylint: disable=broad-except
    log(f"FAST-CURSOR host {host}: fallback: no readable sidecar ({e!r})")
    return None
  if why:
    log(f"FAST-CURSOR host {host}: fallback: step {step} sidecar {why}")
    return None
  c = obj["cursor"]
  log(f"FAST-CURSOR host {host}: candidate step {step}, batches {c['batches']}, raw rows {c['raw_rows_consumed']}, "
      f"identity {c['identity']}")
  return c


def verify_restored(cursor, restored_step, restored_batches, restored_metadata, shard_index, shard_count):
  """None if Tunix restored exactly the save the cursor belongs to, on this host's recorded shard; else the reason."""
  got = (restored_metadata or {}).get(METADATA_KEY)
  if is_replay(cursor) and METADATA_KEY in (restored_metadata or {}):
    return (f"the save carries fast-cursor identity {got!r}: a replay-derived cursor is only for a save written "
            "without one")
  if restored_step != cursor["step"]:
    return f"Tunix restored step {restored_step}, the cursor is for {cursor['step']}"
  if restored_batches != cursor["batches"]:
    return f"Tunix is at batch {restored_batches}, the cursor at {cursor['batches']}"
  if not is_replay(cursor) and got != cursor["identity"]:
    return f"checkpoint identity {got!r} != the cursor's {cursor['identity']!r} (a sidecar from another attempt)"
  if (shard_index, shard_count) != (cursor["shard_index"], cursor["shard_count"]):
    return f"input shard {shard_index}/{shard_count} != recorded {cursor['shard_index']}/{cursor['shard_count']}"
  return None
