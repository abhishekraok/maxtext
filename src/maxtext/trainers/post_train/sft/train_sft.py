#  Copyright 2023-2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
SFT training script that calls a trainer in Tunix to run SFT on a MaxText model
using `HuggingFaceH4/ultrachat_200k` dataset. The configurations for the dataset
are defined inside `src/maxtext/configs/post_train/sft.yml`.

Example command:
Training & Evaluation:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml \
    run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL_NAME?} load_parameters_path=${CHECKPOINT_PATH?} \
    hf_access_token=${HF_ACCESS_TOKEN?} tokenizer_path=${TOKENIZER_PATH?} \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane

Training:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml \
    run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL_NAME?} load_parameters_path=${CHECKPOINT_PATH?} \
    hf_access_token=${HF_ACCESS_TOKEN?} tokenizer_path=${TOKENIZER_PATH?} \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=-1 steps=10 profiler=xplane
"""

from typing import Any, Sequence

from absl import app
import os
import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import optax
import pathwaysutils

from flax import nnx
from flax.nnx import tracers
from flax.core.spmd import logical_axis_rules

from orbax import checkpoint as ocp

from tunix.sft import diffusion as tunix_diffusion_sft
from tunix.sft import checkpoint_manager as checkpoint_manager_lib
from tunix.sft import metrics_logger, peft_trainer, profiler

from maxtext.optimizers import optimizers
from maxtext.configs import pyconfig
from maxtext.integration.tunix import diffusion_sft as maxtext_diffusion_sft
from maxtext.trainers.pre_train.train import loss_fn, required_rbf_log_lines
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.trainers.post_train.sft import hooks
from maxtext.utils import lora_utils
from maxtext.utils import max_utils
from maxtext.utils import max_logging
# Placeholder: internal
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


def _host_scalar(x):
  """A replicated scalar (jax.Array or Python number) as a float, read from this host's first shard."""
  return float(x.addressable_data(0)) if isinstance(x, jax.Array) else float(x)


def _host_value(x):
  """A replicated array (jax.Array or numpy) as numpy, read from this host's first shard."""
  return np.asarray(x.addressable_data(0)) if isinstance(x, jax.Array) else np.asarray(x)


def _log_hbm(where):
  """Logging only: HBM in use, peak and limit on this host's first device (nothing on backends without stats)."""
  stats = jax.local_devices()[0].memory_stats()
  if not stats:
    return
  gib = lambda k: stats.get(k, 0) / 2**30
  max_logging.log(
      f"HBM {where}: in_use={gib('bytes_in_use'):.2f} GiB peak={gib('peak_bytes_in_use'):.2f} GiB "
      f"limit={gib('bytes_limit'):.2f} GiB host={jax.process_index()}"
  )


def _install_data_state_save_hook(mt_config):
  """Fast cursor: each save of update S carries the identity "<nonce>:<S>" in its custom metadata, and after Tunix
  initiates it each host writes its cursor for batch S x grad_accum with that identity (write_data_state). The nonce is
  drawn on process 0 and broadcast, so it is the same on every host and new on every attempt."""
  from jax.experimental import multihost_utils  # pylint: disable=import-outside-toplevel
  from maxtext.input_pipeline import fast_cursor, multihost_dataloading  # pylint: disable=import-outside-toplevel

  orig = checkpoint_manager_lib.CheckpointManager.save
  if getattr(orig, "_fast_cursor", False):
    return
  ga = max(int(mt_config.gradient_accumulation_steps or 1), 1)
  loader = fast_cursor.loader_config(mt_config.global_batch_size_to_load, jax.process_count(), ga)
  nonce = int(multihost_utils.broadcast_one_to_all(np.int32(int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF)))

  def save(self, step, *args, custom_metadata=None, **kwargs):
    ident = fast_cursor.identity(nonce, step)
    meta = dict(custom_metadata or {})
    meta[fast_cursor.METADATA_KEY] = ident
    ok = orig(self, step, *args, custom_metadata=meta, **kwargs)
    if ok:
      multihost_dataloading.write_data_state(int(step), int(step) * ga, ident, loader)
    return ok

  save._fast_cursor = True  # pylint: disable=protected-access
  checkpoint_manager_lib.CheckpointManager.save = save


def _restore_targets_from_live_shardings(state):
  """Restore targets for the optimizer state, taken from the live arrays' own shardings.

  Tunix's checkpoint_manager._fix_sharding rebuilds them from nnx partition specs, where a logical axis name with no
  rule (e.g. "mlp_moe") passes through unchanged; it then sees an axis the mesh lacks and replicates the whole leaf, so
  every expert tensor's Adam moments are restored as a full copy on each chip and the restore runs out of HBM. The live
  arrays already carry the sharding training uses, so restoring into it changes nothing but the load. Scalars with a
  single-device sharding (e.g. the step count) are replicated, as Tunix intends.
  """
  mesh = next(
      (x.sharding.mesh for x in jax.tree_util.tree_leaves(state) if isinstance(getattr(x, "sharding", None), NamedSharding)),
      None,
  )
  if mesh is None:
    return state

  def target(x):
    sharding = x.sharding if isinstance(getattr(x, "sharding", None), NamedSharding) else NamedSharding(mesh, PartitionSpec())
    return jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x), sharding=sharding)

  return jax.tree_util.tree_map(target, state)



_FORWARD_BF16_ALL = os.environ.get("MAXTEXT_FORWARD_BF16_ALL_PARAMS", "") == "1"
_FORWARD_BF16_CHECK = os.environ.get("MAXTEXT_FORWARD_BF16_CHECK", "") == "1"
_LOG_UPDATE_NORM = os.environ.get("MAXTEXT_LOG_UPDATE_NORM", "") == "1"
_TRACK_PARAMS = os.environ.get("MAXTEXT_TRACK_PARAMS", "") == "1"
_EXPERT_GRAD_SCALE = float(os.environ.get("MAXTEXT_EXPERT_GRAD_SCALE", "1") or 1)


def _round_bf16_bits(x):
  """Round-to-nearest-even of float32 values to the bfloat16 grid, kept in float32, via integer bit ops (XLA may drop an
  f32->bf16->f32 convert pair under xla_allow_excess_precision; it cannot drop these). Non-f32 inputs are returned as is."""
  if x.dtype != jnp.float32:
    return x
  u = jax.lax.bitcast_convert_type(x, jnp.uint32)
  lsb = (u >> 16) & jnp.uint32(1)
  u = (u + jnp.uint32(0x7FFF) + lsb) & jnp.uint32(0xFFFF0000)
  return jax.lax.bitcast_convert_type(u, jnp.float32)


def _param_group(path) -> str:
  """Coarse parameter group for diagnostics: routed experts, router, norms, or dense (everything else)."""
  k = jax.tree_util.keystr(path).lower()
  if ("moeblock" in k or "moe_block" in k) and any(w in k for w in ("['wi']", "['wi_0']", "['wi_1']", "['wo']")):
    return "expert"
  if "gate" in k or "router" in k:
    return "router"
  if "norm" in k:
    return "norm"
  return "dense"

class MaxTextPeftTrainer(peft_trainer.PeftTrainer):
  """MaxText-specific PeftTrainer that avoids nested NNX transformations.

  Tunix's default PeftTrainer._train_step creates nnx.value_and_grad inside
  nnx.jit. This nesting causes Flax NNX to assign conflicting outer_index
  values to graph nodes, resulting in:
    ValueError: The graph structure of a node added to cached_partial was
    mutated inside the transformation.

  This subclass overrides create_train_step_fn to use jax.value_and_grad
  with an explicit split/merge pattern (matching MaxText's pre-training NNX
  train_step), which avoids the nested NNX transformation issue entirely.
  """

  # The MaxText config; set by _create_trainer. self.config is Tunix's TrainingConfig, which has none of the MaxText
  # flags (reading log_moe_dropped_tokens from it is why the 51f8cc4b drop-count image never printed MOE-DROPPED).
  mt_config = None

  def _post_process_train_step(self, aux):
    """Logs the loss's parts (LOSS-PARTS), the MoE buffer probe (REQUIRED_RBF, MOE-DROPPED) and HBM, on host 0.

    Tunix calls this once per micro-batch, after buffering the step's metrics; the step number is the update step.
    """
    if not aux or jax.process_index() != 0:
      return
    step = self._train_steps + 1
    # The scalars are replicated, so this host's copy is the global value.
    xent_sum, weights, z, lb = (
        _host_scalar(aux.get(k, 0.0)) for k in ("xent_sum", "total_weights", "z_loss", "moe_lb_loss")
    )
    ce = xent_sum / (weights + 1e-8) - z  # xent_sum includes the z-loss term
    max_logging.log(f"LOSS-PARTS step={step} ce={ce:.6f} moe_lb={lb:.6f} z={z:.6f} tokens={weights:.0f}")
    if "doc_xent_sum" in aux:
      # MAXTEXT_LOG_DOC_CE: CE split into document-role segments (e.g. long-context replay) and chat segments.
      dsum, dw, csum_all = (_host_scalar(aux[k]) for k in ("doc_xent_sum", "doc_weights", "ce_only_sum"))
      cw = weights - dw
      max_logging.log(
          f"DOC-CE step={step} doc_ce={dsum / max(dw, 1e-8):.6f} doc_tokens={dw:.0f} "
          f"chat_ce={(csum_all - dsum) / max(cw, 1e-8):.6f} chat_tokens={cw:.0f} doc_share={dw / max(weights, 1e-8):.4f}"
      )
    if "sp_xent_sum" in aux:
      ssum, sw = _host_scalar(aux["sp_xent_sum"]), _host_scalar(aux["sp_weights"])
      max_logging.log(
          f"SPECIAL-CE step={step} sp_ce={ssum / max(sw, 1e-8):.6f} sp_tokens={sw:.0f} "
          f"rest_ce={(_host_scalar(aux['ce_only_sum']) - ssum) / max(weights - sw, 1e-8):.6f} sp_share={sw / max(weights, 1e-8):.5f}"
      )
    if "moe_required_rbf" in aux and aux["moe_required_rbf"] is not None:
      # log_required_ragged_buffer_factor: per MoE layer, the smallest ragged_buffer_factor that would not have
      # dropped a token this step; at or below the run's factor, the step dropped nothing. With
      # log_moe_dropped_tokens, also the dropped and routed (token, expert) rows.
      rbf = aux["moe_required_rbf"]
      rbf = rbf.addressable_data(0) if isinstance(rbf, jax.Array) else rbf
      for line in required_rbf_log_lines(self.mt_config, step, rbf):
        max_logging.log(line)
    if "moe_expert_counts" in aux and aux["moe_expert_counts"] is not None:
      # MAXTEXT_LOG_EXPERT_COUNTS: accumulate per-layer per-expert routed counts; write sum, sum of squares and the
      # per-step max-group load to MAXTEXT_EXPERT_COUNTS_OUT (npz) every 50 steps.
      c = np.asarray(_host_value(aux["moe_expert_counts"]), dtype=np.float64)
      acc = getattr(self, "_expert_counts_acc", None)
      if acc is None:
        acc = self._expert_counts_acc = {"sum": np.zeros_like(c), "sumsq": np.zeros_like(c), "n": 0, "per_step": []}
      acc["sum"] += c
      acc["sumsq"] += c * c
      acc["n"] += 1
      acc["per_step"].append(c.astype(np.float32))
      out = os.environ.get("MAXTEXT_EXPERT_COUNTS_OUT", "")
      if out and step % 50 == 0:
        import io  # pylint: disable=import-outside-toplevel
        from etils import epath  # pylint: disable=import-outside-toplevel
        buf = io.BytesIO()
        from maxtext.trainers.pre_train import train as _pt  # pylint: disable=import-outside-toplevel
        np.savez_compressed(buf, sum=acc["sum"], sumsq=acc["sumsq"], n=acc["n"], per_step=np.stack(acc["per_step"]),
                            names=np.asarray(_pt.EXPERT_COUNT_NAMES))
        epath.Path(out).write_bytes(buf.getvalue())
        max_logging.log(f"EXPERT-COUNTS step={step} wrote {out} ({acc['n']} steps, shape {c.shape})")
    if self._train_steps < 20 or step % 500 == 0:
      _log_hbm(f"step {step}")

  def create_train_step_fn(self):
    """Creates a train step using jax.value_and_grad with explicit NNX split/merge."""
    loss_fn_ref = self.loss_fn
    has_aux = self._has_aux
    gen_fn = self.gen_model_input_fn
    is_lora_enabled = self._lora_enabled
    wrt = nnx.LoRAParam if is_lora_enabled else nnx.Param

    # Capture the graphdef once outside of JIT so that split/merge inside
    # jax.value_and_grad can use a stable (non-traced) structural descriptor.
    nnx.pop(self.model, nnx.Intermediate)
    graphdef, _, _ = nnx.split(self.model, wrt, ...)
    _uses_gradient_accumulation = not (
        self.config.get_with_default("gradient_accumulation_steps", 1) == 1 and self.config.max_seq_token_per_tpu is None
    )

    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        grad_accumulator: Any = None,
        inputs: Any = None,
        is_update_step: Any = True,
    ):
      if inputs is None and grad_accumulator is not None:
        # In Tunix versions where train_step only receives (model, optimizer, inputs)
        inputs = grad_accumulator
        grad_accumulator = getattr(self, "grad_accumulator", None)

      inputs = gen_fn(inputs)

      # Split model into differentiable params and non-differentiable rest.
      # Using jax.value_and_grad (not nnx.value_and_grad) avoids nesting NNX
      # transforms inside nnx.jit, which would corrupt outer_index tracking.
      nnx.pop(model, nnx.Intermediate)
      _, diff_params, rest = nnx.split(model, wrt, ...)

      def loss_wrapper(diff_params, rest, **inputs_kw):
        if _TRACK_PARAMS:
          # MAXTEXT_TRACK_PARAMS=1: print, every step, the dtype and the first 3 elements (full fp32 precision) of the
          # first param in each group (expert / router / norm / dense), as the step sees them (pre-update masters).
          seen = set()
          for path, p in jax.tree_util.tree_leaves_with_path(diff_params):
            g = _param_group(path)
            if g in seen or not hasattr(p, "dtype") or p.size < 3:
              continue
            seen.add(g)
            jax.debug.print(
                "TRACK-PARAM " + g + " " + jax.tree_util.keystr(path) + " dtype=" + str(p.dtype) + " v={v}",
                v=jnp.ravel(p)[:3].astype(jnp.float32),
            )
        if _FORWARD_BF16_CHECK:
          # MAXTEXT_FORWARD_BF16_CHECK=1: print, every step, the max |p - bf16(p)| (master off the bf16 grid) over params
          # whose path names a norm and over router/gate params, and the same after the forward rounding (should be 0).
          def _offgrid(tree, pred):
            vals = [
                jnp.max(jnp.abs(p.astype(jnp.float32) - _round_bf16_bits(p.astype(jnp.float32))))
                for path, p in jax.tree_util.tree_leaves_with_path(tree)
                if hasattr(p, "dtype") and jnp.issubdtype(p.dtype, jnp.floating) and pred(jax.tree_util.keystr(path).lower())
            ]
            return jnp.max(jnp.stack(vals)) if vals else jnp.float32(-1)

          is_norm = lambda k: "norm" in k
          is_router = lambda k: "router" in k or "gate" in k
          rounded = jax.tree_util.tree_map(
              lambda p: _round_bf16_bits(p) if hasattr(p, "dtype") and p.dtype == jnp.float32 else p,
              diff_params,
          )
          jax.debug.print(
              "BF16-CHECK master_offgrid norm={a} router={b} | forward_offgrid(flag {f}) norm={c} router={d}",
              a=_offgrid(diff_params, is_norm), b=_offgrid(diff_params, is_router),
              c=_offgrid(rounded if _FORWARD_BF16_ALL else diff_params, is_norm),
              d=_offgrid(rounded if _FORWARD_BF16_ALL else diff_params, is_router), f=int(_FORWARD_BF16_ALL),
          )
        if _FORWARD_BF16_ALL:
          # MAXTEXT_FORWARD_BF16_ALL_PARAMS=1: the forward/backward sees every floating param rounded to bf16
          # (straight-through: the gradient goes to the fp32 master unchanged), as OLMo-core does with bf16 model
          # params + fp32 optimizer masters. MaxText otherwise keeps routers, norms and other small tensors in fp32.
          diff_params = jax.tree_util.tree_map(
              lambda p: p + jax.lax.stop_gradient(_round_bf16_bits(p) - p)
              if hasattr(p, "dtype") and p.dtype == jnp.float32
              else p,
              diff_params,
          )
        local_model = nnx.merge(graphdef, diff_params, rest)

        # JAX AD tracing ignores non-differentiable variables (like RngCount ints).
        # They get instantiated with the base/JIT trace instead of the active
        # LinearizeTrace, causing flax.errors.TraceContextError when mutated later.
        # We manually update their trace_state to the current AD trace context.

        for _, v in nnx.iter_graph(local_model):
          if isinstance(v, nnx.Variable) and hasattr(v, "_trace_state"):
            if not v._trace_state.is_valid():  # pylint: disable=protected-access
              object.__setattr__(v, "_trace_state", tracers.TraceState())  # pylint: disable=protected-access

        out = loss_fn_ref(local_model, **inputs_kw)
        # Capture updated non-param state (e.g. RNG counters) from local_model.
        _, _, new_rest = nnx.split(local_model, wrt, ...)
        if has_aux:
          loss, aux = out
          return loss, (aux, new_rest)
        else:
          return out, (None, new_rest)

      grad_fn = jax.value_and_grad(loss_wrapper, argnums=0, has_aux=True)
      (out_val, (aux, new_rest)), grads = grad_fn(diff_params, rest, **inputs)

      # Propagate updated non-param state (RNG counters, etc.) back to model.
      # Fix flax.errors.TraceContextError when returning from jax.value_and_grad

      for _, v in nnx.iter_graph(model):
        if isinstance(v, nnx.Variable) and hasattr(v, "_trace_state"):
          if not v._trace_state.is_valid():  # pylint: disable=protected-access
            object.__setattr__(v, "_trace_state", tracers.TraceState())  # pylint: disable=protected-access

      nnx.update(model, new_rest)

      # Handle gradient accumulation and conditional/direct optimizer update
      if not _uses_gradient_accumulation:
        if isinstance(grads, dict) and not isinstance(grads, nnx.State):
          grads = nnx.State(grads)
        if _EXPERT_GRAD_SCALE != 1.0:
          # MAXTEXT_EXPERT_GRAD_SCALE=s (diagnostic): multiply routed-expert gradients by s before the optimizer (so before
          # its global-norm clip), as OLMo-core's EP path divides expert grads by the EP size (H203: 8).
          grads = jax.tree_util.tree_map_with_path(
              lambda path, g: g * jnp.asarray(_EXPERT_GRAD_SCALE, g.dtype) if _param_group(path) == "expert" else g, grads
          )
        if _LOG_UPDATE_NORM:
          # Snapshot the masters as fresh arrays: nnx State leaves may be the module's own Variables, updated in place.
          pre_update = jax.tree_util.tree_map(lambda x: x + jnp.zeros((), x.dtype) if hasattr(x, "dtype") else x,
                                              nnx.split(model, wrt, ...)[1])
        optimizer.update(model, grads)
        grad_norm = optax.global_norm(jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), grads))
        if _LOG_UPDATE_NORM:
          # MAXTEXT_LOG_UPDATE_NORM=1: per step, the norm of the actual parameter change (fp32 masters) overall and per
          # group (expert / router / norm / dense), the max |delta|, and the fraction of elements that moved; plus
          # the fraction whose bf16 value (what the forward sees) changed.
          _, new_params, _ = nnx.split(model, wrt, ...)
          old_l = jax.tree_util.tree_leaves_with_path(pre_update)
          new_l = jax.tree_util.tree_leaves(new_params)
          sums = {"all": [], "expert": [], "router": [], "norm": [], "dense": []}
          moved, moved16, total, mx = [], [], [], []
          for (path, a), b in zip(old_l, new_l):
            if not (hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating)):
              continue
            d = b.astype(jnp.float32) - a.astype(jnp.float32)
            sq = jnp.sum(d * d)
            sums["all"].append(sq)
            sums[_param_group(path)].append(sq)
            moved.append(jnp.sum((d != 0).astype(jnp.float32)))
            moved16.append(jnp.sum((_round_bf16_bits(b.astype(jnp.float32)) != _round_bf16_bits(a.astype(jnp.float32))).astype(jnp.float32)))
            total.append(jnp.float32(a.size))
            mx.append(jnp.max(jnp.abs(d)))
          f = lambda xs: jnp.sqrt(sum(xs)) if xs else jnp.float32(0)
          jax.debug.print(
              "UPDATE-NORM all={a} expert={e} router={r} norm={n} dense={d} max_abs={m} moved_frac={mf} bf16_moved_frac={mb}",
              a=f(sums["all"]), e=f(sums["expert"]), r=f(sums["router"]), n=f(sums["norm"]), d=f(sums["dense"]),
              m=jnp.max(jnp.stack(mx)),
              mf=sum(moved) / sum(total), mb=sum(moved16) / sum(total),
          )
      else:
        grad_accumulator.add(grads)

        def apply_updates(model, optimizer, grad_accumulator):
          acc_grads = grad_accumulator.get()
          norm = optax.global_norm(jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), acc_grads))
          if isinstance(acc_grads, dict) and not isinstance(acc_grads, nnx.State):
            acc_grads = nnx.State(acc_grads)
          optimizer.update(model, acc_grads)
          grad_accumulator.reset()
          return norm

        def skip_updates(model, optimizer, grad_accumulator):
          return jnp.array(0.0, dtype=jnp.float32)

        grad_norm = nnx.cond(
            is_update_step,
            apply_updates,
            skip_updates,
            model,
            optimizer,
            grad_accumulator,
        )

      aux_out = aux if has_aux else None
      return out_val, aux_out, grad_norm

    return train_step


def get_tunix_config(mt_config):
  """Gets the Tunix training configurations from the MaxText config.

  Args:
    mt_config: MaxText config.

  Returns:
    A Tunix `TrainingConfig` object.
  """
  # Checkpointing configurations
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=mt_config.checkpoint_period,
      enable_async_checkpointing=mt_config.async_checkpointing,
      max_to_keep=mt_config.max_num_checkpoints_to_keep,
  )

  # Metrics configurations
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=mt_config.tensorboard_dir)

  # Profiler configurations
  profiler_options = None
  if mt_config.profiler:
    set_profile_options = True
    platform_version = jax.extend.backend.get_backend().platform_version.strip()
    if platform_version.startswith("Pathways"):
      max_logging.log("Pathways backend detected. Disabling setting profile options.")
      set_profile_options = False
    profiler_options = profiler.ProfilerOptions(
        log_dir=mt_config.tensorboard_dir,
        skip_first_n_steps=mt_config.skip_first_n_steps_for_profiler,
        profiler_steps=mt_config.profiler_steps,
        set_profile_options=set_profile_options,
        enable_continuous_profiling=mt_config.enable_continuous_profiling,
    )

  return peft_trainer.TrainingConfig(
      eval_every_n_steps=mt_config.eval_interval,
      max_steps=mt_config.steps,
      gradient_accumulation_steps=(
          mt_config.gradient_accumulation_steps if mt_config.gradient_accumulation_steps > 1 else None
      ),
      checkpoint_root_directory=mt_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
      data_sharding_axis=tuple(mt_config.data_sharding),
  )


def use_maxtext_loss_function(trainer, mt_config):
  """Configures the trainer to use MaxText's loss function.

  This function creates a wrapper around MaxText's `loss_fn` to make it
  compatible with the Tunix trainer's expected loss function signature.

  Args:
    trainer: The PeftTrainer instance.
    mt_config: MaxText config.

  Returns:
    The trainer configured with the MaxText loss function.
  """

  def make_loss_func(is_train):
    def loss_func(
        model,
        inputs,
        inputs_position,
        inputs_segmentation,
        targets,
        targets_position,
        targets_segmentation,
    ):
      data = {
          "inputs": inputs,
          "inputs_position": inputs_position,
          "inputs_segmentation": inputs_segmentation,
          "targets": targets,
          "targets_position": targets_position,
          "targets_segmentation": targets_segmentation,
      }
      return loss_fn(model, mt_config, data, dropout_rng=None, params=None, is_train=is_train)

    return loss_func

  # Tunix PeftTrainer.with_loss_fn() defaults `eval_loss_fn` to the same callable
  # as `loss_fn`. We override `eval_loss_fn` with `is_train=False` so that evaluation:
  #   - Disables dropout to ensure deterministic eval loss / perplexity.
  #   - Excludes auxiliary Multi-Token Prediction (MTP) loss from the reported eval loss.
  #   - Prevents updating running sparsity statistics (`batch_stats`) during eval.
  trainer = trainer.with_loss_fn(make_loss_func(is_train=True), has_aux=True)
  trainer.eval_loss_fn = make_loss_func(is_train=False)
  return trainer


def configure_training_objective(trainer, mt_config):
  """Configures causal or target-aligned diffusion SFT."""
  if getattr(mt_config, "training_objective", "causal_lm") != "block_diffusion":
    return use_maxtext_loss_function(trainer, mt_config)
  max_logging.log("Configuring Tunix target-aligned block-diffusion SFT adapter.")
  return tunix_diffusion_sft.configure_diffusion_sft(
      trainer,
      maxtext_diffusion_sft.create_batch_adapter(mt_config),
      maxtext_diffusion_sft.create_target_aligned_logits_fn(mt_config),
  )


def _create_trainer(model, optimizer, tunix_config, mt_config):
  """Creates the trainer implementation required by the selected objective.

  Causal SFT uses ``MaxTextPeftTrainer`` so its MaxText loss can run through
  the explicit NNX split/merge train step above. Block-diffusion SFT instead
  uses Tunix's base ``PeftTrainer`` because ``configure_diffusion_sft`` installs
  a Tunix ``LossOutput`` with an unreduced weighted loss and denominator-aware
  gradient accumulation, which the MaxText-specific train step does not handle.
  """
  if getattr(mt_config, "training_objective", "causal_lm") == "block_diffusion":
    return peft_trainer.PeftTrainer(model, optimizer, tunix_config)
  trainer = MaxTextPeftTrainer(model, optimizer, tunix_config)
  trainer.mt_config = mt_config
  return trainer


def validate_config(config):
  """Validates the configuration parameters for SFT training."""
  if config.optimizer_memory_host_offload:
    raise ValueError(
        "optimizer_memory_host_offload=True is not supported on the post-training "
        "SFT path because the underlying Tunix PeftTrainer does not support "
        "host offloading of the optimizer state."
    )
  if getattr(config, "training_objective", "causal_lm") == "block_diffusion":
    unsupported_auxiliary_losses = {
        name: value
        for name, value in (
            ("z_loss_multiplier", getattr(config, "z_loss_multiplier", 0.0)),
            ("load_balance_loss_weight", getattr(config, "load_balance_loss_weight", 0.0)),
            ("mtp_num_layers", getattr(config, "mtp_num_layers", 0)),
            ("indexer_loss_scaling_factor", getattr(config, "indexer_loss_scaling_factor", 0.0)),
        )
        if value != 0
    }
    if unsupported_auxiliary_losses:
      settings = ", ".join(f"{name}={value!r}" for name, value in unsupported_auxiliary_losses.items())
      raise ValueError(
          "Block-diffusion SFT currently supports only weighted cross-entropy; "
          f"set unsupported auxiliary-loss settings to zero: {settings}"
      )



def _apply_rbf_layer_overrides(model):
  """MAXTEXT_RBF_LAYER_OVERRIDES="<decoder layer number or 'first'>:<factor>,...": per-MoE-layer ragged_buffer_factor
  (RoutedMoE.ragged_buffer_factor_override, the hook the eval/first-phase programs use). Layers keep the global
  ragged_buffer_factor unless listed; 'first' is the lowest-numbered MoE layer. A buffer only changes which tokens a
  full buffer would drop, so with no drops the math is unchanged. Logs every MoE layer's path and factor."""
  spec = os.environ.get("MAXTEXT_RBF_LAYER_OVERRIDES", "").strip()
  if not spec:
    return
  import re  # pylint: disable=import-outside-toplevel
  from maxtext.layers import moe as moe_lib  # pylint: disable=import-outside-toplevel

  mods = []
  for path, module in nnx.iter_graph(model):
    if isinstance(module, moe_lib.RoutedMoE):
      keys = [str(k) for k in path]
      nums = [int(m.group(1)) for k in keys for m in [re.fullmatch(r"(?:\w*_)?layers_(\d+)", k)] if m]
      mods.append((nums[-1] if nums else None, "/".join(keys), module))
  if any(n is None for n, _, _ in mods):
    raise ValueError(f"MAXTEXT_RBF_LAYER_OVERRIDES: could not read a layer number from {[p for n, p, _ in mods if n is None]}")
  by_layer = {n: m for n, _, m in mods}
  first = min(by_layer)
  for item in spec.split(","):
    key, val = item.split(":")
    layer = first if key.strip() == "first" else int(key)
    if layer not in by_layer:
      raise ValueError(f"MAXTEXT_RBF_LAYER_OVERRIDES: no MoE layer {layer}; MoE layers: {sorted(by_layer)}")
    by_layer[layer].ragged_buffer_factor_override = float(val)
  for n, path, m in sorted(mods):
    max_logging.log(f"RBF-LAYER layer={n} path={path} factor={m.get_ragged_buffer_factor()}")

def setup_trainer_state(mt_config, goodput_recorder=None):
  """Set up prerequisites for training loop."""
  validate_config(mt_config)
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = model_creation_utils.from_pretrained(mt_config)

  with jax.set_mesh(mesh), logical_axis_rules(mt_config.logical_axis_rules):
    if mt_config.lora.enable_lora:
      model = lora_utils.apply_lora_to_model(model, mesh, mt_config)
    _apply_rbf_layer_overrides(model)

    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    # pass in model for muon
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule, model)

    if mt_config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(
          optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold),
          optimizer,
      )

    with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
      training_hooks = hooks.SFTTrainingHooks(mt_config, mesh, learning_rate_schedule, goodput_recorder)
      data_hooks = hooks.SFTDataHooks(mt_config, mesh, goodput_recorder)

      nnx.pop(model, nnx.Intermediate)
      if mt_config.lora.lora_restore_path:
        lora_utils.restore_lora_from_path(model, mt_config)
      # Restore the optimizer state into the live arrays' shardings (see _restore_targets_from_live_shardings).
      checkpoint_manager_lib._fix_sharding = _restore_targets_from_live_shardings  # pylint: disable=protected-access
      if os.environ.get("MAXTEXT_DATA_STATE_DIR"):
        _install_data_state_save_hook(mt_config)
      trainer = _create_trainer(model, optimizer, tunix_config, mt_config)
      _log_hbm("after trainer init (restore included)")
      trainer.with_training_hooks(training_hooks)
      trainer.with_data_hooks(data_hooks)
      trainer = configure_training_objective(trainer, mt_config)

  return trainer, mesh


def train_model(mt_config, trainer, mesh):
  """Runs the SFT training loop in Tunix."""
  with jax.set_mesh(mesh), logical_axis_rules(mt_config.logical_axis_rules):
    is_block_diffusion = getattr(mt_config, "training_objective", "causal_lm") == "block_diffusion"
    records_internal_metrics = bool(getattr(mt_config, "record_internal_nn_metrics", False))
    enable_nnx_cache = mt_config.num_experts <= 1 and not (is_block_diffusion and records_internal_metrics)

    trainer.train(
        trainer.data_hooks.train_data_iterator,
        trainer.data_hooks.eval_data_iterator,
        cache_nnx_graph=enable_nnx_cache,
    )
  return trainer


def train(mt_config, goodput_recorder=None):
  """Main method for SFT training.

  Args:
    mt_config: MaxText config.
    goodput_recorder: An optional GoodputRecorder to record performance metrics.
  """
  trainer, mesh = setup_trainer_state(mt_config, goodput_recorder)
  _job_completed_gracefully = False
  try:
    trainer = train_model(mt_config, trainer, mesh)
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
  return trainer, mesh


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  mt_config = pyconfig.initialize_pydantic(argv)
  validate_config(mt_config)
  max_utils.print_system_information()

  goodput_recorder = create_goodput_recorder(mt_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(mt_config):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
