"""Big-slice KDA check on 16 CPU devices: OLMoE3KimiDeltaAttention gives the same output and gradients under
olmo35-ep-as-cp (fsdp 2 x expert 8, r2's rule) and olmo35-cp-ep (context 2 x expert 8, value dim on context), with the
same params and a 2 x 2048 batch holding packed documents and padding. (CPU XLA has no ragged-all-to-all, so the MoE
path is checked on TPU.)  XLA_FLAGS=--xla_force_host_platform_device_count=16 python cpu_kda_eq.py"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from absl import flags
from flax import nnx
from flax.linen import partitioning as nn_partitioning

sys.path.insert(0, "/opt/scratch/mt-big/src")
from maxtext.configs import pyconfig  # noqa: E402
from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention  # noqa: E402
from maxtext.utils import maxtext_utils  # noqa: E402
from maxtext.utils.globals import MAXTEXT_PKG_DIR  # noqa: E402

T, B = 2048, 2


def cfg_for(rule, ici):
  return pyconfig.initialize(["", os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), "model_name=olmo35-tiny",
      "override_model_config=true", "run_name=kda_eq", "enable_checkpointing=False", f"per_device_batch_size={B / 16}",
      f"max_target_length={T}", "dtype=float32", "weight_dtype=float32", "skip_jax_distributed_system=True",
      "use_tokamax_kda=false", "kda_chunked_impl=subblock", "kda_matmul_precision=highest", "base_emb_dim=256",
      f"custom_mesh_and_rule={rule}", "context_parallel_load_balance=false", "allow_split_physical_axes=true", *ici])


def run(cfg, state=None):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  rng = np.random.default_rng(0)
  x = jnp.asarray(rng.standard_normal((B, T, cfg.emb_dim)).astype(np.float32))
  seg = np.ones((B, T), np.int32); seg[0, 1200:] = 2; seg[1, T - 300:] = 0
  seg = jnp.asarray(seg)
  with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
    layer = OLMoE3KimiDeltaAttention(cfg, mesh, None, rngs=nnx.Rngs(params=0))
    graphdef, params = nnx.split(layer)
    if state is not None:
      params = jax.tree.map(jnp.asarray, state)
    # Inputs sharded as the decoder hands them over: batch on fsdp, sequence on the length axes.
    xs = jax.device_put(x, NamedSharding(mesh, nn_partitioning.logical_to_mesh_axes(
        ("activation_batch", "activation_length", "activation_embed"), cfg.logical_axis_rules)))

    def f(p, x_):
      out = nnx.merge(graphdef, p)(x_, seg)
      return (out ** 2 * (seg != 0)[..., None]).sum(), out
    (val, out), g = jax.jit(jax.value_and_grad(f, has_aux=True))(params, xs)
    hlo = jax.jit(jax.value_and_grad(f, has_aux=True)).lower(params, xs).compile().as_text()
    import re as _re
    shapes = sorted(set(_re.findall(r"f32\[(\d+,\d+,\d+,64,\d+)\]", hlo)))[:12]; ops = {k: hlo.count(k + "(") for k in ("all-to-all", "all-gather", "all-reduce", "collective-permute", "reduce-scatter")}
    print(cfg.custom_mesh_and_rule, "per-device [B,H,N,C,*] last dims:", shapes, ops, flush=True)
    gx = jax.jit(jax.grad(lambda x_: f(params, x_)[0]))(xs)
  return float(val), np.asarray(out), jax.tree.map(np.asarray, g), np.asarray(gx), jax.tree.map(np.asarray, params)


if __name__ == "__main__":
  flags.FLAGS(sys.argv[:1])
  assert jax.device_count() == 16, jax.device_count()
  va, oa, ga, gxa, p = run(cfg_for("olmo35-ep-as-cp", ["ici_fsdp_parallelism=2", "ici_expert_parallelism=8"]))
  vb, ob, gb, gxb, _ = run(cfg_for("olmo35-cp-ep", ["ici_fsdp_parallelism=1", "ici_context_parallelism=2", "ici_expert_parallelism=8"]), p)
  rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
  grel = [rel(a, b) for a, b in zip(jax.tree.leaves(ga), jax.tree.leaves(gb))]
  print(f"out rel {rel(oa, ob):.3e}  dx rel {rel(gxa, gxb):.3e}  dparam rel max {max(grel):.3e} over {len(grel)} leaves  val {va:.6e} {vb:.6e}")
  print("KDA-EQ", "PASS" if max(rel(oa, ob), rel(gxa, gxb), max(grel)) < 1e-5 else "FAIL")
