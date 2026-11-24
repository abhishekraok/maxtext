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

"""OLMo 3.5 / Molmo2 vision tower, connector, splicing, and checkpoint conversion.

Implements the exact vision stack from ``allenai/OLMo-core`` (#916):
  1. ``OLMo35VisionEncoder``: ``SigLIP2-SO400M/14-378`` truncated to 25 blocks
     (no CLS token, no pre/post-LN, C-first ``14x14x3 = 588`` patch projection
     with bias, learned ``[729, 1152]`` positional embedding, ``LayerNorm(eps=1e-6)``
     with bias, 16x72 MHA with bias, 4304 ``gelu_pytorch_tanh`` MLP with bias),
     concatenating hidden states from blocks ``(24, 18)`` -> ``[B, N_crops * 729, 2304]``.
     Supports per-block remat, crop chunking, and automatic Context-Parallel
     crop sharding (each CP rank encodes ``N_crops // CP`` crops before AllGather).
  2. ``OLMo35VisionConnector``: gathers 4-patch groups via ``pooled_patches_idx``
     (``-1`` slots masked), pools via ``attention_meanq`` (masked mean query,
     FP32 ``_few_key_attention`` with finite mask fill), and projects via a
     bias-free SwiGLU MLP (``1152 -> 9728 -> emb_dim``).
  3. ``merge_olmo35_image_embeddings``: adds (``+=``) connector features at
     ``<im_patch>`` positions *after* ``sqrt(emb_dim)`` and ``embedding_norm``,
     supporting Context Parallelism via global prefix counts across ``context``.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import Config
from maxtext.layers import linears
from maxtext.utils import sharding

# Default Molmo2 / Dolma2 special token IDs when appended to dolma2-tokenizer (100,278 base -> 100,352 padded):
# 100278: <im_start>, 100279: <im_end>, 100280: <im_patch>, 100281: <im_col>, 100282: <low_res_im_start>, 100283: <|image|>
DEFAULT_OLMO_IMAGE_PATCH_TOKEN_ID = 100280


def _normal_kernel_init(stddev: float = 0.02):
  """MaxText ``DenseGeneral`` initializer accepting ``(key, shape, dtype, in_axis, out_axis)``."""
  base_init = jax.nn.initializers.normal(stddev=stddev)

  def _init(key, shape, dtype=jnp.float32, *_):
    return base_init(key, shape, dtype)

  return _init


def _vit_kernel_axes(in_f: int, out_f: int, config: Config, mesh: Mesh | None) -> tuple[str | None, str | None]:
  """Selects divisible logical sharding axes for ViT and Connector ``DenseGeneral`` kernels."""
  if mesh is None or bool(getattr(config, "olmo_vit_replicate_weights", False)):
    return (None, None)
  for axis_name in ("vit_embed_cp", "vit_embed"):
    spec = sharding.logical_to_mesh_axes((axis_name,), mesh, rules=config.logical_axis_rules)
    if spec is None or not spec.partitions or spec.partitions[0] is None:
      continue
    phys_axes = sharding.mesh_axes_for_dim(spec.partitions[0])
    prod = 1
    for axis in phys_axes:
      prod *= mesh.shape.get(axis, 1)
    if prod <= 1:
      continue
    if in_f % prod == 0:
      return (axis_name, None)
    if out_f % prod == 0:
      return (None, axis_name)
  return (None, None)


def _batch_mesh_spec_entry(config: Config, mesh: Mesh | None):
  """Returns active ``mesh`` axis (or tuple of axes) for ``activation_batch``, or ``None``."""
  if mesh is None:
    return None
  spec = sharding.logical_to_mesh_axes(("activation_batch",), mesh, rules=config.logical_axis_rules)
  if spec is None or not spec.partitions or not spec.partitions[0]:
    return None
  return spec.partitions[0]


# log_nonfinite_probe (trace-time state): while a ViT stage is traced inside ``_vit_layer_scope``, every weight is passed
# through ``_vit_weight_as_used`` (identity; a test hook) and, if probing, its non-finite count is appended to
# ``_WEIGHT_PROBE``. The counts are taken on the weights as consumed (after the explicit all-gather to ``P()``).
_WEIGHT_PROBE: list | None = None
_PROBE_LAYER: int = -1


def _vit_weight_as_used(w: jax.Array, layer: int) -> jax.Array:  # pylint: disable=unused-argument
  """Identity; unit tests patch it to corrupt the weights of one ViT stage as used (``layer``: 0 = patch/pos embed,
  ``i + 1`` = block ``i``; -1 outside the ViT stages, e.g. the connector)."""
  return w


def _use_weight(w: jax.Array) -> jax.Array:
  w = _vit_weight_as_used(w, _PROBE_LAYER)
  if _WEIGHT_PROBE is not None:
    _WEIGHT_PROBE.append(jnp.sum(jnp.logical_not(jnp.isfinite(w)), dtype=jnp.int32))
  return w


@contextlib.contextmanager
def _vit_layer_scope(layer: int, probe: bool):
  """Traces one ViT stage: sets the stage index for the weight hook and (``probe``) collects as-used weight counts."""
  global _WEIGHT_PROBE, _PROBE_LAYER  # pylint: disable=global-statement
  prev = (_WEIGHT_PROBE, _PROBE_LAYER)
  collected: list = []
  _WEIGHT_PROBE, _PROBE_LAYER = (collected if probe else None), layer
  try:
    yield collected
  finally:
    _WEIGHT_PROBE, _PROBE_LAYER = prev


class _ViTDense(linears.DenseGeneral):
  """``DenseGeneral`` that explicitly FSDP-gathers sharded ViT/Connector weights to ``P()`` before ``dot_general``.

  Prevents XLA GSPMD from propagating ``vit_embed`` / ``vit_embed_cp`` weight sharding onto activation
  feature dimensions (which would otherwise trigger involuntary full-tensor replication across ``context * expert``).
  """

  def __call__(self, inputs: jax.Array, *args, **kwargs) -> jax.Array:
    if self.mesh is None or self.quant is not None or args or kwargs:
      return super().__call__(inputs, *args, **kwargs)
    x = jnp.asarray(inputs, self.dtype)
    k = jnp.asarray(self.kernel[...], self.dtype)
    rep = NamedSharding(self.mesh, PartitionSpec())
    k = _use_weight(jax.lax.with_sharding_constraint(k, rep))
    prec = jax.lax.Precision(self.matmul_precision)
    y = jax.lax.dot_general(x, k, (((x.ndim - 1,), (0,)), ((), ())), precision=prec)
    if self.bias is not None:
      b = _use_weight(jax.lax.with_sharding_constraint(jnp.asarray(self.bias[...], self.dtype), rep))
      y = y + b
    return y


def patchify_hwc_to_olmo_cfirst(images_hwc: jax.Array, patch_size: int = 14) -> jax.Array:
  """Converts ``[B, N_crops, H, W, 3]`` images into OLMo-core's C-first patches ``[B, N_crops, (H/p)*(W/p), 3*p*p]``."""
  b, n_crops, h, w, c = images_hwc.shape
  p = patch_size
  if h % p != 0 or w % p != 0:
    raise ValueError(f"Image spatial dims ({h}, {w}) must be divisible by patch_size ({p}).")
  gh, gw = h // p, w // p
  # OLMo-core flattens each patch in PyTorch NCHW order: (3, p, p) -> 3 * p * p.
  patches = images_hwc.reshape(b, n_crops, gh, p, gw, p, c)
  patches = patches.transpose(0, 1, 2, 4, 6, 3, 5)
  return patches.reshape(b, n_crops, gh * gw, c * p * p)


def olmo35_crop_valid(pooled_patches_idx: jax.Array, n_crops: int, patches_per_crop: int) -> jax.Array:
  """``[B, N_crops]`` bool: crop ``c`` of row ``b`` is below ``n_used[b] = max(pooled_patches_idx[b]) // P + 1``.

  The connector reads only crops ``< n_used`` (every valid pooled index points into them); crops at or above it are
  padding (text rows: all of them).
  """
  idx = pooled_patches_idx.astype(jnp.int32)
  has_any = jnp.any(idx >= 0, axis=tuple(range(1, idx.ndim)))
  n_used = jnp.where(has_any, jnp.max(idx, axis=tuple(range(1, idx.ndim))) // int(patches_per_crop) + 1, 0)
  n_used = jnp.minimum(n_used, n_crops).astype(jnp.int32)
  return jnp.arange(n_crops, dtype=jnp.int32)[None, :] < n_used[:, None]


def _select_crops(x: jax.Array, crop_valid: jax.Array) -> jax.Array:
  """Zeros (by selection, so NaN / inf / garbage never survive) the crops of ``x`` ``[B, N_crops, ...]`` where
  ``crop_valid`` ``[B, N_crops]`` is False."""
  mask = crop_valid.reshape(crop_valid.shape + (1,) * (x.ndim - 2))
  return jnp.where(mask, x, jnp.zeros((), x.dtype))


def params_nonfinite_count(module: nnx.Module) -> jax.Array:
  """log_nonfinite_probe: the number of non-finite entries over all ``nnx.Param`` leaves of ``module`` (int32 scalar)."""
  total = jnp.zeros((), jnp.int32)
  for leaf in jax.tree.leaves(nnx.state(module, nnx.Param)):
    total = total + jnp.sum(jnp.logical_not(jnp.isfinite(leaf)), dtype=jnp.int32)
  return total


class _ViTLayerNorm(nnx.Module):
  """FP32 LayerNorm with weight and bias matching PyTorch ``nn.LayerNorm(eps=1e-6)``."""

  def __init__(self, dim: int, eps: float, dtype: Any, weight_dtype: Any, *, rngs: nnx.Rngs):
    self.dim = dim
    self.eps = eps
    self.dtype = dtype
    self.scale = nnx.Param(jnp.ones((dim,), dtype=weight_dtype))
    self.bias = nnx.Param(jnp.zeros((dim,), dtype=weight_dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    x_f = x.astype(jnp.float32)
    mean = jnp.mean(x_f, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_f - mean), axis=-1, keepdims=True)
    normed = (x_f - mean) * jax.lax.rsqrt(var + self.eps)
    out = normed * _use_weight(self.scale[...].astype(jnp.float32)) + _use_weight(self.bias[...].astype(jnp.float32))
    return out.astype(self.dtype)


class OLMo35ViTAttention(nnx.Module):
  """Bidirectional multi-head self-attention for SigLIP2 ViT blocks."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.embed_dim = int(config.hidden_size_for_vit)
    self.num_heads = int(config.num_attention_heads_for_vit)
    if self.embed_dim % self.num_heads != 0:
      raise ValueError(f"hidden_size_for_vit ({self.embed_dim}) must be divisible by heads ({self.num_heads}).")
    self.head_dim = self.embed_dim // self.num_heads
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)

    def _lin(in_f: int, out_f: int):
      return _ViTDense(
          in_features_shape=in_f,
          out_features_shape=out_f,
          use_bias=True,
          dtype=self.dtype,
          weight_dtype=config.weight_dtype,
          kernel_init=init,
          kernel_axes=_vit_kernel_axes(in_f, out_f, config, mesh),
          matmul_precision=config.matmul_precision,
          mesh=mesh,
          shard_mode=config.shard_mode,
          rngs=rngs,
      )

    self.wq = _lin(self.embed_dim, self.embed_dim)
    self.wk = _lin(self.embed_dim, self.embed_dim)
    self.wv = _lin(self.embed_dim, self.embed_dim)
    self.wo = _lin(self.embed_dim, self.embed_dim)

  def __call__(self, x: jax.Array) -> jax.Array:
    *prefix, n_patches, _ = x.shape
    h, d = self.num_heads, self.head_dim
    q = self.wq(x).reshape(*prefix, n_patches, h, d)
    k = self.wk(x).reshape(*prefix, n_patches, h, d)
    v = self.wv(x).reshape(*prefix, n_patches, h, d)
    scale = d**-0.5
    scores = jnp.einsum(
        "...qhd,...khd->...hqk",
        q.astype(self.dtype),
        k.astype(self.dtype),
        precision=self.config.matmul_precision,
        preferred_element_type=jnp.float32,
    ) * scale
    probs = jax.nn.softmax(scores, axis=-1).astype(self.dtype)
    attn_out = jnp.einsum(
        "...hqk,...khd->...qhd",
        probs,
        v.astype(self.dtype),
        precision=self.config.matmul_precision,
        preferred_element_type=jnp.float32,
    ).astype(self.dtype)
    return self.wo(attn_out.reshape(*prefix, n_patches, self.embed_dim))


class OLMo35ViTMLP(nnx.Module):
  """Two-layer ViT MLP with ``gelu_pytorch_tanh`` activation."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    emb_dim = int(config.hidden_size_for_vit)
    mlp_dim = int(config.intermediate_size_for_vit)
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)
    self.w1 = _ViTDense(
        in_features_shape=emb_dim,
        out_features_shape=mlp_dim,
        use_bias=True,
        dtype=self.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=init,
        kernel_axes=_vit_kernel_axes(emb_dim, mlp_dim, config, mesh),
        matmul_precision=config.matmul_precision,
        mesh=mesh,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.w2 = _ViTDense(
        in_features_shape=mlp_dim,
        out_features_shape=emb_dim,
        use_bias=True,
        dtype=self.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=init,
        kernel_axes=_vit_kernel_axes(mlp_dim, emb_dim, config, mesh),
        matmul_precision=config.matmul_precision,
        mesh=mesh,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    h = self.w1(x)
    # PyTorch F.gelu(..., approximate="tanh") computes tanh approximation in input dtype / fp32.
    h = jax.nn.gelu(h.astype(jnp.float32), approximate=True).astype(self.dtype)
    return self.w2(h)


class OLMo35ViTBlock(nnx.Module):
  """Pre-LN SigLIP2 ViT residual block."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    dim = int(config.hidden_size_for_vit)
    eps = float(config.normalization_layer_epsilon)
    self.attn_norm = _ViTLayerNorm(dim, eps, config.dtype_mm, config.weight_dtype, rngs=rngs)
    self.attn = OLMo35ViTAttention(config, mesh, rngs=rngs)
    self.ffn_norm = _ViTLayerNorm(dim, eps, config.dtype_mm, config.weight_dtype, rngs=rngs)
    self.ffn = OLMo35ViTMLP(config, mesh, rngs=rngs)

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x + self.attn(self.attn_norm(x))
    x = x + self.ffn(self.ffn_norm(x))
    return x


def _balanced_crop_tables(b_dim: int, n_crops: int, m_batch: int, m_cp: int, shifts: list[int]) -> dict[str, np.ndarray]:
  """Static all-to-all tables between the ``[B, N]`` crop layout and the balanced ``[M, B * J]`` one.

  Device ``s = fb * m_cp + e`` holds rows ``fb * B_loc ...`` and crops ``e * N_loc ...`` of the ``[B, N]`` layout.
  Crop ``c`` of row ``b`` belongs to device ``d = (c % M + shifts[b]) % M``, slot ``b * J + c // M``. Every source
  sends ``K`` (padded) crops to every device. ``send[s]``: local crop per (dest, k), ``B_loc * N_loc`` = zero pad;
  ``recv[d]``: per slot, its ``source * K + k``; ``back[d]``: slot per (source, k), ``B * J`` = zero pad;
  ``src[s]``: per local crop, its ``dest * K + k``.
  """
  m = m_batch * m_cp
  b_loc, n_loc, j_dim = b_dim // m_batch, n_crops // m_cp, n_crops // m
  slots = b_dim * j_dim
  pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
  for b in range(b_dim):
    fb, bl = divmod(b, b_loc)
    for c in range(n_crops):
      e, i = divmod(c, n_loc)
      src = fb * m_cp + e
      dst = (c % m + shifts[b]) % m
      pairs.setdefault((src, dst), []).append((bl * n_loc + i, b * j_dim + c // m))
  k_max = max(len(v) for v in pairs.values())
  send = np.full((m, m, k_max), b_loc * n_loc, dtype=np.int32)
  back = np.full((m, m, k_max), slots, dtype=np.int32)
  recv = np.zeros((m, slots), dtype=np.int32)
  src_t = np.zeros((m, b_loc * n_loc), dtype=np.int32)
  for (src, dst), items in pairs.items():
    for k, (local, slot) in enumerate(items):
      send[src, dst, k] = local
      back[dst, src, k] = slot
      recv[dst, slot] = src * k_max + k
      src_t[src, local] = dst * k_max + k
  return {
      "send": send.reshape(m, m * k_max),
      "back": back.reshape(m, m * k_max),
      "recv": recv,
      "src": src_t,
  }


class OLMo35VisionEncoder(nnx.Module):
  """SigLIP2-SO400M/14-378 25-block vision tower returning concatenated multi-layer features."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.patch_size = int(config.patch_size_for_vit)
    img_size = config.image_size_for_vit
    if isinstance(img_size, list):
      img_h, img_w = int(img_size[0]), int(img_size[1])
    else:
      img_h = img_w = int(img_size)
    self.num_patches = (img_h // self.patch_size) * (img_w // self.patch_size)
    self.patch_pixels = 3 * self.patch_size * self.patch_size
    self.embed_dim = int(config.hidden_size_for_vit)
    self.num_layers = int(config.num_hidden_layers_for_vit)
    self.feature_layers = tuple(int(i) for i in config.olmo_vit_feature_layers)
    for idx in self.feature_layers:
      if not 0 <= idx < self.num_layers:
        raise ValueError(f"olmo_vit_feature_layers entry {idx} out of range [0, {self.num_layers}).")
    self.dtype = config.dtype_mm

    self.patch_embedding = _ViTDense(
        in_features_shape=self.patch_pixels,
        out_features_shape=self.embed_dim,
        use_bias=True,
        dtype=self.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=_normal_kernel_init(0.02),
        kernel_axes=_vit_kernel_axes(self.patch_pixels, self.embed_dim, config, mesh),
        matmul_precision=config.matmul_precision,
        mesh=mesh,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    pos_init = jax.nn.initializers.normal(stddev=self.embed_dim**-0.5)
    self.positional_embedding = nnx.Param(
        pos_init(rngs.params(), (self.num_patches, self.embed_dim), config.weight_dtype)
    )
    for i in range(self.num_layers):
      setattr(self, f"blocks_{i}", OLMo35ViTBlock(config, mesh, rngs=rngs))

  def _encode_flat_crops(self, crops: jax.Array, act_pspec: PartitionSpec | None = None, probe: bool = False):
    """Encodes ``[..., M, num_patches, patch_pixels]`` into ``[..., M, num_patches, len(feature_layers) * embed_dim]``.

    ``probe`` (log_nonfinite_probe): also returns ``{"layers": c, "wused": w}``, int32 ``[num_layers + 1, ..., M]`` per
    crop: ``c[0]`` non-finite entries of the ViT input (patch embedding + positional embedding), ``c[i + 1]`` of block
    ``i``'s output; ``w[0]`` / ``w[i + 1]`` the non-finite entries of the patch/pos-embedding / block ``i`` weights as
    consumed on the device holding that crop (after the all-gather), broadcast over its crops.
    """
    lead = crops.shape[:-2]
    use_mesh = act_pspec is not None and self.mesh is not None
    probe_pspec = PartitionSpec(*tuple(act_pspec)[: len(lead)]) if use_mesh else None

    def _place(a):
      return sharding.maybe_shard_with_pspec(a, probe_pspec, self.mesh, self.config.shard_mode) if use_mesh else a

    def _crop_count(a):
      return _place(jnp.sum(jnp.logical_not(jnp.isfinite(a)), axis=(-2, -1), dtype=jnp.int32))

    def _weight_count(collected):
      total = functools.reduce(lambda u, v: u + v, collected, jnp.zeros((), jnp.int32))
      return _place(jnp.broadcast_to(total, lead))

    with _vit_layer_scope(0, probe) as w_embed:
      x = self.patch_embedding(crops.astype(self.dtype))
      x = x + _use_weight(self.positional_embedding[...].astype(self.dtype))
    if use_mesh:
      x = sharding.maybe_shard_with_pspec(x, act_pspec, self.mesh, self.config.shard_mode)
    layer_counts = [_crop_count(x)] if probe else []
    w_counts = [_weight_count(w_embed)] if probe else []
    remat = bool(self.config.olmo_vit_remat_blocks)
    wanted = set(self.feature_layers)
    saved: dict[int, jax.Array] = {}
    for i in range(self.num_layers):
      block = getattr(self, f"blocks_{i}")
      if remat:
        graphdef, state = nnx.split(block)

        def _run_block(st, inp, gd=graphdef, li=i + 1):
          mod = nnx.merge(gd, st)
          with _vit_layer_scope(li, probe) as w_blk:
            out_b = mod(inp)
          if use_mesh:
            out_b = sharding.maybe_shard_with_pspec(out_b, act_pspec, self.mesh, self.config.shard_mode)
          if probe:
            return out_b, (_crop_count(out_b), _weight_count(w_blk))
          return out_b

        res = jax.checkpoint(_run_block, prevent_cse=True)(state, x)
        if probe:
          x, (c_i, w_i) = res
          layer_counts.append(c_i)
          w_counts.append(w_i)
        else:
          x = res
      else:
        with _vit_layer_scope(i + 1, probe) as w_blk:
          x = block(x)
        if use_mesh:
          x = sharding.maybe_shard_with_pspec(x, act_pspec, self.mesh, self.config.shard_mode)
        if probe:
          layer_counts.append(_crop_count(x))
          w_counts.append(_weight_count(w_blk))
      if i in wanted:
        saved[i] = x
    feats = jnp.concatenate([saved[i] for i in self.feature_layers], axis=-1)
    if probe:
      return feats, {"layers": jnp.stack(layer_counts), "wused": jnp.stack(w_counts)}
    return feats

  def _encode_crops_maybe_chunked(self, crops: jax.Array, *, cp_sharded: bool = False, probe: bool = False):
    """Encodes ``[B, N_crops, P, D_in]`` while preserving batch (and optional CP) sharding across scan.

    ``probe``: also returns ``_encode_flat_crops``'s per-crop probes mapped back to ``[num_layers + 1, B, N_crops]``.
    """
    b_dim, n_crops, p, d_in = crops.shape
    chunk = int(self.config.olmo_vit_crop_chunk_size)
    batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
    cp_axis = self.config.context_sharding
    cp_size = self.mesh.shape.get(cp_axis, 1) if (cp_sharded and self.mesh is not None) else 1
    crops_per_rank = n_crops // cp_size if cp_size > 1 else n_crops

    if chunk > 0 and crops_per_rank > chunk and crops_per_rank % chunk == 0:
      n_chunks = crops_per_rank // chunk
      remat = bool(self.config.olmo_vit_remat_blocks)
      graphdef, state = nnx.split(self)
      if cp_size > 1:
        reshaped = crops.reshape(b_dim, cp_size, n_chunks, chunk, p, d_in)
        reshaped = sharding.maybe_shard_with_pspec(
            reshaped,
            PartitionSpec(batch_axes, cp_axis, None, None, None, None),
            self.mesh,
            self.config.shard_mode,
        )
        chunked = jnp.transpose(reshaped, (2, 0, 1, 3, 4, 5))
        chunk_pspec = PartitionSpec(batch_axes, cp_axis, None, None, None)

        def _scan_body_cp(carry, crop_chunk):
          mod = nnx.merge(graphdef, carry)
          crop_chunk = sharding.maybe_shard_with_pspec(
              crop_chunk, chunk_pspec, self.mesh, self.config.shard_mode
          )
          return carry, mod._encode_flat_crops(crop_chunk, act_pspec=chunk_pspec, probe=probe)  # pylint: disable=protected-access

        chunk_remat = remat and bool(getattr(self.config, "olmo_vit_chunk_remat", True))
        scan_fn_cp = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body_cp) if chunk_remat else _scan_body_cp
        _, out_chunks = jax.lax.scan(scan_fn_cp, state, chunked)
        if probe:
          out_chunks, pr = out_chunks
          # [n_chunks, L + 1, B, cp, chunk] -> [L + 1, B, cp, n_chunks, chunk]: crop = rank * per_rank + c * chunk + j.
          pr = {k: jnp.transpose(v, (1, 2, 3, 0, 4)).reshape(v.shape[1], b_dim, n_crops) for k, v in pr.items()}
        out = jnp.transpose(out_chunks, (1, 2, 0, 3, 4, 5)).reshape(b_dim, n_crops * p, -1)
      else:
        # Chunk along axis 1 (n_crops) and place n_chunks at axis 0 for scan so axis 1 (B)
        # remains batch-sharded across devices inside every scan iteration.
        chunked = jnp.swapaxes(crops.reshape(b_dim, n_chunks, chunk, p, d_in), 0, 1)
        chunk_pspec = PartitionSpec(batch_axes, None, None, None) if self.mesh is not None else None

        def _scan_body(carry, crop_chunk):
          mod = nnx.merge(graphdef, carry)
          return carry, mod._encode_flat_crops(crop_chunk, act_pspec=chunk_pspec, probe=probe)  # pylint: disable=protected-access

        chunk_remat = remat and bool(getattr(self.config, "olmo_vit_chunk_remat", True))
        scan_fn = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body) if chunk_remat else _scan_body
        _, out_chunks = jax.lax.scan(scan_fn, state, chunked)
        if probe:
          out_chunks, pr = out_chunks
          # [n_chunks, L + 1, B, chunk] -> [L + 1, B, n_chunks, chunk].
          pr = {k: jnp.transpose(v, (1, 2, 0, 3)).reshape(v.shape[1], b_dim, n_crops) for k, v in pr.items()}
        out = jnp.swapaxes(out_chunks, 0, 1).reshape(b_dim, n_crops * p, -1)
    else:
      flat_pspec = (
          PartitionSpec(batch_axes, cp_axis if cp_size > 1 else None, None, None)
          if self.mesh is not None
          else None
      )
      out = self._encode_flat_crops(crops, act_pspec=flat_pspec, probe=probe)
      if probe:
        out, pr = out
      out = out.reshape(b_dim, n_crops * p, -1)
    return (out, pr) if probe else out

  def _encode_balanced(self, patches: jax.Array, pooled_patches_idx: jax.Array) -> jax.Array | None:
    """Encodes ``[B, N_crops, P, D_in]`` spreading the crops over all devices and skipping unused crop chunks.

    Every crop is encoded on its own (ViT attention is per crop), and the connector only reads crops
    ``< n_used[b] = max(pooled_patches_idx[b]) // P + 1``. Crop ``c = j * M + q`` of row ``b`` (``M`` = devices on the
    batch and context axes) goes to device ``(q + b * M // B) % M``, slot ``(b, j)``: each row's leading crops land on
    all devices, so every device holds about ``sum(n_used) / M`` used crops. Each device moves its used slots to the
    front and the scan over crop chunks encodes only the chunks below the largest per-device count (one
    ``lax.cond`` predicate for all devices); the other chunks and unused crops get zero features, which the
    connector never reads. Returns ``None`` if the layout does not apply (the caller falls back).
    """
    b_dim, n_crops, p, d_in = patches.shape
    chunk = int(self.config.olmo_vit_crop_chunk_size)
    batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
    cp_axis = self.config.context_sharding
    bt = tuple(batch_axes) if isinstance(batch_axes, (tuple, list)) else ((batch_axes,) if batch_axes else ())
    dev_axes = (*bt, cp_axis)
    if cp_axis in bt:
      return None
    if int(self.mesh.shape.get("data", 1)) > 1:
      # Validated on TPU only with data = 1 (see MaxTextConfig.validate_olmo_vit_balance_crops).
      raise ValueError("olmo_vit_balance_crops=true needs a data mesh axis of 1; set olmo_vit_balance_crops=false.")
    m_batch = 1
    for a in bt:
      m_batch *= int(self.mesh.shape.get(a, 1))
    m_cp = int(self.mesh.shape.get(cp_axis, 1))
    m = m_batch * m_cp
    if m <= 1 or chunk <= 0 or n_crops % m != 0 or b_dim % m_batch != 0:
      return None
    j_dim = n_crops // m
    slots = b_dim * j_dim
    if slots % chunk != 0:
      return None
    n_chunks = slots // chunk
    shifts = [(b * m) // b_dim % m for b in range(b_dim)]
    dev_spec = dev_axes if len(dev_axes) > 1 else dev_axes[0]
    sm = self.config.shard_mode

    # Crops each row's connector reads: a prefix [0, n_used[b]).
    idx = pooled_patches_idx.astype(jnp.int32)
    n_used = jnp.where(jnp.any(idx >= 0, axis=(1, 2)), jnp.max(idx, axis=(1, 2)) // p + 1, 0)
    n_used = jnp.minimum(n_used, n_crops).astype(jnp.int32)  # [B]

    # Static crop index held by (device d, row b, slot j): j * M + (d - shift_b) % M.
    crop_of = np.zeros((m, b_dim, j_dim), dtype=np.int32)
    for b in range(b_dim):
      q = (np.arange(m) - shifts[b]) % m
      crop_of[:, b, :] = np.arange(j_dim)[None, :] * m + q[:, None]
    valid = (jnp.asarray(crop_of) < n_used[None, :, None]).reshape(m, slots)
    valid = sharding.maybe_shard_with_pspec(valid, PartitionSpec(dev_spec, None), self.mesh, sm)
    order = jnp.argsort(jnp.logical_not(valid).astype(jnp.int32), axis=1, stable=True).astype(jnp.int32)
    inv_order = jnp.argsort(order, axis=1).astype(jnp.int32)
    k_max = jnp.max(jnp.sum(valid.astype(jnp.int32), axis=1))

    # Move crops between the [B, N] (batch x context) layout and the [M, B * J] layout with one all-to-all each way.
    tabs = _balanced_crop_tables(b_dim, n_crops, m_batch, m_cp, shifts)
    in_spec = PartitionSpec(batch_axes, cp_axis, None, None)
    bal_spec = PartitionSpec(dev_spec, None, None, None)

    def _lin_index():
      lin = 0
      for a in dev_axes:
        lin = lin * int(self.mesh.shape[a]) + jax.lax.axis_index(a)
      return lin

    def _to_balanced(x_loc):
      me = _lin_index()
      flat = x_loc.reshape(-1, p, d_in)
      flat = jnp.concatenate([flat, jnp.zeros((1, p, d_in), flat.dtype)], axis=0)
      send = jnp.take(flat, jnp.asarray(tabs["send"])[me], axis=0)
      recv = jax.lax.all_to_all(send, dev_axes, 0, 0, tiled=True)
      return jnp.take(recv, jnp.asarray(tabs["recv"])[me], axis=0)[None]

    def _from_balanced(f_loc):
      me = _lin_index()
      f = f_loc[0]
      f = jnp.concatenate([f, jnp.zeros((1,) + f.shape[1:], f.dtype)], axis=0)
      send = jnp.take(f, jnp.asarray(tabs["back"])[me], axis=0)
      recv = jax.lax.all_to_all(send, dev_axes, 0, 0, tiled=True)
      out = jnp.take(recv, jnp.asarray(tabs["src"])[me], axis=0)
      return out.reshape(b_dim // m_batch, (n_crops // m_cp) * p, f.shape[-1])

    y = jax.shard_map(_to_balanced, mesh=self.mesh, in_specs=in_spec, out_specs=bal_spec, check_vma=False)(patches)
    y = sharding.maybe_shard_with_pspec(y, bal_spec, self.mesh, sm)
    y = jnp.take_along_axis(y, order[:, :, None, None], axis=1)
    y = sharding.maybe_shard_with_pspec(y, bal_spec, self.mesh, sm)
    chunked = jnp.transpose(y.reshape(m, n_chunks, chunk, p, d_in), (1, 0, 2, 3, 4))

    remat = bool(self.config.olmo_vit_remat_blocks)
    graphdef, state = nnx.split(self)
    act_pspec = PartitionSpec(dev_spec, None, None, None)

    def _encode(st, crop_chunk):
      mod = nnx.merge(graphdef, st)
      crop_chunk = sharding.maybe_shard_with_pspec(crop_chunk, act_pspec, self.mesh, sm)
      return mod._encode_flat_crops(crop_chunk, act_pspec=act_pspec)  # pylint: disable=protected-access

    out_aval = jax.eval_shape(_encode, state, jax.ShapeDtypeStruct((m, chunk, p, d_in), patches.dtype))

    def _skip(st, crop_chunk):
      del st, crop_chunk
      return sharding.maybe_shard_with_pspec(
          jnp.zeros(out_aval.shape, out_aval.dtype), act_pspec, self.mesh, sm
      )

    def _scan_body(carry, xs):
      crop_chunk, i = xs
      return carry, jax.lax.cond(i * chunk < k_max, _encode, _skip, carry, crop_chunk)

    chunk_remat = remat and bool(getattr(self.config, "olmo_vit_chunk_remat", True))
    scan_fn = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body) if chunk_remat else _scan_body
    _, out_chunks = jax.lax.scan(scan_fn, state, (chunked, jnp.arange(n_chunks, dtype=jnp.int32)))
    f_dim = out_aval.shape[-1]
    feats = jnp.transpose(out_chunks, (1, 0, 2, 3, 4)).reshape(m, slots, p, f_dim)
    feats = sharding.maybe_shard_with_pspec(feats, bal_spec, self.mesh, sm)
    feats = jnp.take_along_axis(feats, inv_order[:, :, None, None], axis=1)
    feats = jnp.where(valid[:, :, None, None], feats, jnp.zeros((), feats.dtype))
    feats = sharding.maybe_shard_with_pspec(feats, bal_spec, self.mesh, sm)
    out_spec = PartitionSpec(batch_axes, cp_axis, None)
    return jax.shard_map(_from_balanced, mesh=self.mesh, in_specs=bal_spec, out_specs=out_spec, check_vma=False)(feats)

  def __call__(
      self,
      input_images: jax.Array,
      deterministic: bool = True,
      pooled_patches_idx: jax.Array | None = None,
      return_probes: bool = False,
  ):
    """Encodes the crops; with ``pooled_patches_idx`` the padding crops (``>= n_used`` per row) are zeroed by selection
    before the ViT (so their activations, which enter the ViT weight gradients, are finite whatever the pixel buffer
    holds) and their features after it. Valid crops are unchanged.

    ``return_probes``: also returns a dict of per-row int32 [B] counts (``log_nonfinite_probe``), plus per crop block
    (``K`` = size of the context mesh axis; block ``k`` = crops ``[k * N / K, (k + 1) * N / K)``, the crops CP rank
    ``k`` encodes on the CP-sharded path):
      ``nonfinite_probe_vit_feats_blocks`` ``[B, K]``: non-finite ViT output entries (all crops, before the mask);
      ``nonfinite_probe_vit_layer_blocks`` ``[L + 1, B, K]``: index 0 = ViT input (patch + pos embedding), ``i + 1`` =
        output of block ``i`` (not on the balanced path);
      ``nonfinite_probe_vit_wused_blocks`` ``[L + 1, B, K]``: non-finite entries of that stage's weights as consumed
        (after the all-gather) on the device(s) encoding the block (max over the block's crops / chunks).
    """
    del deterministic
    if input_images.ndim == 5:
      patches = patchify_hwc_to_olmo_cfirst(input_images, self.patch_size)
    elif input_images.ndim == 4:
      patches = input_images
    else:
      raise ValueError(
          f"OLMo35VisionEncoder expects [B, N_crops, P, 588] or [B, N_crops, H, W, 3], got {input_images.shape}."
      )

    b_dim, n_crops, p = patches.shape[0], patches.shape[1], patches.shape[2]
    crop_valid = None if pooled_patches_idx is None else olmo35_crop_valid(pooled_patches_idx, n_crops, p)
    probes = {}
    cp_axis = self.config.context_sharding
    cp_size = self.mesh.shape.get(cp_axis, 1) if self.mesh is not None else 1
    k_blocks = cp_size if n_crops % cp_size == 0 else 1

    def _blocks(per_crop, reduce):  # [..., B, N_crops] -> [..., B, K]
      return reduce(per_crop.reshape(per_crop.shape[:-1] + (k_blocks, n_crops // k_blocks)), axis=-1)

    def _layer_probes(pr):
      if return_probes and pr is not None:
        probes["nonfinite_probe_vit_layer_blocks"] = _blocks(pr["layers"], jnp.sum)
        probes["nonfinite_probe_vit_wused_blocks"] = _blocks(pr["wused"], jnp.max)

    if return_probes:
      probes["nonfinite_probe_vit_params"] = jnp.broadcast_to(params_nonfinite_count(self), (b_dim,))
      # Pixels outside [-1, 1] (SigLIP range) or non-finite: catches finite garbage the plain non-finite count misses.
      bad_px = jnp.logical_not(jnp.abs(patches.astype(jnp.float32)) <= 1.0 + 1e-3)
      probes["nonfinite_probe_pixels_out_of_range"] = jnp.sum(bad_px, axis=(1, 2, 3), dtype=jnp.int32)

    def _finish(out, feat_pspec):
      # out: [B, N_crops * P, F]. Probe, then zero the padding crops' features by selection.
      f_dim = out.shape[-1]
      if return_probes:
        bad = jnp.logical_not(jnp.isfinite(out)).reshape(b_dim, n_crops, p, f_dim)
        probes["nonfinite_probe_vit_feats"] = jnp.sum(bad, axis=(1, 2, 3), dtype=jnp.int32)
        probes["nonfinite_probe_vit_feats_blocks"] = _blocks(jnp.sum(bad, axis=(2, 3), dtype=jnp.int32), jnp.sum)
        if crop_valid is not None:
          probes["nonfinite_probe_vit_feats_valid"] = jnp.sum(
              jnp.logical_and(bad, crop_valid[:, :, None, None]), axis=(1, 2, 3), dtype=jnp.int32
          )
      if crop_valid is not None:
        out = _select_crops(out.reshape(b_dim, n_crops, p, f_dim), crop_valid).reshape(b_dim, n_crops * p, f_dim)
        if feat_pspec is not None and self.mesh is not None:
          out = sharding.maybe_shard_with_pspec(out, feat_pspec, self.mesh, self.config.shard_mode)
      return (out, probes) if return_probes else out

    if bool(self.config.olmo_vit_cp_shard_crops) and cp_size > 1 and n_crops % cp_size == 0:
      # Every CP rank shares the same sequence's `n_crops` images. Shard `patches` on `cp_axis`
      # in GSPMD so each CP rank encodes `n_crops // cp_size` crops while ViT weights stay
      # FSDP/EP-sharded (`vit_embed` / `vit_embed_cp`), then AllGather features across `cp_axis`.
      batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
      in_pspec = PartitionSpec(batch_axes, cp_axis, None, None)
      patches = sharding.maybe_shard_with_pspec(patches, in_pspec, self.mesh, self.config.shard_mode)
      if crop_valid is not None:
        patches = sharding.maybe_shard_with_pspec(
            _select_crops(patches, crop_valid), in_pspec, self.mesh, self.config.shard_mode
        )
      out = None
      if bool(getattr(self.config, "olmo_vit_balance_crops", False)) and pooled_patches_idx is not None:
        out = self._encode_balanced(patches, pooled_patches_idx)
      if out is None:
        out = self._encode_crops_maybe_chunked(patches, cp_sharded=True, probe=return_probes)
        if return_probes:
          out, pr = out
          _layer_probes(pr)
      cp_feat_pspec = PartitionSpec(batch_axes, cp_axis, None)
      out = sharding.maybe_shard_with_pspec(out, cp_feat_pspec, self.mesh, self.config.shard_mode)
      res = _finish(out, cp_feat_pspec)
      out = res[0] if return_probes else res
      if not bool(getattr(self.config, "olmo_connector_gather_once", False)):
        out = sharding.maybe_shard_with_pspec(
            out, PartitionSpec(batch_axes, None, None), self.mesh, self.config.shard_mode
        )
      # (olmo_connector_gather_once: the connector all-gathers the features itself and reduce-scatters their gradient.)
      return (out, probes) if return_probes else out

    if self.mesh is not None:
      patches = sharding.maybe_shard_with_logical(
          patches, ("activation_batch", None, None, None), self.mesh, self.config.shard_mode, self.config.logical_axis_rules
      )
    if crop_valid is not None:
      patches = _select_crops(patches, crop_valid)
    out = self._encode_crops_maybe_chunked(patches, probe=return_probes)
    if return_probes:
      out, pr = out
      _layer_probes(pr)
    feat_pspec = PartitionSpec(_batch_mesh_spec_entry(self.config, self.mesh), None, None) if self.mesh is not None else None
    return _finish(out, feat_pspec)


def _few_key_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    valid: jax.Array | None,
) -> jax.Array:
  """FP32 group pooling attention matching ``olmo_core.nn.vision.connector._few_key_attention``.

  Args:
    q: ``[..., H, D]``
    k: ``[..., P, H, D]``
    v: ``[..., P, H, D]``
    valid: ``[..., P]`` bool, ``True`` where key is valid (``pooled_patches_idx >= 0``).
  Returns:
    ``[..., H, D]`` in ``q.dtype``, zero on groups where all slots are ``-1``.
  """
  k = k.astype(jnp.float32)
  if valid is not None:
    k = jnp.where(valid[..., None, None], k, 0.0)  # selection: an invalid key can not reach q's gradient either
  scores = jnp.sum(q.astype(jnp.float32)[..., None, :, :] * k, axis=-1) * (q.shape[-1] ** -0.5)
  if valid is not None:
    # Finite fill (finfo.min) keeps softmax and its backward finite on all-(-1) padding rows.
    fmin = jnp.finfo(jnp.float32).min
    scores = jnp.where(valid[..., None], scores, fmin)
  probs = jax.nn.softmax(scores, axis=-2)
  v = v.astype(jnp.float32)
  if valid is not None:
    probs = jnp.where(valid[..., None], probs, 0.0)
    # Select the invalid keys' values too (not only multiply them by a zero prob): a non-finite value there then
    # reaches neither the sum (0 * NaN) nor the probs' gradient (g * NaN).
    v = jnp.where(valid[..., None, None], v, 0.0)
  return jnp.sum(probs[..., None] * v, axis=-3).astype(q.dtype)


class OLMo35PoolingCrossAttention(nnx.Module):
  """``attention_meanq`` 2x2 patch-group cross-attention pooling."""

  def __init__(self, config: Config, mesh: Mesh, input_dim: int, *, rngs: nnx.Rngs):
    self.config = config
    self.emb_dim = int(config.hidden_size_for_vit)
    self.num_heads = int(config.num_attention_heads_for_vit)
    self.head_dim = self.emb_dim // self.num_heads
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)

    def _lin(in_f: int, out_f: int):
      return _ViTDense(
          in_features_shape=in_f,
          out_features_shape=out_f,
          use_bias=True,
          dtype=self.dtype,
          weight_dtype=config.weight_dtype,
          kernel_init=init,
          kernel_axes=_vit_kernel_axes(in_f, out_f, config, mesh),
          matmul_precision=config.matmul_precision,
          mesh=mesh,
          shard_mode=config.shard_mode,
          rngs=rngs,
      )

    self.wq = _lin(input_dim, self.emb_dim)
    self.wk = _lin(input_dim, self.emb_dim)
    self.wv = _lin(input_dim, self.emb_dim)
    self.wo = _lin(self.emb_dim, self.emb_dim)

  def __call__(self, query: jax.Array, context: jax.Array, valid: jax.Array | None) -> jax.Array:
    """Pools ``context`` ``[..., P, input_dim]`` with ``query`` ``[..., input_dim]`` -> ``[..., emb_dim]``."""
    *prefix, p, _ = context.shape
    q = self.wq(query).reshape(*prefix, self.num_heads, self.head_dim)
    k = self.wk(context).reshape(*prefix, p, self.num_heads, self.head_dim)
    v = self.wv(context).reshape(*prefix, p, self.num_heads, self.head_dim)
    out = _few_key_attention(q, k, v, valid)
    return self.wo(out.reshape(*prefix, self.emb_dim))


class OLMo35ConnectorMLP(nnx.Module):
  """Bias-free SwiGLU projector ``w2(silu(w1(x)) * w3(x))`` into LM ``emb_dim``."""

  def __init__(self, config: Config, mesh: Mesh, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)

    def _lin(in_f: int, out_f: int):
      return _ViTDense(
          in_features_shape=in_f,
          out_features_shape=out_f,
          use_bias=False,
          dtype=self.dtype,
          weight_dtype=config.weight_dtype,
          kernel_init=init,
          kernel_axes=_vit_kernel_axes(in_f, out_f, config, mesh),
          matmul_precision=config.matmul_precision,
          mesh=mesh,
          shard_mode=config.shard_mode,
          rngs=rngs,
      )

    self.w1 = _lin(input_dim, hidden_dim)
    self.w3 = _lin(input_dim, hidden_dim)
    self.w2 = _lin(hidden_dim, output_dim)

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = jax.nn.silu(self.w1(x).astype(jnp.float32)).astype(self.dtype)
    return self.w2(gate * self.w3(x))


class OLMo35VisionConnector(nnx.Module):
  """Molmo2 / OLMo 3.5 gather-pool-project vision connector."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.dtype = config.dtype_mm
    num_layers_used = len(config.olmo_vit_feature_layers)
    pooling_input_dim = num_layers_used * int(config.hidden_size_for_vit)
    hidden_dim = int(config.olmo_connector_mlp_hidden_size)
    self.pooling = OLMo35PoolingCrossAttention(config, mesh, pooling_input_dim, rngs=rngs)
    self.projector = OLMo35ConnectorMLP(
        config,
        mesh,
        int(config.hidden_size_for_vit),
        hidden_dim,
        int(config.emb_dim),
        rngs=rngs,
    )

  def _forward_connector(
      self,
      image_features: jax.Array,
      pooled_patches_idx: jax.Array,
      act_pspec: PartitionSpec | None = None,
  ) -> jax.Array:
    b, n_total, dim = image_features.shape
    n_pooled, pool_size = pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]
    safe_idx = jnp.clip(pooled_patches_idx, 0, max(n_total - 1, 0))
    # Gather strictly within axis 1 (n_total) per batch item so axis 0 (B) stays shard-local.
    gather_idx = safe_idx.reshape(b, n_pooled * pool_size, 1)
    if act_pspec is not None and self.mesh is not None:
      gather_idx = sharding.maybe_shard_with_pspec(gather_idx, act_pspec, self.mesh, self.config.shard_mode)
    to_pool = jnp.take_along_axis(image_features, gather_idx, axis=1).reshape(b, n_pooled, pool_size, dim)
    return self._pool_and_project(to_pool, pooled_patches_idx, act_pspec=act_pspec)

  def _pool_and_project(
      self,
      to_pool: jax.Array,
      pooled_patches_idx: jax.Array,
      act_pspec: PartitionSpec | None = None,
  ) -> jax.Array:
    """Pools the gathered ``to_pool`` ``[B, N_pooled, pool_size, dim]`` features and projects them to ``emb_dim``."""
    valid = pooled_patches_idx >= 0
    if act_pspec is not None and self.mesh is not None:
      pool4_pspec = PartitionSpec(act_pspec[0], act_pspec[1], None, None)
      to_pool = sharding.maybe_shard_with_pspec(to_pool, pool4_pspec, self.mesh, self.config.shard_mode)
    to_pool = jnp.where(valid[..., None], to_pool, jnp.zeros((), dtype=to_pool.dtype))

    denom = jnp.maximum(jnp.sum(valid.astype(jnp.int32), axis=-1, keepdims=True), 1).astype(to_pool.dtype)
    query = jnp.sum(to_pool, axis=2) / denom
    pooled = self.pooling(query, to_pool, valid)
    if act_pspec is not None and self.mesh is not None:
      pooled = sharding.maybe_shard_with_pspec(pooled, act_pspec, self.mesh, self.config.shard_mode)
    projected = self.projector(pooled)
    row_valid = jnp.any(valid, axis=-1, keepdims=True)
    out = jnp.where(row_valid, projected, jnp.zeros((), dtype=projected.dtype))
    if act_pspec is not None and self.mesh is not None:
      out = sharding.maybe_shard_with_pspec(out, act_pspec, self.mesh, self.config.shard_mode)
    return out

  def _forward_connector_gather_once(self, image_features: jax.Array, pooled_patches_idx: jax.Array) -> jax.Array:
    """CP connector: one gather per micro-batch instead of one per chunk (``olmo_connector_gather_once``).

    ``image_features`` ``[B, N_total, dim]`` arrive sharded on the context axis. A shard_map all-gathers them on the
    context axis and gathers each rank's pooled patches (``[B, N_pooled / CP, pool_size, dim]``); in the backward
    that is one local scatter-add plus one reduce-scatter. The chunk scan then pools and projects slices of the
    gathered patches with the same per-chunk math as ``_forward_connector_maybe_chunked``.
    """
    b_dim, n_pooled, pool_size = pooled_patches_idx.shape
    n_total, dim = image_features.shape[1], image_features.shape[2]
    batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
    cp_axis = self.config.context_sharding
    cp_size = self.mesh.shape.get(cp_axis, 1)
    sm = self.config.shard_mode
    feat_spec = PartitionSpec(batch_axes, cp_axis, None)
    pool_spec = PartitionSpec(batch_axes, cp_axis, None, None)
    image_features = sharding.maybe_shard_with_pspec(image_features, feat_spec, self.mesh, sm)
    pooled_patches_idx = sharding.maybe_shard_with_pspec(pooled_patches_idx, feat_spec, self.mesh, sm)

    def _gather(f_loc, i_loc):
      full = jax.lax.all_gather(f_loc, cp_axis, axis=1, tiled=True)
      bl, npl, ps = i_loc.shape
      safe = jnp.clip(i_loc, 0, max(n_total - 1, 0)).reshape(bl, npl * ps, 1)
      return jnp.take_along_axis(full, safe, axis=1).reshape(bl, npl, ps, full.shape[-1])

    to_pool = jax.shard_map(
        _gather, mesh=self.mesh, in_specs=(feat_spec, feat_spec), out_specs=pool_spec, check_vma=False
    )(image_features, pooled_patches_idx)
    to_pool = sharding.maybe_shard_with_pspec(to_pool, pool_spec, self.mesh, sm)

    pooled_per_rank = n_pooled // cp_size
    chunk = int(getattr(self.config, "olmo_vit_crop_chunk_size", 0))
    remat = bool(getattr(self.config, "olmo_vit_remat_blocks", True))
    chunk_remat = remat and bool(getattr(self.config, "olmo_vit_chunk_remat", True))
    skip_empty = bool(getattr(self.config, "olmo_connector_skip_empty_chunks", False))
    chunk_pooled = 0
    if chunk > 0:
      for mult in (196, 128, 1):
        cand = chunk * mult
        if pooled_per_rank > cand and pooled_per_rank % cand == 0:
          chunk_pooled = cand
          break
    if chunk_pooled == 0:
      chunk_pooled = pooled_per_rank
    n_chunks = pooled_per_rank // chunk_pooled
    graphdef, state = nnx.split(self)

    tp = to_pool.reshape(b_dim, cp_size, n_chunks, chunk_pooled, pool_size, dim)
    tp = sharding.maybe_shard_with_pspec(
        tp, PartitionSpec(batch_axes, cp_axis, None, None, None, None), self.mesh, sm
    )
    tp = jnp.transpose(tp, (2, 0, 1, 3, 4, 5))
    ix = pooled_patches_idx.reshape(b_dim, cp_size, n_chunks, chunk_pooled, pool_size)
    ix = sharding.maybe_shard_with_pspec(ix, PartitionSpec(batch_axes, cp_axis, None, None, None), self.mesh, sm)
    ix = jnp.transpose(ix, (2, 0, 1, 3, 4))

    def _run(st, tp_chunk, idx_chunk):
      mod = nnx.merge(graphdef, st)
      tp_flat = tp_chunk.reshape(b_dim, cp_size * chunk_pooled, pool_size, dim)
      tp_flat = sharding.maybe_shard_with_pspec(tp_flat, pool_spec, self.mesh, sm)
      idx_flat = idx_chunk.reshape(b_dim, cp_size * chunk_pooled, pool_size)
      idx_flat = sharding.maybe_shard_with_pspec(idx_flat, feat_spec, self.mesh, sm)
      out_flat = mod._pool_and_project(tp_flat, idx_flat, act_pspec=feat_spec)  # pylint: disable=protected-access
      return out_flat.reshape(b_dim, cp_size, chunk_pooled, -1)

    if skip_empty:
      out_aval = jax.eval_shape(
          _run,
          state,
          jax.ShapeDtypeStruct((b_dim, cp_size, chunk_pooled, pool_size, dim), to_pool.dtype),
          jax.ShapeDtypeStruct((b_dim, cp_size, chunk_pooled, pool_size), pooled_patches_idx.dtype),
      )
      out_chunk_spec = PartitionSpec(batch_axes, cp_axis, None, None)

      def _skip(st, tp_chunk, idx_chunk):
        del st, tp_chunk, idx_chunk
        return sharding.maybe_shard_with_pspec(jnp.zeros(out_aval.shape, out_aval.dtype), out_chunk_spec, self.mesh, sm)

      def _scan_body(carry, xs):
        tp_chunk, idx_chunk = xs
        return carry, jax.lax.cond(jnp.any(idx_chunk >= 0), _run, _skip, carry, tp_chunk, idx_chunk)

    else:

      def _scan_body(carry, xs):
        tp_chunk, idx_chunk = xs
        return carry, _run(carry, tp_chunk, idx_chunk)

    scan_fn = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body) if chunk_remat else _scan_body
    _, out_chunks = jax.lax.scan(scan_fn, state, (tp, ix))
    return jnp.transpose(out_chunks, (1, 2, 0, 3, 4)).reshape(b_dim, n_pooled, -1)

  def _forward_connector_maybe_chunked(
      self,
      image_features: jax.Array,
      pooled_patches_idx: jax.Array,
      *,
      cp_sharded: bool = False,
  ) -> jax.Array:
    """Pools and projects ``pooled_patches_idx`` ``[B, N_pooled, 4]`` with optional CP sharding and scan chunking."""
    b_dim, n_pooled, pool_size = pooled_patches_idx.shape
    chunk = int(getattr(self.config, "olmo_vit_crop_chunk_size", 0))
    batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
    cp_axis = self.config.context_sharding
    cp_size = self.mesh.shape.get(cp_axis, 1) if (cp_sharded and self.mesh is not None) else 1
    pooled_per_rank = n_pooled // cp_size if cp_size > 1 else n_pooled
    remat = bool(getattr(self.config, "olmo_vit_remat_blocks", True))

    chunk_pooled = 0
    if chunk > 0:
      for mult in (196, 128, 1):
        cand = chunk * mult
        if pooled_per_rank > cand and pooled_per_rank % cand == 0:
          chunk_pooled = cand
          break

    if chunk_pooled > 0:
      n_chunks = pooled_per_rank // chunk_pooled
      graphdef, state = nnx.split(self)
      if cp_size > 1:
        reshaped = pooled_patches_idx.reshape(b_dim, cp_size, n_chunks, chunk_pooled, pool_size)
        reshaped = sharding.maybe_shard_with_pspec(
            reshaped,
            PartitionSpec(batch_axes, cp_axis, None, None, None),
            self.mesh,
            self.config.shard_mode,
        )
        chunked = jnp.transpose(reshaped, (2, 0, 1, 3, 4))
        flat_pspec = PartitionSpec(batch_axes, cp_axis, None)

        def _scan_body_cp(carry, idx_chunk):
          mod = nnx.merge(graphdef, carry)
          idx_flat = idx_chunk.reshape(b_dim, cp_size * chunk_pooled, pool_size)
          idx_flat = sharding.maybe_shard_with_pspec(idx_flat, flat_pspec, self.mesh, self.config.shard_mode)
          out_flat = mod._forward_connector(image_features, idx_flat, act_pspec=flat_pspec)  # pylint: disable=protected-access
          return carry, out_flat.reshape(b_dim, cp_size, chunk_pooled, -1)

        chunk_remat = remat and bool(getattr(self.config, "olmo_vit_chunk_remat", True))
        scan_fn_cp = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body_cp) if chunk_remat else _scan_body_cp
        _, out_chunks = jax.lax.scan(scan_fn_cp, state, chunked)
        return jnp.transpose(out_chunks, (1, 2, 0, 3, 4)).reshape(b_dim, n_pooled, -1)

      chunked = jnp.swapaxes(pooled_patches_idx.reshape(b_dim, n_chunks, chunk_pooled, pool_size), 0, 1)
      flat_pspec = PartitionSpec(batch_axes, None, None) if self.mesh is not None else None

      def _scan_body(carry, idx_chunk):
        mod = nnx.merge(graphdef, carry)
        return carry, mod._forward_connector(image_features, idx_chunk, act_pspec=flat_pspec)  # pylint: disable=protected-access

      scan_fn = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body) if remat else _scan_body
      _, out_chunks = jax.lax.scan(scan_fn, state, chunked)
      return jnp.swapaxes(out_chunks, 0, 1).reshape(b_dim, n_pooled, -1)

    flat_pspec = (
        PartitionSpec(batch_axes, cp_axis if cp_size > 1 else None, None)
        if self.mesh is not None
        else None
    )
    if not remat:
      return self._forward_connector(image_features, pooled_patches_idx, act_pspec=flat_pspec)

    graphdef, state = nnx.split(self)

    @functools.partial(jax.checkpoint, prevent_cse=True)
    def _remat_connector(st, feats, p_idx):
      mod = nnx.merge(graphdef, st)
      return mod._forward_connector(feats, p_idx, act_pspec=flat_pspec)  # pylint: disable=protected-access

    return _remat_connector(state, image_features, pooled_patches_idx)

  def __call__(self, image_features: jax.Array, pooled_patches_idx: jax.Array | None = None) -> jax.Array:
    """Pools ``image_features`` ``[B, N_total_patches, 2304]`` via ``pooled_patches_idx`` ``[B, N_pooled, 4]``."""
    b, n_total, _ = image_features.shape
    if pooled_patches_idx is None:
      pool_size = int(self.config.olmo_connector_pool_size)
      cfg_pooled = int(getattr(self.config, "olmo_max_pooled_patches_per_seq", 0))
      if cfg_pooled > 0:
        n_pooled = cfg_pooled
        pooled_patches_idx = jnp.broadcast_to(
            (jnp.arange(n_pooled * pool_size, dtype=jnp.int32) % max(n_total, 1)).reshape(1, n_pooled, pool_size),
            (b, n_pooled, pool_size),
        )
      else:
        if n_total % pool_size != 0:
          raise ValueError(
              f"pooled_patches_idx was not provided and total patches ({n_total}) is not divisible by {pool_size}."
          )
        n_pooled = n_total // pool_size
        pooled_patches_idx = jnp.broadcast_to(
            jnp.arange(n_total, dtype=jnp.int32).reshape(1, n_pooled, pool_size),
            (b, n_pooled, pool_size),
        )
    else:
      pooled_patches_idx = pooled_patches_idx.astype(jnp.int32)

    cp_axis = self.config.context_sharding
    cp_size = self.mesh.shape.get(cp_axis, 1) if self.mesh is not None else 1
    n_pooled = pooled_patches_idx.shape[1]
    if (
        bool(self.config.olmo_vit_cp_shard_crops)
        and bool(getattr(self.config, "olmo_connector_gather_once", False))
        and cp_size > 1
        and n_pooled % cp_size == 0
        and n_total % cp_size == 0
    ):
      batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
      out = self._forward_connector_gather_once(image_features, pooled_patches_idx)
      out = sharding.maybe_shard_with_pspec(
          out, PartitionSpec(batch_axes, cp_axis, None), self.mesh, self.config.shard_mode
      )
      return sharding.maybe_shard_with_pspec(
          out, PartitionSpec(batch_axes, None, None), self.mesh, self.config.shard_mode
      )

    if bool(self.config.olmo_vit_cp_shard_crops) and cp_size > 1 and n_pooled % cp_size == 0:
      batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
      image_features = sharding.maybe_shard_with_pspec(
          image_features, PartitionSpec(batch_axes, None, None), self.mesh, self.config.shard_mode
      )
      pooled_patches_idx = sharding.maybe_shard_with_pspec(
          pooled_patches_idx, PartitionSpec(batch_axes, cp_axis, None), self.mesh, self.config.shard_mode
      )
      out = self._forward_connector_maybe_chunked(image_features, pooled_patches_idx, cp_sharded=True)
      out = sharding.maybe_shard_with_pspec(
          out, PartitionSpec(batch_axes, cp_axis, None), self.mesh, self.config.shard_mode
      )
      return sharding.maybe_shard_with_pspec(
          out, PartitionSpec(batch_axes, None, None), self.mesh, self.config.shard_mode
      )

    if self.mesh is not None:
      image_features = sharding.maybe_shard_with_logical(
          image_features, ("activation_batch", None, None), self.mesh, self.config.shard_mode, self.config.logical_axis_rules
      )
      pooled_patches_idx = sharding.maybe_shard_with_logical(
          pooled_patches_idx, ("activation_batch", None, None), self.mesh, self.config.shard_mode, self.config.logical_axis_rules
      )
    return self._forward_connector_maybe_chunked(image_features, pooled_patches_idx, cp_sharded=False)


def merge_olmo35_image_embeddings(
    text_embeddings: jax.Array,
    decoder_input_tokens: jax.Array,
    image_embeddings: jax.Array,
    *,
    patch_token_id: int,
    mesh: Mesh | None,
    config: Config,
) -> jax.Array:
  """Adds ``image_embeddings`` ``[B, N_pooled, D]`` (``+=``) at ``<im_patch>`` positions after ``embedding_norm``.

  Supports Context Parallelism (``CP > 1``): ``decoder_input_tokens`` and ``text_embeddings``
  are sharded on ``("activation_batch", "activation_norm_length", ...)``, while
  ``image_embeddings`` holds all ``N_pooled`` pooled tokens for each sequence in the batch.
  """
  is_patch = decoder_input_tokens == int(patch_token_id)
  n_pooled = image_embeddings.shape[1]
  cp_axis = config.context_sharding
  cp_size = mesh.shape.get(cp_axis, 1) if mesh is not None else 1

  if cp_size > 1 and mesh is not None:
    batch_axes = _batch_mesh_spec_entry(config, mesh)
    emb_spec = PartitionSpec(batch_axes, cp_axis, None)
    tok_spec = PartitionSpec(batch_axes, cp_axis)
    img_spec = PartitionSpec(batch_axes, None, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(emb_spec, tok_spec, img_spec),
        out_specs=emb_spec,
        check_vma=False,
    )
    def _cp_add(y_loc, is_patch_loc, img_emb_loc):
      b_loc, t_loc, _ = y_loc.shape
      is_patch_full = jax.lax.all_gather(is_patch_loc, cp_axis, axis=1, tiled=True)
      full_ranks = jnp.cumsum(is_patch_full.astype(jnp.int32), axis=1) - 1
      c_idx = jax.lax.axis_index(cp_axis)
      local_ranks = jax.lax.dynamic_slice_in_dim(full_ranks, c_idx * t_loc, t_loc, axis=1)
      safe_ranks = jnp.clip(local_ranks, 0, max(n_pooled - 1, 0))
      gathered = jnp.take_along_axis(img_emb_loc, safe_ranks[..., None], axis=1)
      delta = jnp.where(is_patch_loc[..., None], gathered.astype(y_loc.dtype), jnp.zeros((), dtype=y_loc.dtype))
      return y_loc + delta

    return _cp_add(text_embeddings, is_patch, image_embeddings)

  ranks = jnp.cumsum(is_patch.astype(jnp.int32), axis=1) - 1
  safe_ranks = jnp.clip(ranks, 0, max(n_pooled - 1, 0))
  gathered = jnp.take_along_axis(image_embeddings, safe_ranks[..., None], axis=1)
  delta = jnp.where(is_patch[..., None], gathered.astype(text_embeddings.dtype), jnp.zeros((), dtype=text_embeddings.dtype))
  return text_embeddings + delta


def olmo_core_vision_to_maxtext_params(olmo_state: dict[str, Any], num_layers: int = 25) -> dict[str, Any]:
  """Maps OLMo-core ``MultimodalLM`` ``vision.*`` and ``connector.*`` tensors (numpy/array) to MaxText flat paths.

  Keys follow ``olmo_core.nn.maxtext.convert``'s ``/``-joined unscanned convention under
  ``vision_encoder/OLMo35VisionEncoder_0/...`` and ``vision_encoder/OLMo35VisionConnector_0/...``.
  """
  enc = "vision_encoder/OLMo35VisionEncoder_0"
  con = "vision_encoder/OLMo35VisionConnector_0"
  out: dict[str, Any] = {
      f"{enc}/patch_embedding/kernel": olmo_state["vision.patch_embedding.weight"].T,
      f"{enc}/patch_embedding/bias": olmo_state["vision.patch_embedding.bias"],
      f"{enc}/positional_embedding": olmo_state["vision.positional_embedding"],
  }
  for i in range(num_layers):
    src = f"vision.blocks.{i}"
    dst = f"{enc}/blocks_{i}"
    for norm_name in ("attn_norm", "ffn_norm"):
      out[f"{dst}/{norm_name}/scale"] = olmo_state[f"{src}.{norm_name}.weight"]
      out[f"{dst}/{norm_name}/bias"] = olmo_state[f"{src}.{norm_name}.bias"]
    for proj in ("wq", "wk", "wv", "wo"):
      out[f"{dst}/attn/{proj}/kernel"] = olmo_state[f"{src}.attn.{proj}.weight"].T
      out[f"{dst}/attn/{proj}/bias"] = olmo_state[f"{src}.attn.{proj}.bias"]
    for proj in ("w1", "w2"):
      out[f"{dst}/ffn/{proj}/kernel"] = olmo_state[f"{src}.ffn.{proj}.weight"].T
      out[f"{dst}/ffn/{proj}/bias"] = olmo_state[f"{src}.ffn.{proj}.bias"]

  for proj in ("wq", "wk", "wv", "wo"):
    out[f"{con}/pooling/{proj}/kernel"] = olmo_state[f"connector.pooling.{proj}.weight"].T
    out[f"{con}/pooling/{proj}/bias"] = olmo_state[f"connector.pooling.{proj}.bias"]
  for proj in ("w1", "w2", "w3"):
    out[f"{con}/projector/{proj}/kernel"] = olmo_state[f"connector.projector.{proj}.weight"].T
  return out


def maxtext_vision_to_olmo_core_params(mt_state: dict[str, Any], num_layers: int = 25) -> dict[str, Any]:
  """Inverse of :func:`olmo_core_vision_to_maxtext_params` (bit-exact transpose/rename)."""
  enc = "vision_encoder/OLMo35VisionEncoder_0"
  con = "vision_encoder/OLMo35VisionConnector_0"
  out: dict[str, Any] = {
      "vision.patch_embedding.weight": mt_state[f"{enc}/patch_embedding/kernel"].T,
      "vision.patch_embedding.bias": mt_state[f"{enc}/patch_embedding/bias"],
      "vision.positional_embedding": mt_state[f"{enc}/positional_embedding"],
  }
  for i in range(num_layers):
    src = f"{enc}/blocks_{i}"
    dst = f"vision.blocks.{i}"
    for norm_name in ("attn_norm", "ffn_norm"):
      out[f"{dst}.{norm_name}.weight"] = mt_state[f"{src}/{norm_name}/scale"]
      out[f"{dst}.{norm_name}.bias"] = mt_state[f"{src}/{norm_name}/bias"]
    for proj in ("wq", "wk", "wv", "wo"):
      out[f"{dst}.attn.{proj}.weight"] = mt_state[f"{src}/attn/{proj}/kernel"].T
      out[f"{dst}.attn.{proj}.bias"] = mt_state[f"{src}/attn/{proj}/bias"]
    for proj in ("w1", "w2"):
      out[f"{dst}.ffn.{proj}.weight"] = mt_state[f"{src}/ffn/{proj}/kernel"].T
      out[f"{dst}.ffn.{proj}.bias"] = mt_state[f"{src}/ffn/{proj}/bias"]

  for proj in ("wq", "wk", "wv", "wo"):
    out[f"connector.pooling.{proj}.weight"] = mt_state[f"{con}/pooling/{proj}/kernel"].T
    out[f"connector.pooling.{proj}.bias"] = mt_state[f"{con}/pooling/{proj}/bias"]
  for proj in ("w1", "w2", "w3"):
    out[f"connector.projector.{proj}.weight"] = mt_state[f"{con}/projector/{proj}/kernel"].T
  return out
