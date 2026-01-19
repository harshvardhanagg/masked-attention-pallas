"""Custom masked attention for sequential recommendation models.

This module implements a custom masked multi-head attention kernel where:
- History tokens (first history_len positions) attend bidirectionally to each other
- Candidate tokens (last candidate_len positions) attend only to history tokens

This is useful for recommendation/ranking tasks where candidates should be scored
independently conditioned on the history.
"""
from __future__ import annotations

import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import numpy as np
import dataclasses

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
  """
  Tile sizes parameterizing the attention kernel. These block sizes
  should be tuned for the model and hardware for optimal performance.

  Attributes:
    block_q: Block size along Q sequence length for forward kernel.
    block_k: Block size along KV sequence length for forward kernel.
  """
  block_q: int
  block_k: int

  @classmethod
  def get_default(cls):
    return BlockSizes(
        block_q=128,
        block_k=128,
    )


def custom_masked_attention_forward_kernel(
    q_ref, k_ref, v_ref, o_ref, *residual_refs,
    history_len: int, sm_scale: float, block_q: int, block_k: int, head_dim: int,
):
  """
  Forward kernel for structured masked attention:
  - History tokens attend to history (this kernel only iterates over history KV).
  - Candidate tokens attend only to history KV.

  This version is optimized for H100 + head_dim=64 and supports "q_tile" that may
  contain multiple Q blocks per program. It avoids the big slowdown you saw by:
    - NOT materializing [q_tile, block_k] softmax probs for a large q_tile at once
    - Splitting q_tile into sub-blocks of size block_q and reusing each loaded KV block
      across those sub-blocks (the correct "KV outer" pattern).
    - Only masking the last KV block when history_len % block_k != 0.
    - Unrolling the <=4 KV blocks (since history_len is static and block_k=128).
  """
  q_rows = q_ref.shape[0]              # may be block_q, 2*block_q, 4*block_q, ...
  head_dim_padded = q_ref.shape[-1]
  head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]

  # Scale (base-2 exp path)
  qk_scale = math.log2(math.e)
  if sm_scale != 1.0:
    qk_scale *= sm_scale

  # How many KV blocks in the history prefix (with your settings block_k=128 => 1..4)
  upper_bound = pl.cdiv(history_len, block_k)
  aligned = (history_len % block_k == 0)
  valid_last = history_len - (upper_bound - 1) * block_k  # 1..block_k

  # Number of Q sub-blocks inside this program
  if q_rows % block_q != 0:
    raise ValueError(f"Kernel expects q_rows multiple of block_q, got {q_rows=} {block_q=}")
  q_subblocks = q_rows // block_q

  # Per-subblock online-softmax state
  m = [jnp.full((block_q,), -jnp.inf, dtype=jnp.float32) for _ in range(q_subblocks)]
  l = [jnp.zeros((block_q,), dtype=jnp.float32) for _ in range(q_subblocks)]
  o = [jnp.zeros((block_q, head_dim_padded), dtype=jnp.float32) for _ in range(q_subblocks)]

  # Load Q sub-blocks (keep them separate to limit temporary sizes)
  q_blocks = []
  for sb in range(q_subblocks):
    q_slice = pl.dslice(sb * block_q, block_q)
    q_blk = plgpu.load(q_ref.at[q_slice, :], mask=head_mask, other=0.0)
    q_blocks.append(q_blk)

  def load_kv(b: int):
    sl = pl.dslice(b * block_k, block_k)
    k_blk = plgpu.load(k_ref.at[sl, :], mask=head_mask, other=0.0)
    v_blk = plgpu.load(v_ref.at[sl, :], mask=head_mask, other=0.0)
    return k_blk, v_blk

  def update(q_blk, o_prev, m_prev, l_prev, k_blk, v_blk, is_last: bool):
    # QK^T
    qk = pl.dot(q_blk, k_blk.T).astype(jnp.float32) * qk_scale  # [block_q, block_k]

    # Only the final block may be partial (avoid per-iter mask)
    if (not aligned) and is_last:
      mask_k = jnp.arange(block_k) < valid_last
      qk = jnp.where(mask_k[None, :], qk, DEFAULT_MASK_VALUE)

    # Online softmax (FlashAttention)
    m_curr = jnp.max(qk, axis=-1)                # [block_q]
    m_next = jnp.maximum(m_prev, m_curr)         # [block_q]

    correction = jnp.exp2(m_prev - m_next)       # [block_q]
    l_prev_corr = l_prev * correction            # [block_q]

    s = jnp.exp2(qk - m_next[:, None])           # [block_q, block_k]
    l_curr = jnp.sum(s, axis=-1)                 # [block_q]
    l_next = l_prev_corr + l_curr                # [block_q]

    o_prev_corr = o_prev * correction[:, None]   # [block_q, d]
    # Cast probs to V dtype for tensorcore-friendly PV
    o_curr = pl.dot(s.astype(v_blk.dtype), v_blk).astype(jnp.float32)

    o_next = o_prev_corr + o_curr
    return o_next, m_next, l_next

  # Step over one KV block: load once, update all Q sub-blocks (reuse!)
  def kv_step(b: int, is_last: bool):
    k_blk, v_blk = load_kv(b)
    for sb in range(q_subblocks):
      o[sb], m[sb], l[sb] = update(q_blocks[sb], o[sb], m[sb], l[sb], k_blk, v_blk, is_last=is_last)

  # Unroll <=4 KV blocks (history_len is static in your jit)
  if upper_bound == 1:
    kv_step(0, True)
  elif upper_bound == 2:
    kv_step(0, False)
    kv_step(1, True)
  elif upper_bound == 3:
    kv_step(0, False)
    kv_step(1, False)
    kv_step(2, True)
  elif upper_bound == 4:
    kv_step(0, False)
    kv_step(1, False)
    kv_step(2, False)
    kv_step(3, True)
  elif upper_bound == 5:
    kv_step(0, False)
    kv_step(1, False)
    kv_step(2, False)
    kv_step(3, False)
    kv_step(4, True)
  elif upper_bound == 6:
    kv_step(0, False)
    kv_step(1, False)
    kv_step(2, False)
    kv_step(3, False)
    kv_step(4, False)
    kv_step(5, True)
  elif upper_bound == 7:
    kv_step(0, False)
    kv_step(1, False)
    kv_step(2, False)
    kv_step(3, False)
    kv_step(4, False)
    kv_step(5, False)
    kv_step(6, True)
  else:
    kv_step(0, False)
    kv_step(1, False)
    kv_step(2, False)
    kv_step(3, False)
    kv_step(4, False)
    kv_step(5, False)
    kv_step(6, False)
    kv_step(7, True)

  # Normalize + write out per sub-block
  for sb in range(q_subblocks):
    o_norm = (o[sb] / l[sb][:, None]).astype(o_ref.dtype)
    out_slice = pl.dslice(sb * block_q, block_q)
    plgpu.store(o_ref.at[out_slice, :], o_norm, mask=head_mask)

  # Optional residual LSE for backward: shape matches q_rows
  if residual_refs:
    lse_ref = residual_refs[0]
    # Concatenate in order (block_q chunks)
    lse = jnp.concatenate([m[sb] + jnp.log2(l[sb]) for sb in range(q_subblocks)], axis=0)
    lse_ref[...] = lse



@functools.partial(
    jax.jit,
    static_argnames=[
        "history_len",
        "candidate_len",
        "sm_scale",
        "block_sizes",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "return_residuals",
    ],
)
def custom_masked_mha_v2(
    q,
    k,
    v,
    history_len: int,
    candidate_len: int,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False,
):
  """
  Custom masked multi-head attention for sequential recommendation.

  This function implements a custom attention pattern where history tokens
  attend bidirectionally to each other, while candidate tokens attend only
  to history tokens. This is useful for ranking/recommendation tasks.

  Args:
    q: Query tensor [batch_size, seq_len, num_heads, head_dim]
    k: Key tensor [batch_size, seq_len, num_heads, head_dim]
    v: Value tensor [batch_size, seq_len, num_heads, head_dim]
    history_len: Number of history tokens (attend bidirectionally)
    candidate_len: Number of candidate tokens (attend to history only)
    sm_scale: Softmax scale (default 1.0). For correct attention, use 1/sqrt(head_dim).
              If 1.0 is passed, auto-computed as 1/sqrt(head_dim).
    block_sizes: Block sizes for tiling (default: 128x128)
    num_warps: Number of warps for GPU kernel (default: auto-tuned)
    num_stages: Number of pipeline stages (default: 2)
    grid: Grid dimensions (default: auto-computed)
    interpret: Whether to use interpreter mode (default: False)
    debug: Whether to enable debug mode (default: False)
    return_residuals: Whether to return LSE for backward pass (default: False)

  Returns:
    Output tensor [batch_size, seq_len, num_heads, head_dim]
    If return_residuals=True, also returns LSE [batch_size, num_heads, seq_len]

  Example:
    >>> q = jax.random.normal(key, (2, 1024, 8, 64))
    >>> k = jax.random.normal(key, (2, 1024, 8, 64))
    >>> v = jax.random.normal(key, (2, 1024, 8, 64))
    >>> out = custom_masked_mha(q, k, v, history_len=512, candidate_len=512)
    >>> out.shape
    (2, 1024, 8, 64)
  """
  if block_sizes is None:
    block_sizes = BlockSizes.get_default()

  batch_size, q_seq_len, num_heads, head_dim = q.shape
  kv_seq_len = k.shape[1]

  # Auto-compute sm_scale if default value is used
  # Use pure Python math to avoid JAX array in static args
  if sm_scale == 1.0:
    sm_scale = 1.0 / math.sqrt(float(head_dim))

  # Validate inputs
  if (q.shape[-1] != k.shape[-1]) or (q.shape[-1] != v.shape[-1]):
    raise ValueError(
        f"This kernel expects q, k, and v to have the same head dimension, but"
        f" found {q.shape=}, {k.shape=}, {v.shape=}."
    )

  if q_seq_len != history_len + candidate_len:
    raise ValueError(
        f"q_seq_len ({q_seq_len}) must equal history_len + candidate_len "
        f"({history_len} + {candidate_len} = {history_len + candidate_len})"
    )

  block_q = min(block_sizes.block_q, q_seq_len)
  block_k = min(block_sizes.block_k, kv_seq_len)
  head_dim_padded = pl.next_power_of_2(head_dim)

  if q_seq_len % block_q != 0:
    raise ValueError(f"{q_seq_len=} must be a multiple of {block_q=}")
  if kv_seq_len % block_k != 0:
    raise ValueError(f"{kv_seq_len=} must be a multiple of {block_k=}")
  
  q_blocks_per_program = 2
  q_tile = block_q * q_blocks_per_program

  # Grid computation: one program per Q block, per batch, per head
  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(q_seq_len, q_tile), batch_size, num_heads)

  # Heuristic for number of warps
  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 8 if head_dim <= 64 else 8

  kernel = functools.partial(
      custom_masked_attention_forward_kernel,
      history_len=history_len,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      head_dim=head_dim,
  )

  # BlockSpec defines how data is tiled and distributed across the grid
  in_specs = [
      # Q: each program processes one Q block
      pl.BlockSpec((None, q_tile, None, head_dim_padded),
                   lambda i, j, k: (j, i, k, 0)),
      # K: full sequence available to all programs
      pl.BlockSpec((None, kv_seq_len, None, head_dim_padded),
                   lambda _, j, k: (j, 0, k, 0)),
      # V: full sequence available to all programs
      pl.BlockSpec((None, kv_seq_len, None, head_dim_padded),
                   lambda _, j, k: (j, 0, k, 0)),
  ]

  out_shape = [q]
  out_specs = [
      pl.BlockSpec((None, q_tile, None, head_dim_padded),
                   lambda i, j, k: (j, i, k, 0))
  ]

  # Add LSE output if requested
  if return_residuals:
    out_shape.append(
        jax.ShapeDtypeStruct(
            shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32
        )
    )
    out_specs.append(
        pl.BlockSpec((None, None, q_tile), lambda i, j, k: (j, k, i))
    )

  out = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=in_specs,
      out_specs=out_specs,
      compiler_params=plgpu.CompilerParams(
          num_warps=num_warps_, num_stages=num_stages
      ),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="custom_masked_mha_forward",
  )(q, k, v)

  return out if return_residuals else out[0]
