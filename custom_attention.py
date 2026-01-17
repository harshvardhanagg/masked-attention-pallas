# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    q_ref,
    k_ref,
    v_ref,  # Input arrays
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    history_len: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    head_dim: int,
):
  """
  Custom masked attention kernel for recommendation models.

  This kernel implements a structured mask where:
  - History tokens (positions < history_len) attend bidirectionally to all history
  - Candidate tokens (positions >= history_len) attend only to history tokens

  The implementation follows the FlashAttention algorithm with online softmax
  computation, adapted from the JAX Pallas reference implementation.

  Args:
    q_ref: Query tensor reference [seq_len, head_dim_padded]
    k_ref: Key tensor reference [seq_len, head_dim_padded]
    v_ref: Value tensor reference [seq_len, head_dim_padded]
    o_ref: Output tensor reference [seq_len, head_dim_padded]
    residual_refs: Optional residual outputs (e.g., LSE for backward pass)
    history_len: Number of history tokens that attend bidirectionally
    sm_scale: Softmax scale factor (typically 1/sqrt(head_dim))
    block_q: Block size for Q dimension
    block_k: Block size for K/V dimension
    head_dim: Actual head dimension (before padding)
  """
  seq_len = k_ref.shape[0]
  start_q = pl.program_id(0)
  head_dim_padded = q_ref.shape[-1]

  # Online softmax accumulators (FlashAttention algorithm)
  # m_i tracks the running maximum, l_i tracks the running sum
  m_i = jnp.zeros(block_q, dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros(block_q, dtype=jnp.float32)
  # Accumulator for output
  o = jnp.zeros((block_q, head_dim_padded), dtype=jnp.float32)

  # Load Q block: it will stay in L1/SRAM throughout
  curr_q_slice = pl.dslice(start_q * block_q, block_q)
  head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
  q = plgpu.load(q_ref, mask=head_mask, other=0.0)

  # Determine K/V iteration bounds based on custom mask structure
  # Key optimization: Both history and candidate queries only need to attend
  # to history K/V blocks. We skip all K/V blocks beyond history_len.
  upper_bound = pl.cdiv(history_len, block_k)

  def body(start_k, carry):
    """Process one K/V block and update accumulators."""
    o_prev, m_prev, l_prev = carry
    curr_k_slice = pl.dslice(start_k * block_k, block_k)

    # Load K and V blocks
    k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
    qk = pl.dot(q, k.T)  # [block_q, block_k]

    # Scale logits to convert from base-2 to natural log domain
    # This is based on the identity: e^x = 2^(x * log2(e))
    # Using base-2 is more hardware-friendly on GPUs
    qk_scale = math.log2(math.e)
    if sm_scale != 1.:
      qk_scale *= sm_scale
    qk *= qk_scale

    # Apply custom mask for this Q-K block pair
    span_q = start_q * block_q + jnp.arange(block_q)
    span_k = start_k * block_k + jnp.arange(block_k)

    # Custom mask logic:
    # - History queries (span_q < history_len) attend to history keys (span_k < history_len)
    # - Candidate queries (span_q >= history_len) attend to history keys (span_k < history_len)
    # The condition simplifies to: mask is True when span_k < history_len
    # (since we already limited upper_bound to history K blocks)
    mask = (span_q[:, None] < history_len) & (span_k[None, :] < history_len) | \
           (span_q[:, None] >= history_len) & (span_k[None, :] < history_len)

    # Apply mask to scores
    qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    # Online softmax update (FlashAttention algorithm)
    m_curr = jnp.max(qk, axis=-1)  # Current block max
    m_next = jnp.maximum(m_prev, m_curr)  # Global running max
    correction = jnp.exp2(m_prev - m_next)  # Correction factor for previous values
    l_prev_corr = correction * l_prev
    s_curr = jnp.exp2(qk - m_next[:, None])  # Softmax of current block
    l_curr = s_curr.sum(axis=-1)
    l_next = l_prev_corr + l_curr
    o_prev_corr = correction[:, None] * o_prev
    v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask)
    o_curr = pl.dot(s_curr.astype(v.dtype), v)

    o_next = o_prev_corr + o_curr
    return o_next, m_next, l_next

  # Process all history K/V blocks
  o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

  # Final normalization (divide by sum of weights)
  o /= l_i[:, None]

  # Store LSE (log-sum-exp) for potential backward pass
  if residual_refs:
    lse_ref = residual_refs[0]
    lse_ref[...] = m_i + jnp.log2(l_i)

  # Write output to HBM
  plgpu.store(o_ref.at[:, : o.shape[-1]], o.astype(o_ref.dtype), mask=head_mask)


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
def custom_masked_mha(
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

  # Grid computation: one program per Q block, per batch, per head
  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

  # Heuristic for number of warps
  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8

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
      pl.BlockSpec((None, block_q, None, head_dim_padded),
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
      pl.BlockSpec((None, block_q, None, head_dim_padded),
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
        pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i))
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
