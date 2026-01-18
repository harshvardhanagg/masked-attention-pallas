# Custom Masked Attention for Sequential Recommendation

This implementation provides a custom masked multi-head attention kernel using JAX Pallas, optimized for sequential recommendation tasks.

## Mask Structure

- **History tokens** (first `history_len` positions): Attend **bidirectionally** to all history tokens
- **Candidate tokens** (last `candidate_len` positions): Attend **only to history** tokens (not to other candidates)

This pattern is useful for ranking/recommendation where candidates should be scored independently conditioned on the history.

## Files

- **`custom_attention.py`**: Custom masked attention implementation using Pallas
- **`attention.py`**: Reference JAX Pallas bidirectional attention (from JAX library)
- **`masked_attention_benchmark.ipynb`**: Correctness validation and performance benchmarking

## Performance Characteristics

Since we only need to attend to tokens in the `history_len`, this allows the kernel to process K/V blocks efficiently:
- Total sequence length: `seq_len = history_len + candidate_len`
- K/V iteration: Only processes `history_len` blocks (not full `seq_len`)

For example, with `block_k=128`:
- If `history_len=512`, we iterate over `512/128 = 4` K/V blocks
- If `seq_len=1024`, a naive approach would iterate over `1024/128 = 8` blocks
- **Savings: 50% fewer K/V block loads**


```python
# Custom masked attention kernel
upper_bound = pl.cdiv(history_len, block_k)  # Only history blocks!

# vs. Bidirectional attention
upper_bound = pl.cdiv(seq_len, block_k)  # All blocks
``
