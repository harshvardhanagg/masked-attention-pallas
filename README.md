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

## Key Optimization

Since we only need to attend to tokens in the `history_len`, this allows the kernel to process K/V blocks efficiently:
- Total sequence length: `seq_len = history_len + candidate_len`
- K/V iteration: Only processes `history_len` blocks (not full `seq_len`)

For example, with `block_k=128`:
- If `history_len=512`, we iterate over `512/128 = 4` K/V blocks
- If `seq_len=1024`, a naive approach would iterate over `1024/128 = 8` blocks
- **Savings: 50% fewer K/V block loads**

Here's the plot that shows the comparison between the custom attention and bidirectional attention. As expected the execution time increases as history length is increased.
<img width="1268" height="450" alt="Screenshot 2026-01-17 at 4 53 27 PM" src="https://github.com/user-attachments/assets/e29127f4-dabc-46b8-b692-6de2e1234bde" />


## AI Usage
1. AI is used to edit this markdown.
2. AI is used to quickly understand the reference code.  
