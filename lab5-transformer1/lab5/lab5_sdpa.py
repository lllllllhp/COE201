# lab5_sdpa.py
# Lab 5 Task 1: Single-Head Scaled Dot-Product Attention

import torch
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Manually implement single-head scaled dot-product attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, dim)
        K: Key tensor, shape (batch_size, seq_len, dim)
        V: Value tensor, shape (batch_size, seq_len, dim)
        mask: Optional mask tensor, shape (batch_size, seq_len, seq_len) or (seq_len, seq_len)
              0 means masked out, 1 means keep

    Returns:
        output: Attention output, shape (batch_size, seq_len, dim)
        attn_weights: Attention weights, shape (batch_size, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    ### TODO: 1. Compute dot product attention scores (Q @ K^T)
    
    ### TODO: 2. Scale the scores by dividing by sqrt(d_k)
    
    ### TODO: 3. Apply the mask (if provided) by setting masked positions to -inf
    # Hint: use scores.masked_fill(mask == 0, float('-inf')) if mask is not None
    
    ### TODO: 4. Apply softmax to get the attention weights
    
    ### TODO: 5. Multiply weights with V to get the final output
    
    # --- Your code starts here ---
    pass
    # --- Your code ends here ---
    
    # return output, attn_weights

def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive generation.
    The mask should be a lower triangular matrix where elements (i, j) with j <= i are 1, and 0 otherwise.
    """
    ### TODO: Create the causal mask (lower triangular matrix)
    # Hint: use torch.tril and torch.ones
    
    # --- Your code starts here ---
    pass
    # --- Your code ends here ---
    # return mask

def main():
    print("Testing Task 1: Single-Head Scaled Dot-Product Attention")
    batch_size, seq_len, dim = 2, 4, 8
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)

    # 1. Basic test
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    print(f"Basic output shape: {output.shape}")
    
    # 2. Mask test
    mask = create_causal_mask(seq_len)
    output_masked, attn_weights_masked = scaled_dot_product_attention(Q, K, V, mask)
    print(f"Masked weights (first row should only have first element non-zero):\n{attn_weights_masked[0, 0]}")

if __name__ == "__main__":
    main()
