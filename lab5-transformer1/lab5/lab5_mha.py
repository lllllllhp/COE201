# lab5_mha.py
# Lab 5 Task 2: Multi-Head Attention

import torch
import torch.nn as nn
from lab5_sdpa import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        ### TODO: Define linear projection layers
        # --- Your code starts here ---
        self.W_Q = None
        self.W_K = None
        self.W_V = None
        self.W_O = None
        # --- Your code ends here ---

    def forward(self, Q_x, K_x, V_x, mask=None):
        batch_size = Q_x.size(0)

        # --- Your code starts here ---
        ### TODO: 1. Apply linear projections to Q_x, K_x, V_x to get Q, K, V
        # Hint: In self-attention, the incoming Q_x, K_x, and V_x are all exactly the same input sequence x.
        
        ### TODO: 2. Reshape for multi-head attention: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        # Hint: Use .view() and .transpose()
        
        ### TODO: 3. Compute scaled dot-product attention
        # Implement the masking (Optional)
        if mask is not None:
            if mask.dim() == 2: mask = mask.view(1, 1, mask.size(0), mask.size(1))
            elif mask.dim() == 3: mask = mask.unsqueeze(1)

        ### TODO: Call scaled_dot_product_attention to get attn_output and attn_weights
        
        ### TODO: 4. Concatenate the heads back together
        # Hint: Use .transpose(), .contiguous(), and .view()
        
        ### TODO: 5. Apply the final linear projection (W_O)
        
        # return output, attn_weights
        pass
        # --- Your code ends here ---
        
        # return output, attn_weights

def main():
    print("Testing Task 2: Multi-Head Attention")
    batch_size, seq_len, d_model = 2, 4, 64
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = mha(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {attn_weights.shape}")

if __name__ == "__main__":
    main()
