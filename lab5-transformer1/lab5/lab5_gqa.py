# lab5_gqa.py
# Lab 5 Task 3: Grouped Query Attention

import torch
import torch.nn as nn
from lab5_sdpa import scaled_dot_product_attention

class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA).
    G Query heads share 1 Key/Value head.
    """
    def __init__(self, d_model, num_queries, num_groups):
        super().__init__()
        assert num_queries % num_groups == 0
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_groups = num_groups # This is the number of Key/Value heads
        self.heads_per_group = num_queries // num_groups
        self.d_k = d_model // num_queries

        ### TODO: Define linear projection layers
        # Note: The output dimension of W_K and W_V should be num_groups * self.d_k
        self.W_Q = None
        self.W_K = None
        self.W_V = None
        self.W_O = None

    def forward(self, Q_x, K_x, V_x, mask=None):
        """
        ### TODO: Implement the forward pass for GQA.
        Hints:
        1. Apply linear projections to Q_x, K_x, V_x to get Q, K, V (Note the output dimensions of K and V).
        2. Expand K and V to match the number of Q heads.
           Tip: Use torch.repeat_interleave or expand.
        3. Call scaled_dot_product_attention.
        
        Note: In self-attention, the incoming Q_x, K_x, and V_x are all exactly the same input sequence x.
        """
        batch_size = Q_x.size(0)

        # --- Your code starts here ---

        # Implement the masking (Optional)
        if mask is not None:
            if mask.dim() == 2: mask = mask.view(1, 1, mask.size(0), mask.size(1))
            elif mask.dim() == 3: mask = mask.unsqueeze(1)

        # return output, attn_weights
        pass
        # --- Your code ends here ---

def main():
    print("Testing Task 3: Grouped Query Attention (GQA)")
    batch_size, seq_len, d_model = 2, 4, 64
    num_queries = 8
    num_groups = 2 # 4 queries per group
    
    gqa = GroupedQueryAttention(d_model, num_queries, num_groups)
    x = torch.randn(batch_size, seq_len, d_model)
    output, _ = gqa(x, x, x)
    print(f"GQA Output shape: {output.shape}")

if __name__ == "__main__":
    main()
