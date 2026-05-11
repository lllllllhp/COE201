# Lab 6 Task 3: Implementing and Comparing RoPE [Bonus]

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def get_sinusoidal_pe(max_len, d_model):
    """
    Compute sinusoidal positional encoding.
    
    ### TODO: Implement sinusoidal PE calculation 
    # Steps:
    # 1. Initialize a (max_len, d_model) zero tensor
    # 2. Compute the division term: exp(arange(0, d, 2) * -log(10000)/d)
    # 3. Apply sin to even indices and cos to odd indices
    # 4. Return as a numpy array
    """
    # --- Your code starts here ---
    pass
    # --- Your code ends here ---

def get_rope_matrix(max_len, d_model, base=10000):
    """
    Construct the (max_len, d_model, d_model) block-diagonal rotation matrix.
    
    ### TODO: Implement the RoPE rotation matrix
    """
    # --- Your code starts here ---
    return None
    # --- Your code ends here ---

def apply_rope_matrix(q, rope_matrix):
    """
    Apply RoPE rotation using pure matrix multiplication.
    Formula: q_rope = q @ R
    """
    # ### TODO: Implement apply_rope_matrix
    # --- Your code starts here ---
    return q
    # --- Your code ends here ---

def main():
    print("Testing Bonus Task: Implementing and Comparing RoPE")
    # Reduced dimensions for easier analysis
    max_len = 50
    d_model = 64
    
    # 1. Get Sinusoidal PE
    sin_pe = get_sinusoidal_pe(max_len, d_model)
    
    # 2. Get RoPE rotation matrix
    rope_matrix = get_rope_matrix(max_len, d_model)
    
    if rope_matrix is None:
        print("Please implement get_rope_matrix() first.")
        return

    # 3. Test Rotation Logic
    print("Testing Rotation Logic...")
    q = torch.randn(1, max_len, d_model)
    q_rope = apply_rope_matrix(q, torch.from_numpy(rope_matrix).float())
    print(f"Applied RoPE to query, output shape: {q_rope.shape}")
    
    # 4. Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    im1 = axes[0].pcolormesh(sin_pe, cmap='RdBu')
    axes[0].set_title('Sinusoidal Positional Encoding')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Position')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot the diagonals of the rotation matrix (which corresponds to the cosine values)
    rope_diag = np.diagonal(rope_matrix, axis1=1, axis2=2)
    im2 = axes[1].pcolormesh(rope_diag, cmap='RdBu')
    axes[1].set_title('RoPE Matrix Diagonals (Cosine values)')
    axes[1].set_xlabel('Dimension Pair Index')
    axes[1].set_ylabel('Position')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    asset_dir = "assets"
    if not os.path.exists(asset_dir): os.makedirs(asset_dir)
    plt.savefig(os.path.join(asset_dir, 'pe_comparison.png'), dpi=300)
    print("Comparison plot saved to assets/pe_comparison.png")
    
    plt.show()

if __name__ == "__main__":
    main()
