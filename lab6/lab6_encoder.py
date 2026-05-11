# Lab 6 Task 1: Transformer Encoder Block [30 points]

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network.

    FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

    This is provided for you — use it as a sub-layer.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Block with 2 sub-layers:
    1. Self-Attention
    2. Feed-Forward Network

    Each sub-layer uses Pre-LN residual connections:
        x = x + SubLayer(LayerNorm(x))

    Input/Output shape: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, nhead, d_ff):
        """
        ### TODO: Initialize all layers

        You need:
        1. self.self_attn: nn.MultiheadAttention(d_model, nhead, batch_first=True)
        2. self.ffn: FeedForwardNetwork(d_model, d_ff)
        3. self.norm1: nn.LayerNorm(d_model)  -- for sub-layer 1
        4. self.norm2: nn.LayerNorm(d_model)  -- for sub-layer 2
        """
        super().__init__()
        # --- Your code starts here ---
        pass
        # --- Your code ends here ---

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor, shape (batch, seq_len, d_model)

        ### TODO: Implement 2 sub-layers with Pre-LN residual connections
        Sub-layer 1: Self-Attention
          - Apply norm1 to x
          - Pass normalized x as query, key, value to self_attn
          - Add residual connection: x = x + attn_out

        Sub-layer 2: FFN
          - Apply norm2 to x
          - Pass normalized x through ffn
          - Add residual connection: x = x + ffn_out
        """
        # --- Your code starts here ---
        return x
        # --- Your code ends here ---


def main():
    print("Testing Task 1: Transformer Encoder Block")
    d_model, nhead, d_ff = 512, 8, 2048
    encoder_block = TransformerEncoderBlock(d_model, nhead, d_ff)

    x = torch.randn(2, 10, d_model)
    output = encoder_block(x)
    print(f"Encoder Block output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("Task 1 test passed!")


if __name__ == "__main__":
    main()
