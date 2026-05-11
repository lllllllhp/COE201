# lab6_transformer.py
# Lab 6 Task 3: Full Encoder-Decoder Transformer [40 points]

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding from 'Attention is All You Need'.

    This is provided for you — use it in the Transformer.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# Import your Task 1 & Task 2 implementations
from lab6_encoder import TransformerEncoderBlock
from lab6_decoder import TransformerDecoderBlock


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Architecture:
        Encoder: src -> Embedding -> scale -> PosEnc -> N x EncoderBlock -> enc_output
        Decoder: tgt -> Embedding -> scale -> PosEnc -> N x DecoderBlock(enc_output) -> dec_output
        Output:  dec_output -> Linear -> logits
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        max_len,
    ):
        super().__init__()
        self.d_model = d_model

        ### TODO: Initialize all layers
        # 1. src_embedding: nn.Embedding(src_vocab_size, d_model)
        # 2. tgt_embedding: nn.Embedding(tgt_vocab_size, d_model)
        # 3. pos_encoding: PositionalEncoding(d_model, max_len)
        # 4. encoder_layers: nn.ModuleList of TransformerEncoderBlock (from Task 1)
        # 5. decoder_layers: nn.ModuleList of TransformerDecoderBlock (from Task 2)
        # 6. fc_out: nn.Linear(d_model, tgt_vocab_size)
        # --- Your code starts here ---
        pass
        # --- Your code ends here ---

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence.

        ### TODO: Implement the encoding pass
        Steps:
        1. Apply src_embedding to src
        2. Scale by sqrt(d_model)
        3. Apply pos_encoding
        4. Pass through each encoder layer
        """
        # --- Your code starts here ---
        return src
        # --- Your code ends here ---

    def decode(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        """
        Decode the target sequence using encoder output.

        ### TODO: Implement the decoding pass
        Steps:
        1. Apply tgt_embedding to tgt
        2. Scale by sqrt(d_model)
        3. Apply pos_encoding
        4. Pass through each decoder layer (pass enc_output to each!)
        """
        # --- Your code starts here ---
        return tgt
        # --- Your code ends here ---

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Full forward pass: encode -> decode -> project to vocab.

        ### TODO: Implement the full forward pass
        Steps:
        1. enc_output = self.encode(src, src_mask)
        2. dec_output = self.decode(tgt, enc_output, tgt_mask, memory_mask)
        3. return self.fc_out(dec_output)
        """
        # --- Your code starts here ---
        return tgt
        # --- Your code ends here ---


def main():
    print("Testing Task 3: Full Transformer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 512
    num_layers = 2

    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, 8, num_layers, num_layers, 2048, 100
    ).to(device)

    src = torch.randint(0, src_vocab_size, (2, 10)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (2, 8)).to(device)

    # Generate causal mask for decoder
    tgt_len = tgt.size(1)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)

    out = model(src, tgt, tgt_mask=tgt_mask)
    print(f"Full Transformer output shape: {out.shape}")
    assert out.shape == (2, 8, tgt_vocab_size), "Output shape mismatch!"
    print("Task 3 test passed!")


if __name__ == "__main__":
    main()
