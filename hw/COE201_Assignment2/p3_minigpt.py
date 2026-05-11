import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Provided Sinusoidal Positional Encoding from Lab 6."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MiniGPT(nn.Module):
    """
    Problem 3: MiniGPT Training and Generation [40 points]
    
    A MiniGPT is a Decoder-only Transformer. It predicts the probability of the
    next token given the previous tokens.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # A GPT is essentially a Transformer Encoder with a causal (upper triangular) mask!
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ### TODO: Implement the forward pass. (10 pts)
        
        1. Look up embeddings for `input_ids`.
        2. Multiply embeddings by sqrt(d_model) for scaling.
        3. Apply positional encoding.
        4. Generate a causal mask using `nn.Transformer.generate_square_subsequent_mask`
           and move it to the correct device.
        5. Pass the sequence and the causal mask through `self.transformer`.
           (Hint: pass the mask to the `mask` argument, not `src_key_padding_mask`).
        6. Project the output to vocab space using `self.lm_head`.
        
        Returns:
            logits: shape (batch_size, seq_len, vocab_size)
        """
        # --- Your code starts here ---
        pass
        # --- Your code ends here ---

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ### TODO: Implement the Causal Language Modeling loss. (15 pts)
        
        For next-token prediction, the target for the token at position `t` is 
        the token at position `t+1`.
        
        1. Get logits from `self.forward(input_ids)`.
        2. Shift logits and targets so that they align.
           - Shifted logits should contain predictions for positions 1 to seq_len-1.
           - Shifted targets should contain the true tokens for positions 1 to seq_len-1.
        3. Flatten the batch and sequence dimensions and compute CrossEntropyLoss.
        
        Returns:
            Scalar loss tensor.
        """
        # --- Your code starts here ---
        pass
        # --- Your code ends here ---

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        ### TODO: Implement autoregressive Greedy Generation. (15 pts)
        
        1. Loop `max_new_tokens` times.
        2. In each iteration, pass the current `input_ids` through the model to get logits.
        3. Extract the logits for the very last token in the sequence.
        4. Find the token ID with the maximum probability (greedy search) using `torch.argmax`.
        5. Append this new token ID to the `input_ids` tensor along the sequence dimension.
        6. Return the full generated sequence.
        
        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens)
        """
        # --- Your code starts here ---
        pass
        # --- Your code ends here ---

# =============================================================================
# Provided Code Below (Do not modify, but run it to train your model)
# =============================================================================

import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def load_dataset(file_path="tinyshakespeare.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found! Please ensure it is in the same directory.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

class CharDataset(Dataset):
    def __init__(self, data: str, seq_len: int):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to integers
        self.data = torch.tensor([self.char2idx[c] for c in data], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Return seq_len + 1 tokens because the compute_loss slices input vs target
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk

def train_minigpt(model, dataloader, epochs=1, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    accuracies = []
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # ### TODO: Implement the training step (10 pts)
            # Step 1: Zero the gradients of the optimizer.
            # Step 2: Compute the loss using `model.compute_loss(batch)`.
            # (Note: we need the variable `loss` for tracking metrics later)
            # Step 3: Backpropagate the loss.
            # Step 4: Step the optimizer.
            
            # --- Your code starts here ---
            pass
            loss = torch.tensor(0.0) # Dummy loss to prevent crash before implementation
            # --- Your code ends here ---
            
            # --- Track Metrics ---
            total_loss += loss.item() * batch.size(0)
            
            # Compute accuracy manually just for metrics
            with torch.no_grad():
                logits = model(batch)
                shift_logits = logits[:, :-1, :]
                shift_labels = batch[:, 1:]
                preds = torch.argmax(shift_logits, dim=-1)
                correct = (preds == shift_labels).sum().item()
                total_correct += correct
                total_tokens += shift_labels.numel()
                
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader.dataset)
        avg_acc = total_correct / total_tokens
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        print(f"--- Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} ---")
        
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses, marker='o', color='red')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accuracies, marker='o', color='blue')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    plt.savefig("minigpt_training_metrics.png")
    print("Saved training metrics plot to 'minigpt_training_metrics.png'")
    
    return losses, accuracies

def main():
    # --- Provided Code Below (Feel free to modify code below to improve performance) ---
    # 1. Prepare Data
    text = load_dataset()
    seq_len = 64
    batch_size = 64
    
    dataset = CharDataset(text, seq_len=seq_len)
    # Using a larger subset (e.g., 50000 samples) to ensure better learning
    # while keeping lab training time reasonable.
    train_size = min(50000, len(dataset))
    subset_indices = torch.randperm(len(dataset))[:train_size]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    print(f"Vocabulary Size: {dataset.vocab_size} (Character-level)")
    
    # 2. Recommended Model Configuration
    vocab_size = dataset.vocab_size # Usually 65 for TinyShakespeare
    d_model = 128
    nhead = 4
    num_layers = 4
    max_seq_len = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    # 3. Train
    train_minigpt(model, dataloader, epochs=10, lr=2e-3, device=device)
    
    # 4. Generate some text
    model.eval()
    context = "O God, "
    input_ids = torch.tensor([dataset.char2idx[c] for c in context], dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(input_ids, max_new_tokens=100)
    generated_text = "".join([dataset.idx2char[i.item()] for i in generated_ids[0]])
    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")

if __name__ == "__main__":
    main()
