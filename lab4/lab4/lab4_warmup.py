import torch
import torch.nn as nn

# Task 1: Warm-up - Sequence Shapes
# Goal: Understand the difference between output and hn

batch_size = 5
seq_len = 3
input_size = 10
hidden_size = 20

# 1. Initialize RNN
rnn = nn.RNN(input_size, hidden_size, batch_first=True)

# 2. Create dummy input
x = torch.randn(batch_size, seq_len, input_size)

# 3. Forward pass
output, hn = rnn(x)

print(f"Input shape:  {x.shape}")    # (Batch, Seq, Feature)
print(f"Output shape: {output.shape}") # (Batch, Seq, Hidden) -> Hidden state for EVERY step
print(f"hn shape:     {hn.shape}")     # (Layers, Batch, Hidden) -> Hidden state for LAST step only

# Verification Task:
# 1. Extract the hidden state of the LAST time step from the 'output' tensor.
# 2. Check if it matches the 'hn' tensor.
# 3. Print the result.

### TODO: Your code here (1-2 lines)
# last_step_output = ...
# are_equal = torch.allclose(...)
# print(f"Are they equal? {are_equal}")
### END TODO

print("\n" + "="*40)
print("Part 2: RNN vs LSTM Shapes")
print("="*40)

# 1. Initialize LSTM
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# 2. Forward pass
# Note: LSTM returns (output, (hn, cn))
output_lstm, (hn_lstm, cn_lstm) = lstm(x)

print(f"LSTM Output shape: {output_lstm.shape}") # (Batch, Seq, Hidden)
print(f"LSTM hn shape:     {hn_lstm.shape}")     # (Layers, Batch, Hidden)
print(f"LSTM cn shape:     {cn_lstm.shape}")     # (Layers, Batch, Hidden) -> Cell state!

print("\nKey Takeaway: LSTM maintains an extra 'Cell State' (cn) for long-term memory.")
