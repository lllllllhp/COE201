# lab4_gru.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import random

# ==========================================
# Task 3 [Bonus]: Extreme GRU Challenge
# ==========================================

class CustomGRUCell(nn.Module):
    """
    Implement a single GRU step from scratch.
    Reference the gate equations in the slides.
    """
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        ### TODO: Define your parameters (wr, wz, wn, etc.)
        pass

    def forward(self, x, h_prev):
        """
        x: (batch, input_size)
        h_prev: (batch, hidden_size)
        """
        ### TODO: Implement the GRU forward logic
        # return h_t
        return h_prev

class SentimentGRU(nn.Module):
    """
    A many-to-one Sentiment Classifier using your CustomGRUCell.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentGRU, self).__init__()
        ### TODO: Initialize embedding, gru_cell, and fc layers
        pass

    def forward(self, x):
        """
        x shape: (Batch_Size, Seq_Len)
        """
        ### TODO: Implement the sequence loop and return the final classification logits
        return None


# ==========================================
# Challenge: Short vs. Long Sequence Analysis
# ==========================================

def subgroup_analysis_test():
    """
    [YOUR TASK]
    1. Load the SST-2 dataset (train.tsv).
    2. Train your Custom GRU on the mixed dataset.
    3. Evaluate its accuracy on two subgroups of the test set:
       - LONG sequences (tokens > 35)
       - SHORT sequences (tokens <= 15)
    
    Observe how the GRU handles different sequence lengths. 
    Does it maintain high accuracy even on the longer reviews?
    """
    print("🚀 Challenge: Implement the subgroup analysis test here!")
    pass

if __name__ == "__main__":
    subgroup_analysis_test()
