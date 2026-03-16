# task2_tokenization.py
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

# ==========================================
# Task 2: Tokenization & The Embedding Matrix
# ==========================================

def task2_tokenization():
    print("--- Task 2: Tokenization & Embedding Matrix ---")
    
    # 1. Load Real-World Vocabulary (from GloVe)
    # NOTE: GloVe is a word-level model, so "tokenization" here is just splitting by space.
    print("Loading GloVe model vocabulary...")
    # This model was used in Task 1
    model = KeyedVectors.load("glove-50d.model")
    
    # gensim model.key_to_index is our "Real World Vocabulary"
    vocab = model.key_to_index 
    print(f"Vocabulary size: {len(vocab)}")
    
    sentence = "the quick brown fox jumps over the lazy dog"
    
    # get token_ids 
    token_ids = [vocab[word] for word in sentence.split() if word in vocab]
    print(f"Token IDs: {token_ids}\n")
    
    # Manual Embedding Lookup
    # In this task, the "embedding" is a simple lookup from a pre-defined matrix.
    # shape: (vocab_size, embedding_dim) -> (400000, 50)
    embedding_weight = torch.tensor(model.vectors)
    
    # TODO (Step 1): Manual Indexing. 
    # 1. Find the integer index of the word "fox" in the vocabulary.
    # 2. Extract its 50-dimensional vector from `embedding_weight` using basic Python/PyTorch indexing.
    manual_vec = None
    print(f"Manual lookup vector for 'fox' (first 5 dims): {manual_vec[:5] if manual_vec is not None else None}")
    
    # 2. One-hot Equivalence
    # Prove that lookup is equivalent to One-hot vector multiplication!
    
    # TODO (Step 2): One-hot Encoding.
    # Create a vector of zeros with length equal to the vocabulary size.
    # Set the value at the index of "fox" to 1.0.
    one_hot = None
    
    # TODO (Step 3): Matrix Multiplication.
    # Perform matrix multiplication: one_hot @ embedding_weight.
    # This proves that 'lookup' is just a fast way to do 'one_hot @ Weights'.
    result_matmul = None
    print(f"Matrix multiplication result (first 5 dims): {result_matmul[0, :5] if result_matmul is not None else None}")
    
    # TODO (Step 4): PyTorch Layer.
    # 1. Initialize nn.Embedding(vocab_size, 50).
    # 2. Load the GloVe `embedding_weight` into its .weight.
    embed_layer = None
    
    if embed_layer is not None and token_ids:
        # Load the weight into the layer
        with torch.no_grad():
            embed_layer.weight.copy_(embedding_weight)
            
        # TODO (Step 4.1): Use the high-level layer to get the vector for "fox" (index 3 in sentence)
        ids_tensor = torch.tensor(token_ids)
        output = None
        
        # TODO (Step 4.2): Manually extract the vector for "fox" from embed_layer.weight 
        # just to prove it's the same matrix!
        layer_weight_vec = None 

        # Verification
        # Index 3 in our sentence "the quick brown fox" is "fox"
        if manual_vec is not None and result_matmul is not None and layer_weight_vec is not None:
            assert torch.allclose(manual_vec, output[3]), "Manual lookup vs nn.Embedding mismatch!"
            assert torch.allclose(manual_vec, result_matmul.squeeze()), "Matmul logic mismatch!"
            assert torch.allclose(manual_vec, layer_weight_vec), "Weight extraction mismatch!"
            print("\nVerification Success!")
            print("1. Manual Lookup (raw weight) == One-hot Matrix Multiplication")
            print("2. One-hot Multiplication == high-level nn.Embedding(ids)")
            print("3. high-level nn.Embedding(ids) == Manual Weight Extraction")

if __name__ == "__main__":
    task2_tokenization()
