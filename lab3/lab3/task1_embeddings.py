# task1_embeddings.py
import torch
import numpy as np
from gensim.models import KeyedVectors

# ==========================================
# Task 1: Real Word Embeddings & Manual Similarity
# ==========================================

def cosine_similarity_manual(v1, v2):
    """
    TODO: Implement Cosine Similarity manually using torch or numpy.
    Formula: (A dot B) / (||A|| * ||B||)
    """
    # Hint: Use torch.dot() and torch.norm() or numpy equivalents
    return None

def task1_embeddings():
    print("--- Task 1: Embeddings ---")
    
    # Load local pre-trained GloVe model
    print("Loading local GloVe model (50d)...")
    model = KeyedVectors.load("glove-50d.model")
    
    words = ["king", "queen", "man", "woman", "apple", "banana"]
    vectors = {word: torch.tensor(model[word]) for word in words}
    
    # 2. Test your manual implementation
    v_king = vectors["king"]
    v_queen = vectors["queen"]
    v_apple = vectors["apple"]
    
    sim_king_queen = cosine_similarity_manual(v_king, v_queen)
    sim_king_apple = cosine_similarity_manual(v_king, v_apple)
    
    print(f"Similarity (king vs queen): {sim_king_queen}")
    print(f"Similarity (king vs apple): {sim_king_apple}")
    
    # 3. Bonus: Word Analogies
    # Test classic patterns using your manual similarity function!
    # Pattern: A - B + C ≈ D
    print("\n--- Bonus: Analogies ---")
    
    # TODO: Analogy 1 (Gender): Queen - Woman + Man ≈ King
    
    # TODO: Analogy 2 (Capital-Country): Paris - France + China ≈ Beijing
    
    # TODO: Find your own! (e.g., Verb Tense, Comparative, etc.)

if __name__ == "__main__":
    task1_embeddings()
