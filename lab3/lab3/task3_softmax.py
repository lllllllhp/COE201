# task3_softmax.py
import torch
import torch.nn.functional as F

# ==========================================
# Task 3: Output Layers (Logits, Softmax, Sampling)
# ==========================================

def manual_softmax(logits, temperature=1.0):
    """
    TODO: Implement Softmax manually with numerical stability.
    Formula: exp(x_i / T) / sum(exp(x_j / T))
    
    Stability Trick: Subtract the maximum value before exponentiating.
    x = (logits / temperature)
    x = x - max(x)
    probs = exp(x) / sum(exp(x))
    """
    # Step 1: Scale logits by temperature
    # Step 2: Subtract max for stability
    # Step 3: Exponentiate and normalize
    return None

def top_k_filtering(probs, k=3):
    """
    TODO: Keep only the top k probabilities and zero-out the rest. 
    Then re-normalize so they sum to 1.
    """
    # Hint: use torch.topk()
    return None

def task3_softmax():
    print("--- Task 3: Logits, Softmax & Decoding ---")
    
    # 1. Input Logits (e.g., from a Language Model)
    logits = torch.tensor([[-2.0, 1.0, 5.0, -1.0, 2.0, 4.5, 0.5]])
    
    # 2. Test Manual Softmax
    probs = manual_softmax(logits, temperature=1.0)
    if probs is not None:
        print(f"Manual Softmax (T=1.0):\n{probs}")
        # Verify with PyTorch
        ref = F.softmax(logits, dim=-1)
        assert torch.allclose(probs, ref), "Softmax implementation mismatch!"
        print("Success: Manual Softmax matches PyTorch!")

    # 3. Effect of Temperature
    # TODO: Calculate and compare probs for T=0.1 (sharp) and T=5.0 (flat)
    # Observe which index becomes dominant as T -> 0.
    
    # 4. Top-K Filtering
    if probs is not None:
        print("\n--- Top-K Filtering (k=3) ---")
        filtered_probs = top_k_filtering(probs, k=3)
        if filtered_probs is not None:
            print(f"Top-K Probs:\n{filtered_probs}")
            print(f"Sum of Top-K Probs: {filtered_probs.sum().item():.2f}")

    # 5. Sampling
    # TODO: Implement a single-step sampling using torch.multinomial
    pass

if __name__ == "__main__":
    task3_softmax()
