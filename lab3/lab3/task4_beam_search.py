# task4_beam_search.py
import numpy as np

# ==========================================
# Task 4 (Optional): Manual Beam Search Walkthrough
# ==========================================
def task4_beam_search():
    print("\n--- Task 4: (Optional) Beam Search Implementation ---")
    
    # Vocabulary: {0: "<PAD>", 1: "The", 2: "A", 3: "cat", 4: "dog"}
    idx_to_word = {0: "<PAD>", 1: "The", 2: "A", 3: "cat", 4: "dog"}
    
    # Step 1 log-probabilities (simulated from model)
    # log(0.6) = -0.5108, log(0.4) = -0.9163
    step1_log_probs = {1: np.log(0.6), 2: np.log(0.4)}
    
    # Step 2 log-probabilities given the previous word
    # If "The" (idx 1): {"cat": 0.1, "dog": 0.9}
    # If "A" (idx 2):   {"cat": 0.9, "dog": 0.1}
    step2_log_probs = {
        1: {3: np.log(0.1), 4: np.log(0.9)},
        2: {3: np.log(0.9), 4: np.log(0.1)}
    }
    
    beam_width = 2
    
    # TODO 1: Initialize candidates for Step 1
    # Each candidate should be a tuple: (cumulative_log_prob, sequence_list)
    candidates = [] # [(log_p, [id1]), (log_p, [id2])]
    
    # TODO 2: Step 2 Expansion
    # 1. For each candidate, expand it by trying all possible next words in step2_log_probs
    # 2. Store all resulting expansions (score + next_log_prob, sequence + [next_word])
    all_expansions = []
    
    # TODO 3: Beam Selection
    # 1. Sort all_expansions by their cumulative score (descending)
    # 2. Pick the top 'beam_width' candidates as the final result
    top_beams = []
    
    print("Top 2 beams:")
    for score, seq in top_beams:
        sentence = " ".join([idx_to_word[i] for i in seq])
        print(f"Sequence: {sentence}, Prob: {np.exp(score):.4f}")

if __name__ == "__main__":
    task4_beam_search()
