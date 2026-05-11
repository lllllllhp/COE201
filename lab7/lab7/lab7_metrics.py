# Lab 7 Task 3: LLM Evaluation Pipeline [30 points]

import torch
import torch.nn.functional as F
import json
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Please install transformers")
    exit(1)


def calculate_perplexity(model, tokenizer, text: str) -> float:
    """
    Calculate perplexity of text under the model.

    Perplexity = exp(average negative log-likelihood)
    Lower perplexity means the model is less "surprised" by the text.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        text: Text to evaluate

    Returns:
        Perplexity value (float)
    """
    device = next(model.parameters()).device

    ### TODO: Implement perplexity calculation (20 pts)
    # Step 1: Tokenize the text into input_ids
    # Step 2: Forward pass to get logits
    # Step 3: Compute log probabilities using F.log_softmax
    # Step 4: For each position, select the log_prob of the actual next token
    #         (Hint: use logits[:, :-1] to predict tokens[1:])
    # Step 5: Calculate average negative log-likelihood
    # Step 6: Return exp(average NLL)

    # --- Your code starts here ---
    return 0.0  # Placeholder, replace with actual implementation
    # --- Your code ends here ---


def compute_log_likelihood(model, tokenizer, prompt: str, continuation: str) -> float:
    """
    Compute log-likelihood of continuation given prompt.

    This is useful for multiple-choice evaluation: compare likelihoods
    of different answer choices given the same question prompt.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        prompt: The context/question
        continuation: The text to evaluate (e.g., answer choice)

    Returns:
        Log-likelihood value (float, negative)
    """
    device = next(model.parameters()).device

    ### TODO: Implement log-likelihood computation (10 pts)
    # Step 1: Tokenize prompt and continuation separately to find boundary
    # Step 2: Concatenate them and do forward pass
    # Step 3: Extract logits only for the continuation tokens
    # Step 4: Sum log probabilities of continuation tokens

    # --- Your code starts here ---
    return 0.0  # Placeholder, replace with actual implementation
    # --- Your code ends here ---


def evaluate_multiple_choice(model, tokenizer, questions_path: str) -> dict:
    """
    Evaluate model on multiple-choice questions.

    For each question, compute log-likelihood of each answer choice
    given the question prompt, and select the one with highest likelihood.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        questions_path: Path to JSON file with questions

    Returns:
        Dictionary with accuracy and per-question results
    """
    # Load questions
    with open(questions_path, 'r') as f:
        questions = json.load(f)

    results = []
    correct = 0

    for q in questions:
        question_text = q['question']
        choices = q['choices']  # List of answer strings
        correct_idx = q['answer']  # Index of correct answer

        ### TODO: Implement multiple-choice evaluation
        # For each choice, compute log-likelihood given the question
        # Select the choice with highest likelihood as prediction
        # Compare with correct answer

        # --- Your code starts here ---
        # Placeholder: predict choice 0 for all questions
        predicted_idx = 0
        # --- Your code ends here ---

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1

        results.append({
            'question': question_text,
            'predicted': choices[predicted_idx],
            'correct': choices[correct_idx],
            'is_correct': is_correct
        })

    accuracy = correct / len(questions)
    return {
        'accuracy': accuracy,
        'total': len(questions),
        'correct': correct,
        'results': results
    }


def main():
    print("=" * 60)
    print("Lab 7 Task 3: LLM Evaluation Pipeline")
    print("=" * 60)

    # Load model
    model_name = "models/Qwen2.5-0.5B"  # Local path
    print(f"\nLoading {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    model.eval()

    # --- Test Perplexity ---
    print("\n" + "-" * 40)
    print("Part 1: Perplexity Calculation")
    print("-" * 40)

    test_texts = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "asdfghjkl random text qwerty",  # Should have higher perplexity
    ]

    for text in test_texts:
        ppl = calculate_perplexity(model, tokenizer, text)
        print(f"Text: {text[:40]}...")
        print(f"Perplexity: {ppl:.2f}")
        print()

    # --- Test Multiple Choice Evaluation ---
    print("\n" + "-" * 40)
    print("Part 2: Multiple Choice Benchmark")
    print("-" * 40)

    benchmark_path = "mini_benchmark.json"

    try:
        eval_results = evaluate_multiple_choice(model, tokenizer, benchmark_path)
        print(f"\nAccuracy: {eval_results['accuracy']:.2%}")
        print(f"Correct: {eval_results['correct']}/{eval_results['total']}")

        # Show some results
        print("\nSample results:")
        for r in eval_results['results'][:3]:
            status = "Correct" if r['is_correct'] else "Wrong"
            print(f"  [{status}] Q: {r['question'][:50]}...")
            print(f"       Predicted: {r['predicted']}, Correct: {r['correct']}")
    except FileNotFoundError:
        print(f"Benchmark file not found: {benchmark_path}")
        print("Make sure mini_benchmark.json exists in the current directory.")

    print("\n" + "=" * 60)
    print("Task 3 execution finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()