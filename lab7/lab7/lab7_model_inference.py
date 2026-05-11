# Lab 7 Task 1: Model Inference Pipeline [35 points]

import torch
import torch.nn.functional as F
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply Top-P (nucleus) sampling filter to logits.

    This helper removes unlikely tokens whose cumulative probability exceeds top_p,
    then returns the filtered logits (with removed tokens set to -inf).

    Args:
        logits: shape (batch_size, vocab_size)
        top_p: cumulative probability threshold (e.g., 0.9)

    Returns:
        Filtered logits with the same shape.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Keep tokens until cumulative prob exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter the mask back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    return logits


def generate_text(model, tokenizer, prompt, max_new_tokens=20, temperature=1.0, top_p=1.0):
    """
    Generate text manually token-by-token.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        prompt (str): Input text
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling
        top_p (float): Top-p (nucleus) sampling threshold

    Returns:
        str: The generated text (including the prompt)
    """
    device = next(model.parameters()).device

    # 1. Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    print(f"Prompt: {prompt}")
    print("Generating...", end="", flush=True)

    for _ in range(max_new_tokens):
        ### TODO: Implement the generation loop
        # 1. Forward pass: get logits from the model
        # 2. Extract logits for the LAST token
        # 3. Apply temperature scaling to the logits
        # 4. Apply top-p filtering using the provided `filter_top_p` helper
        # 5. Convert filtered logits to probabilities and sample using torch.multinomial
        # 6. Append the predicted token to input_ids
        # 7. Check if the predicted token is EOS (tokenizer.eos_token_id), if so break

        # --- Your code starts here ---
        break  # Remove this break
        # --- Your code ends here ---

    # Decode the complete sequence
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text


def main():
    print("Testing Task 1: Manual Inference Loop")
    # Using a small model for testing
    model_name = "models/Qwen2.5-0.5B"  # Local path
    print(f"Loading {model_name} (this may take a minute if downloading)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except Exception as e:
        print(f"Failed to load model. Do you have network access? Error: {e}")
        return

    model.eval()

    prompt = "The capital of France is"

    print("\n--- Greedy (Temp=0.01) ---")
    greedy = generate_text(model, tokenizer, prompt, max_new_tokens=10, temperature=0.01, top_p=1.0)
    print(f"\nResult: {greedy}")

    print("\n--- Sampling (Temp=0.8, Top-p=0.9) ---")
    sampled = generate_text(model, tokenizer, prompt, max_new_tokens=10, temperature=0.8, top_p=0.9)
    print(f"\nResult: {sampled}")

    print("\nTask 1 execution finished! (Check if the outputs make sense)")


if __name__ == "__main__":
    main()
