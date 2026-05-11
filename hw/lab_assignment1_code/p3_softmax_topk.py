"""
Assignment 1 - Problem 3: Softmax, Temperature & Top-K Sampling (30 points)

In this problem, you will implement the core components of the LLM decoding
pipeline, including temperature scaling, stable softmax, and top-k filtering.

Background: Text Generation in Large Language Models
----------------------------------------------------
When an LLM generates text, it predicts a probability distribution over the
vocabulary at each step. The way we select the next token from this distribution
significantly affects the quality and diversity of the generated text.

Decoding Strategies:
1. Greedy Search: Always pick the highest probability token
   - Pro: Deterministic, often high quality
   - Con: Can be repetitive and boring

2. Random Sampling: Sample from the full probability distribution
   - Pro: More diverse and creative
   - Con: Can produce incoherent text

3. Top-K Sampling: Sample only from the K most likely tokens
   - Pro: Balances quality and diversity
   - Con: Requires tuning K

4. Temperature Scaling: Adjust the "sharpness" of the distribution
   - Low T (<1): More deterministic, focused on high-probability tokens
   - High T (>1): More random, explores low-probability tokens

The Pipeline:
-------------
Raw Logits -> Temperature Scaling -> Top-K Filtering -> Softmax -> Sample

Example (Temperature Effect):
    >>> logits = torch.tensor([1.0, 2.0, 3.0])
    >>> softmax(logits, T=1.0)  # [0.09, 0.24, 0.67]
    >>> softmax(logits, T=0.5)  # [0.02, 0.12, 0.86] - sharper
    >>> softmax(logits, T=2.0)  # [0.19, 0.29, 0.52] - flatter
"""
import torch


def manual_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Implement numerically stable softmax with temperature scaling.

    The softmax function converts raw logits into a probability distribution:
        softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    Temperature scaling modifies the distribution before applying softmax:
        softmax_with_temp(x)_i = exp(x_i / T) / sum_j(exp(x_j / T))

    For numerical stability, we subtract the maximum value:
        stable_softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Args:
        logits: Input tensor of shape (..., V) where V is the vocabulary size.
                Can be any shape (the last dimension is treated as the vocabulary).
        temperature: Scaling factor T > 0.
                    - T < 1: Sharper distribution (more confident/deterministic)
                    - T = 1: Standard softmax
                    - T > 1: Flatter distribution (more uncertain/diverse)

    Returns:
        probs: Probability tensor of same shape as logits, where:
               - All values are in [0, 1]
               - Values along the last dimension sum to 1


    Example:
        >>> logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        >>> probs = manual_softmax(logits, temperature=1.0)
        >>> probs
        tensor([[0.0900, 0.2447, 0.6652],
                [0.6652, 0.2447, 0.0900]])
        >>> probs.sum(dim=-1)
        tensor([1., 1.])

    Example (Temperature Effect):
        >>> logits = torch.tensor([1.0, 2.0, 3.0])
        >>> manual_softmax(logits, temperature=0.5)  # Sharper
        tensor([0.0159, 0.1171, 0.8670])
        >>> manual_softmax(logits, temperature=2.0)  # Flatter
        tensor([0.1879, 0.2900, 0.5221])
    """
    ### TODO: Implement stable softmax
    return torch.zeros_like(logits)
    ### END TODO


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the top k most likely tokens and set others to zero probability.

    Top-K filtering is used to:
    - Avoid sampling very unlikely tokens that could make text incoherent
    - Reduce the effective vocabulary size for more controlled generation
    - Balance between diversity (high k) and quality (low k)

    The function:
    1. Identifies the top-k values along the last dimension
    2. Masks all other positions with -infinity (so softmax gives them 0 probability)
    3. Applies softmax to get a valid probability distribution over only top-k tokens

    Args:
        logits: Input tensor of shape (..., V) where V is the vocabulary size.
        k: Number of top elements to keep (e.g., k=50 or k=100 are common values)

    Returns:
        probs: Probability distribution of same shape as logits where:
               - Only the top-k positions have non-zero probability
               - Probabilities sum to 1 along the last dimension


    Example:
        >>> logits = torch.tensor([[0.1, 0.5, 0.2, 0.8, 0.3]])
        >>> probs = top_k_filtering(logits, k=2)
        >>> probs
        tensor([[0.0000, 0.3775, 0.0000, 0.6225, 0.0000]])
        >>> probs.sum(dim=-1)
        tensor([1.])

    Example (Batch Processing):
        >>> logits = torch.randn(2, 100)  # Batch of 2, vocab size 100
        >>> probs = top_k_filtering(logits, k=10)
        >>> probs.shape
        torch.Size([2, 100])
        >>> (probs > 0).sum(dim=-1)  # Exactly 10 non-zero per row
        tensor([10, 10])
    """
    ### TODO: Implement top-k filtering
    return torch.zeros_like(logits)
    ### END TODO


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    k: int = 5,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Sample from logits using temperature scaling and top-k filtering.

    This function combines all three components into a complete text generation
    primitive:

    1. Temperature Scaling: Adjust the distribution sharpness
    2. Top-K Filtering: Keep only the k most likely tokens
    3. Multinomial Sampling: Randomly select tokens based on their probabilities

    Args:
        logits: Input tensor of shape (batch_size, V) where V is vocabulary size.
        temperature: Temperature for scaling (T > 0).
                    - T < 1: More deterministic, conservative generation
                    - T > 1: More diverse, creative generation
        k: Number of top tokens to keep for filtering.
        num_samples: Number of tokens to sample for each batch element.

    Returns:
        samples: Tensor of shape (batch_size, num_samples) containing sampled
                 token indices (integers in range [0, V))

    Hint:
        1. Scale logits by temperature
        2. Apply top_k_filtering (which includes softmax)
        3. Use torch.multinomial() with replacement=True (required if num_samples > k)


    Example:
        >>> logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        >>> samples = sample_with_temperature(logits, temperature=1.0, k=2, num_samples=5)
        >>> samples.shape
        torch.Size([1, 5])
        >>> samples  # Will mostly contain indices 2 and 3 (top 2)
        tensor([[3, 3, 2, 3, 3]])

    Example (Low Temperature - Almost Deterministic):
        >>> logits = torch.randn(1, 10)
        >>> samples = sample_with_temperature(logits, temperature=0.01, num_samples=100)
        >>> # Almost all samples should be the argmax
        >>> (samples == logits.argmax()).all()
        True
    """
    ### TODO: Implement sampling
    return torch.zeros(logits.size(0), num_samples).long()
    ### END TODO
