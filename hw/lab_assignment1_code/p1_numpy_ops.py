"""
Assignment 1 - Problem 1: NumPy Vectorized Operations (30 points)

In this problem, you will implement several core numerical operations using NumPy.
These operations are fundamental to deep learning frameworks and are used extensively
in neural network implementations.

IMPORTANT CONSTRAINTS:
- You are forbidden from using any Python 'for' or 'while' loops.
- All operations must be vectorized using NumPy's built-in functions.
- Using loops will result in a 50% penalty for that specific part, even if correct.

Example:
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5], [6]])
    >>> matrix_multiply(A, B)
    array([[17],
           [39]])
"""
import numpy as np

# ============================================================
# Part A: Basic Operations (10 pts)
# ============================================================

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Implement standard matrix multiplication A * B.

    Args:
        A: Left matrix of shape (m, n)
        B: Right matrix of shape (n, k)

    Returns:
        C: Result matrix of shape (m, k) where C = A @ B

    Hint:
        Use np.dot() or the @ operator for matrix multiplication.

    Example:
        >>> A = np.random.randn(5, 3)
        >>> B = np.random.randn(3, 4)
        >>> C = matrix_multiply(A, B)
        >>> C.shape
        (5, 4)
    """
    ### TODO: Implement matrix multiplication
    return None
    ### END TODO


def normalize_rows(M: np.ndarray) -> np.ndarray:
    """
    Normalize each row of M to have unit L2 norm.

    For each row vector v, compute: v_normalized = v / ||v||_2
    If a row has zero norm (all zeros), leave it as zeros to avoid division by zero.

    Args:
        M: Input matrix of shape (m, n)

    Returns:
        M_normalized: Matrix of shape (m, n) where each row has unit L2 norm


    Example:
        >>> M = np.array([[3, 4], [0, 0], [1, 2, 3]])  # Note: last row would cause error
        >>> normalize_rows(np.array([[3, 4], [0, 0]]))
        array([[0.6, 0.8],
               [0. , 0. ]])
    """
    ### TODO: Write your code below (2-3 lines)
    return None
    ### END TODO


def create_one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert a 1D array of class indices into a 2D one-hot matrix.

    A one-hot encoding is a binary vector where all elements are 0 except for the
    index corresponding to the class, which is 1.

    Args:
        indices: 1D array of shape (N,) containing class indices (0 to num_classes-1)
        num_classes: Total number of classes (K)

    Returns:
        one_hot: 2D array of shape (N, num_classes) where one_hot[i, indices[i]] = 1


    Example:
        >>> indices = np.array([0, 2, 1])
        >>> create_one_hot(indices, 3)
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]])
    """
    ### TODO: Write your code below (2-3 lines)
    return None
    ### END TODO


# ============================================================
# Part B: Statistical Operations (10 pts)
# ============================================================

def column_standardize(X: np.ndarray) -> np.ndarray:
    """
    Standardize features (columns) to have zero mean and unit variance.

    For each feature j, compute: z_j = (x_j - mean_j) / std_j

    This is a common preprocessing step in machine learning that ensures all
    features are on the same scale.

    Args:
        X: Input data matrix of shape (N, D) where each column is a feature

    Returns:
        X_standardized: Matrix of shape (N, D) with zero mean and unit variance per column


    Example:
        >>> X = np.array([[1, 2], [2, 4], [3, 6]])
        >>> X_std = column_standardize(X)
        >>> np.allclose(X_std.mean(axis=0), 0)
        True
        >>> np.allclose(X_std.std(axis=0), 1)
        True
    """
    ### TODO: Write your code below (3-4 lines)
    return None
    ### END TODO


def pairwise_l2_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance between every pair of rows from A and B.

    The pairwise L2 distance formula can be expanded as:
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T * b

    This formulation allows for efficient vectorized computation without loops.

    Args:
        A: Matrix of shape (m, d) containing m data points
        B: Matrix of shape (n, d) containing n data points

    Returns:
        D: Distance matrix of shape (m, n) where D[i, j] = ||A[i] - B[j]||_2

    Example:
        >>> A = np.array([[0, 0], [1, 0]])
        >>> B = np.array([[0, 0], [0, 3]])
        >>> pairwise_l2_distance(A, B)
        array([[0., 3.],
               [1., 3.16...]])
    """
    ### TODO: Implement pairwise L2 distance
    return None
    ### END TODO


# ============================================================
# Part C: Softmax & Loss (10 pts)
# ============================================================

def softmax_batch(logits: np.ndarray) -> np.ndarray:
    """
    Implement numerically stable softmax for a batch of inputs.

    The softmax function converts logits (raw scores) into probabilities:
    softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    For numerical stability, we subtract the maximum value before exponentiating:
    softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Args:
        logits: Input array of shape (N, K) containing raw scores for K classes

    Returns:
        probs: Probability array of shape (N, K) where each row sums to 1


    Example:
        >>> logits = np.array([[1, 2, 3], [3, 2, 1]])
        >>> probs = softmax_batch(logits)
        >>> probs
        array([[0.09..., 0.24..., 0.66...],
               [0.66..., 0.24..., 0.09...]])
        >>> np.allclose(probs.sum(axis=1), 1.0)
        True
    """
    ### TODO: Write your code below (3-4 lines)
    return None
    ### END TODO


def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute the mean cross-entropy loss.

    Cross-entropy loss measures the difference between predicted probabilities
    and ground truth labels:
    L = -1/N * sum_i(log(p_i))

    where p_i is the predicted probability for the true class of sample i.

    Args:
        probs: Probability matrix of shape (N, K) - output of softmax
        targets: 1D array of shape (N,) containing ground truth class indices

    Returns:
        loss: Scalar value representing the mean cross-entropy loss

    Hint:
        Use advanced indexing to extract the probability of the correct class
        for each sample, then compute -mean(log(p)).
        Add a small epsilon (1e-12) inside log to avoid log(0).

    Example:
        >>> probs = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> targets = np.array([1, 0])
        >>> cross_entropy_loss(probs, targets)
        0.1115...
    """
    ### TODO: Implement cross-entropy loss
    return 0.0
    ### END TODO
