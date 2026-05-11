import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Problem 1a: Root Mean Square Normalization (RMSNorm) [15 points]
    
    RMSNorm is a simplified and more efficient alternative to LayerNorm, used in modern
    LLMs like LLaMA. Instead of centering the activations (subtracting the mean) as in
    LayerNorm, it only scales them by the root mean square.
    
    Mathematical Definition:
    RMS(x) = sqrt( 1/d * sum(x_i^2) + eps )
    output = (x / RMS(x)) * weight
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape as x.
            
        ### TODO: Implement the RMSNorm forward pass.
        # Step 1: Compute the mean of the squares of x along the last dimension.
        #         (Keep dimensions so it broadcasts correctly: keepdim=True)
        # Step 2: Add self.eps for numerical stability, then take the square root. (This is RMS(x))
        # Step 3: Divide x by RMS(x).
        # Step 4: Multiply by the learnable parameter self.weight.
        """
        # --- Your code starts here ---
        pass
        # --- Your code ends here ---


class SwiGLU(nn.Module):
    """
    Problem 1b: Swish Gated Linear Unit (SwiGLU) [15 points]
    
    SwiGLU is an activation function variant used in modern Transformers (e.g., LLaMA, PaLM).
    It replaces the standard two-layer FFN and ReLU/GELU.
    
    It operates by gating a linear projection using the Swish activation function on 
    another linear projection.
    
    Mathematical Definition:
    Swish(x) = x * sigmoid(beta * x)  (usually beta=1, which is also called SiLU)
    SwiGLU(x) = ( Swish(x * W_gate) * (x * W_up) ) * W_down
    
    Here, `* W` denotes a linear layer (without bias in common implementations).
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()

        # hint: use nn.Linear with bias=False
        # --- Your code starts here ---
        self.w_gate = None
        self.w_up = None
        self.w_down = None
        # --- Your code ends here ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU FFN.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
            
        ### TODO: Implement the SwiGLU forward pass.
        # Step 1: Compute the gate projection: `self.w_gate(x)`.
        # Step 2: Apply the SiLU (Swish with beta=1) activation to the gate output.
        #         You can use `F.silu(gate)` or manually compute `gate * torch.sigmoid(gate)`.
        # Step 3: Compute the up projection: `self.w_up(x)`.
        # Step 4: Multiply (element-wise) the activated gate and the up projection.
        # Step 5: Pass the result through the down projection: `self.w_down(...)`.
        """
        # --- Your code starts here ---
        pass
        # --- Your code ends here ---
