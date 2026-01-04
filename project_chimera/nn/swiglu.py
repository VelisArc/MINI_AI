# project_chimera/nn/swiglu.py
from .module import Module
from .linear import Linear
from ..l1_calculus.tensor import Tensor
import numpy as np

class SiLU(Module):
    """Sigmoid Linear Unit (SiLU) or Swish activation function: x * sigmoid(x)"""
    def forward(self, x: Tensor) -> Tensor:
        # We don't have a dedicated SiLU op yet, so we compose it.
        # sigmoid(x) = 1 / (1 + exp(-x))
        # silu(x) = x / (1 + exp(-x))
        return x * (x.detach()._ensure_tensor(1.0) / (x.detach()._ensure_tensor(1.0) + (-x).exp()))

class SwiGLU(Module):
    """
    SwiGLU Activation Unit.
    Replaces the standard FeedForward block (Linear -> ReLU -> Linear).
    Formula: SwiGLU(x) = (SiLU(xW) * xV)W2
    """
    def __init__(self, in_features, hidden_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features

        self.w1 = Linear(in_features, hidden_features) # Gate
        self.w2 = Linear(in_features, hidden_features) # Value
        self.w3 = Linear(hidden_features, out_features) # Output

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)

        # SiLU implementation inline for efficiency: x * sigmoid(x)
        # Using native ops: x / (1 + exp(-x))
        silu_x1 = x1 * (Tensor(1.0) / (Tensor(1.0) + (-x1).exp()))

        hidden = silu_x1 * x2
        return self.w3(hidden)
