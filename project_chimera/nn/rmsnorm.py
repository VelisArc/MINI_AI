# project_chimera/nn/rmsnorm.py
import numpy as np
from .module import Module
from ..l1_calculus.tensor import Tensor
from ..l1_calculus.ops import RMSNormOp

class RMSNorm(Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm).
    Slightly faster and often better than LayerNorm.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # Learnable parameter: gamma (weight) only. No beta (bias).
        self.gamma = Tensor(np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return RMSNormOp.apply(x, self.gamma, eps=self.eps)

    def __repr__(self):
        return f"RMSNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"
