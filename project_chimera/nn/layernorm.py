# project_chimera/nn/layernorm.py
import numpy as np
from .module import Module
from ..l1_calculus.tensor import Tensor
from ..l1_calculus.ops import LayerNormOp

class LayerNorm(Module):
    """
    Implements Layer Normalization.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        # Learnable parameters: gamma (weight) and beta (bias)
        self.gamma = Tensor(np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(self.normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # LayerNormOp को कॉल करें, जो बैकप्रॉप को हैंडल करेगा
        return LayerNormOp.apply(x, self.gamma, self.beta, eps=self.eps)

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"
