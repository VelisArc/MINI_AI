# project_chimera/nn/conv_transpose2d.py
import numpy as np
from .module import Module
from ..l1_calculus.tensor import Tensor
from ..l1_calculus.ops import ConvTranspose2dOp

class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        # ... (init अपरिवर्तित)
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

        # --- THE FIX IS HERE ---
        # वेट्स का सही आकार (in_channels, out_channels, HH, WW) है
        weight_scale = np.sqrt(2. / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = Tensor(
            np.random.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]) * weight_scale,
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return ConvTranspose2dOp.apply(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
    def __repr__(self):
        return (f"ConvTranspose2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
