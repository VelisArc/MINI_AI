# project_chimera/nn/conv2d.py
import numpy as np
from .module import Module
from ..l1_calculus.tensor import Tensor
from ..l1_calculus.ops import Conv2dOp

class Conv2d(Module):
    """
    Implements a 2D Convolutional Layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # वेट्स और बायस को इनिशियलाइज़ करें
        # He/Kaiming इनिशियलाइज़ेशन एक अच्छी प्रैक्टिस है
        weight_scale = np.sqrt(2. / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * weight_scale,
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_channels),
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return Conv2dOp.apply(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
