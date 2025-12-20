# project_chimera/nn/activations.py

from project_chimera.nn.module import Module
from project_chimera.l1_calculus.tensor import Tensor

class ReLU(Module):
    """
    Applies the Rectified Linear Unit function element-wise.
    """
    def forward(self, input_tensor: Tensor) -> Tensor:
        # We can directly use the .relu() method we built on our Tensor
        return input_tensor.relu()

    def __repr__(self):
        return "ReLU()"
