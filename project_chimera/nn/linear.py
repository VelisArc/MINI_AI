# project_chimera/nn/linear.py
import numpy as np
from .module import Module
from ..l1_calculus.tensor import Tensor

class Linear(Module):
  """
  A fully connected linear layer.
  """
  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    scale = np.sqrt(2. / in_features)
    self.weight = Tensor(
      np.random.randn(in_features, out_features) * scale,
      requires_grad=True
    )

    self.bias = None
    if bias:
      self.bias = Tensor(
        np.zeros((1, out_features)),
        requires_grad=True
      )

  def forward(self, input_tensor: Tensor) -> Tensor:
    # --- THE FIX IS HERE ---
    # .dot() को .matmul() से बदला गया है, जो अधिक सामान्य और सही है
    output = input_tensor.matmul(self.weight)

    if self.bias is not None:
      output = output + self.bias
    return output

  def __repr__(self):
    return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
