# project_chimera/l1_calculus/tensor.py
import numpy as np
from ..l0_hal.hardware_abstraction import HAL
from .ops import Add, Mul, MatMul, Tanh, Sum, Slice, Exp, Log, Softmax, Reshape, ReLU, Transpose

class Tensor:
 def __init__(self, data, requires_grad=False):
  # --- THE FIX IS HERE ---
  # डेटा को हमेशा numpy array में बदलें अगर वह नहीं है
  if not isinstance(data, (np.ndarray, HAL.ARRAY_LIB.ndarray)):
    data = np.array(data, dtype=np.float32)
  # --- END OF FIX ---

  self.data = HAL.to_device(data.astype(np.float32) if hasattr(data, 'astype') else data)
  self.grad = None
  self.requires_grad = requires_grad
  self._backward = lambda: None
  self._prev = set()

 @property
 def shape(self): return self.data.shape
 @property
 def ndim(self): return self.data.ndim
 @property
 def dtype(self): return self.data.dtype
 @property
 def T(self): return self.transpose(axes=tuple(range(self.ndim))[::-1])

 def __repr__(self):
   grad_fn_info = f", grad_fn=<{self._backward.__closure__[0].cell_contents.__class__.__name__}>" if self._backward.__closure__ else ""
   return f"<Tensor shape={self.shape} on {HAL.device}{grad_fn_info}>"

 def __hash__(self): return id(self)

 def backward(self):
  topo, visited = [], set()
  def build_topo(v):
   if v not in visited:
    visited.add(v)
    for p in v._prev: build_topo(p)
    topo.append(v)
  build_topo(self)

  self.grad = HAL.ones(self.shape, dtype=self.dtype)
  for v in reversed(topo): v._backward()

 @staticmethod
 def _ensure_tensor(other):
  return other if isinstance(other, Tensor) else Tensor(other)

 def __add__(self, other): return Add.apply(self, Tensor._ensure_tensor(other))
 def __mul__(self, other): return Mul.apply(self, Tensor._ensure_tensor(other))
 def __radd__(self, other): return self + other
 def __rmul__(self, other): return self * other
 def __neg__(self): return self * -1.0
 def __sub__(self, other): return self + (-other)
 def __rsub__(self, other): return Tensor._ensure_tensor(other) - self

 def matmul(self, other): return MatMul.apply(self, Tensor._ensure_tensor(other))
 def relu(self): return ReLU.apply(self)
 def tanh(self): return Tanh.apply(self)
 def sum(self, axis=None, keepdims=False): return Sum.apply(self, axis=axis, keepdims=keepdims)
 def __getitem__(self, arg): return Slice.apply(self, arg=arg)
 def exp(self): return Exp.apply(self)
 def log(self): return Log.apply(self)
 def softmax(self, axis=-1): return Softmax.apply(self, axis=axis)

 def reshape(self, *shape):
  if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
  return Reshape.apply(self, shape=shape)

 def transpose(self, axes): return Transpose.apply(self, axes=axes)

 def mean(self, axis=None, keepdims=False):
  total_sum = self.sum(axis=axis, keepdims=keepdims)
  num_elements = np.prod(self.shape) / np.prod(total_sum.shape) if total_sum.shape else np.prod(self.shape)
  return total_sum * (1.0 / num_elements)

 def detach(self):
  return Tensor(self.data, requires_grad=False)
