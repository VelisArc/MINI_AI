# project_chimera/nn/positional_encoding.py
import numpy as np
from .module import Module
from ..l1_calculus.tensor import Tensor

class PositionalEncoding(Module):
  """
  Injects position information into the input embeddings.
  (ब्रॉडकास्टिंग बग फिक्स के साथ)
  """
  def __init__(self, embed_size: int, max_len: int = 5000):
    super().__init__()
    pe = np.zeros((max_len, embed_size), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, embed_size, 2, dtype=np.float32) * -(np.log(10000.0) / embed_size))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe.reshape(1, max_len, embed_size)
    self.pe = Tensor(pe, requires_grad=False)

  def forward(self, x: Tensor) -> Tensor:
    seq_len = x.shape[1]
    # --- THE FIX IS HERE ---
    # self.pe को सही ढंग से स्लाइस करें और Tensor.__add__ ब्रॉडकास्टिंग को हैंडल करेगा।
    pos_encoding_slice = self.pe[:, :seq_len, :]
    return x + pos_encoding_slice

  def parameters(self):
    return []
