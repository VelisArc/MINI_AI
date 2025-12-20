# project_chimera/nn/embedding.py
import numpy as np
from .module import Module
from ..l1_calculus.tensor import Tensor
# --- FIX IS HERE: Import the dedicated Embedding operation ---
from ..l1_calculus.ops import EmbeddingOp

class Embedding(Module):
 """
 An Embedding layer that maps integer indices to dense vectors.
 UPGRADED to use the autograd-aware EmbeddingOp for proper graph tracking.
 """
 def __init__(self, num_embeddings: int, embedding_dim: int):
  super().__init__()
  self.num_embeddings = num_embeddings
  self.embedding_dim = embedding_dim

  # The learnable weights of the embedding layer
  self.weight = Tensor(
   np.random.randn(num_embeddings, embedding_dim) * 0.1,
   requires_grad=True
  )

 def forward(self, input_tensor: Tensor) -> Tensor:
  """
  Takes a Tensor of integer indices and returns the corresponding embeddings.
  Input shape: (batch_size, sequence_length)
  Output shape: (batch_size, sequence_length, embedding_dim)
  """
  # --- THE FIX IS HERE ---
  # Instead of a manual numpy loop, we use the EmbeddingOp.
  # This op understands how to handle tensors and build the computational graph.
  if not isinstance(input_tensor, Tensor):
    raise TypeError(f"Input must be a Tensor, but got {type(input_tensor)}")

  # We pass the weight tensor and the indices (as a numpy array) to the op.
  return EmbeddingOp.apply(self.weight, indices=input_tensor.data.astype(int))

 def __repr__(self):
  return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
