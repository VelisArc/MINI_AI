# project_chimera/nn/self_attention.py
from .module import Module
from ..l1_calculus.tensor import Tensor
from .linear import Linear
import numpy as np

class SelfAttention(Module):
 """
 A true, fully-functional Scaled Dot-Product Self-Attention mechanism.
 This version correctly handles reshaping and transposing for multi-head attention.
 """
 def __init__(self, embed_size, heads):
  super().__init__()
  self.embed_size = embed_size
  self.heads = heads
  self.head_dim = embed_size // heads
  assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

  self.queries_linear = Linear(self.embed_size, self.embed_size, bias=False)
  self.keys_linear = Linear(self.embed_size, self.embed_size, bias=False)
  self.values_linear = Linear(self.embed_size, self.embed_size, bias=False)
  self.fc_out = Linear(self.embed_size, self.embed_size)

 def forward(self, values: Tensor, keys: Tensor, query: Tensor, mask: Tensor = None):
  batch_size = query.shape[0]
  value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

  queries_proj = self.queries_linear(query)
  keys_proj = self.keys_linear(keys)
  values_proj = self.values_linear(values)

  queries = queries_proj.reshape(batch_size, query_len, self.heads, self.head_dim)
  keys = keys_proj.reshape(batch_size, key_len, self.heads, self.head_dim)
  values = values_proj.reshape(batch_size, value_len, self.heads, self.head_dim)

  queries = queries.transpose(axes=(0, 2, 1, 3))
  keys = keys.transpose(axes=(0, 2, 1, 3))
  values = values.transpose(axes=(0, 2, 1, 3))

  keys_T = keys.transpose(axes=(0, 1, 3, 2))
  energy = queries.matmul(keys_T)

  if mask is not None:
   energy = energy + mask

  scaled_energy = energy * (self.head_dim ** -0.5)
  attention = scaled_energy.softmax(axis=-1)
  out = attention.matmul(values)

  out = out.transpose(axes=(0, 2, 1, 3))
  out_concatenated = out.reshape(batch_size, query_len, self.embed_size)
  out = self.fc_out(out_concatenated)
  return out
