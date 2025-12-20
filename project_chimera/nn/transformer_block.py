# project_chimera/nn/transformer_block.py
from .module import Module
from .self_attention import SelfAttention
from .linear import Linear
from .layernorm import LayerNorm
from .activations import ReLU
from .containers import Sequential

class TransformerBlock(Module):
 """
 A complete transformer block.
 NOW FIXED to handle a single input for self-attention.
 """
 def __init__(self, embed_size, heads, forward_expansion=4):
  super().__init__()
  self.attention = SelfAttention(embed_size, heads)
  self.norm1 = LayerNorm(embed_size)
  self.norm2 = LayerNorm(embed_size)

  ff_hidden_size = forward_expansion * embed_size
  self.feed_forward = Sequential(
   Linear(embed_size, ff_hidden_size),
   ReLU(),
   Linear(ff_hidden_size, embed_size)
  )

 # --- THE FIX IS HERE ---
 def forward(self, x, mask=None):
  # When value, key, and query are the same, it's self-attention.
  attention_out = self.attention(values=x, keys=x, query=x, mask=mask)
  h = self.norm1(attention_out + x)
  forward_out = self.feed_forward(h)
  out = self.norm2(forward_out + h)
  return out
