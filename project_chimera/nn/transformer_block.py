# project_chimera/nn/transformer_block.py
from .module import Module
from .self_attention import SelfAttention
from .linear import Linear
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .containers import Sequential

class TransformerBlock(Module):
 """
 A complete transformer block.
 Updated with Pre-Norm, RMSNorm, and SwiGLU.
 """
 def __init__(self, embed_size, heads, forward_expansion=4):
  super().__init__()
  self.attention = SelfAttention(embed_size, heads)
  # Replace LayerNorm with RMSNorm (faster)
  self.norm1 = RMSNorm(embed_size)
  self.norm2 = RMSNorm(embed_size)

  ff_hidden_size = forward_expansion * embed_size
  # Replace standard FFN (Linear->ReLU->Linear) with SwiGLU
  self.feed_forward = SwiGLU(embed_size, ff_hidden_size)

 def forward(self, x, mask=None):
  # Pre-Norm Architecture:
  # x = x + Attention(Norm(x))
  # x = x + FFN(Norm(x))

  # 1. Attention block
  x_norm = self.norm1(x)
  attention_out = self.attention(values=x_norm, keys=x_norm, query=x_norm, mask=mask)
  x = x + attention_out

  # 2. Feed Forward block
  x_norm2 = self.norm2(x)
  forward_out = self.feed_forward(x_norm2)
  out = x + forward_out

  return out
