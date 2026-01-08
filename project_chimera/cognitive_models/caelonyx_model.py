# project_chimera/cognitive_models/caelonyx_model.py
from ..nn.module import Module
from ..nn.embedding import Embedding
from ..nn.positional_encoding import PositionalEncoding
from ..nn.transformer_block import TransformerBlock
from ..nn.containers import Sequential
from ..nn.linear import Linear

class CaelonyxModel(Module):
 """
 The main Caelonyx Language Model, using a robust Transformer architecture.
 Now upgraded with Rotary Positional Embeddings (RoPE).
 """
 def __init__(
  self, vocab_size: int, embed_size: int, num_layers: int,
  heads: int, forward_expansion: int, max_length: int
 ):
  super().__init__()
  self.embed_size = embed_size
  self.embedding = Embedding(vocab_size, embed_size)

  # Removed absolute PositionalEncoding in favor of RoPE in SelfAttention
  # self.positional_encoding = PositionalEncoding(embed_size, max_length)

  # --- FIX: Use Sequential correctly ---
  self.layers = Sequential(
   *[TransformerBlock(embed_size, heads, forward_expansion, use_rope=True, max_len=max_length) for _ in range(num_layers)]
  )

  self.fc_out = Linear(embed_size, vocab_size)

 def forward(self, x_tokens, mask=None):
  x = self.embedding(x_tokens)
  # x = self.positional_encoding(x) # Removed for RoPE

  # Sequential container passes the input through each layer.
  # We assume mask handling needs to be done inside if needed,
  # but for standard LLM, the mask is often handled in the attention layer.
  # Our TransformerBlock now handles this correctly.
  # Note: If mask needs to be passed to each block, manual iteration is needed.
  # For simplicity and given the fix in TransformerBlock, this is cleaner.
  for layer in self.layers.modules:
    x = layer(x, mask=mask)

  logits = self.fc_out(x)
  return logits
