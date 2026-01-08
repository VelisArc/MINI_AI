from .module import Module
from ..l1_calculus.tensor import Tensor
from ..l0_hal.hardware_abstraction import HAL, ARRAY_LIB
import numpy as np

class RotaryEmbedding(Module):
    def __init__(self, dim, max_len=5000, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        # Precompute freqs
        self._precompute_freqs(dim, max_len, base)

    def _precompute_freqs(self, dim, max_len, base):
        # theta = base ^ (-2 * i / dim)
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_len).astype(np.float32)

        # freqs = outer(t, inv_freq) -> (max_len, dim/2)
        freqs = np.outer(t, inv_freq)

        # We need cos and sin for the rotation matrix
        # emb = [x0, x1, x2, x3, ...]
        # rotated = [x0*cos - x1*sin, x0*sin + x1*cos, ...]
        # So we repeat freqs to match the last dimension

        # (max_len, dim/2) -> (max_len, dim) by repeating each freq twice
        # But RoPE usually applies to pairs (0,1), (2,3)...
        # Standard implementation:
        # x_rotated = (x * cos) + (rotate_half(x) * sin)
        # where rotate_half([-x1, x0, -x3, x2, ...])

        # So we just need cos and sin of shape (1, 1, max_len, dim/2) to broadcast?
        # Or (1, max_len, 1, dim/2) depending on layout.
        # SelfAttention is (B, Heads, Seq, HeadDim) after transpose.
        # So we want (1, 1, Seq, HeadDim).
        # But wait, HeadDim must be even.

        # Let's keep cos/sin as (max_len, dim/2) for now and broadcast properly in forward.

        self.cos_cached = Tensor(np.cos(freqs), requires_grad=False)
        self.sin_cached = Tensor(np.sin(freqs), requires_grad=False)

    def forward(self, x, seq_len=None):
        # x: (B, Heads, SeqLen, HeadDim)
        # We need to extract the relevant cos/sin slices
        if seq_len is None:
            seq_len = x.shape[2]

        # Slice: (SeqLen, HeadDim/2)
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]

        # Reshape for broadcasting: (1, 1, SeqLen, HeadDim/2)
        # Since x is (B, H, S, D), we want to broadcast over B and H.
        # We need to make sure HeadDim matches.
        # x is split into x1, x2 each of size D/2.

        return cos, sin

def rotate_half(x):
    # x: (..., D)
    # split into x1, x2
    # returns [-x2, x1]

    # We need a generic way to split last dim.
    # Since we don't have 'split' op, we use slice.
    d = x.shape[-1]
    half = d // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    # -x2
    neg_x2 = -x2

    # concatenate
    # We need a concat function in Tensor or Ops.
    return Tensor.cat([neg_x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, H, S, D)
    # cos, sin: (S, D/2) -> need reshape to (1, 1, S, D/2)

    # Expand cos/sin to match batch and heads
    # cos: (S, D/2) -> (1, 1, S, D/2)
    cos = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.reshape(1, 1, sin.shape[0], sin.shape[1])

    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # But q is (..., D) and cos is (..., D/2).
    # We need to duplicate cos/sin or reshape q.

    # Standard RoPE strategy:
    # q = [q1, q2]
    # rotated = [q1*cos - q2*sin, q1*sin + q2*cos]
    # This is equivalent to q * cos_repeated + rotate_half(q) * sin_repeated

    # Where cos_repeated is [c1, c1, c2, c2, ...] or [c1..cn, c1..cn] depending on implementation.
    # Our rotate_half is [-x2, x1].
    # Corresponds to complex multiplication x * e^(i*theta)
    # (x1 + i x2) * (c + i s) = (x1c - x2s) + i (x1s + x2c)
    # Real part: x1c - x2s
    # Imag part: x2c + x1s
    # So new vector is [x1c - x2s, x2c + x1s]

    # If our storage is [x1, x2, x3, x4...] where pairs are adjacent?
    # Or [x1...xN/2, y1...yN/2]?
    # Llama uses the latter usually with polar coordinates, but standard implementation often assumes adjacent pairs or split half/half.
    # The `_precompute_freqs` above assumes we split [0..D/2] and [D/2..D].
    # So we should treat x as concatenation of x_real and x_imag.

    # Let's ensure consistency.
    # If we define rotate_half(x) as [-x2, x1] where x=[x1, x2] (split by half),
    # Then:
    #   q_new = q * cos + rotate_half(q) * sin
    #   x1_new = x1*c + (-x2)*s = x1c - x2s
    #   x2_new = x2*c + (x1)*s = x2c + x1s
    # This matches the complex multiplication logic exactly.

    # So we need cos and sin to be concatenated with themselves to match D.

    # cos is (1, 1, S, D/2). We need (1, 1, S, D).
    # cat([cos, cos], dim=-1)
    cos = Tensor.cat([cos, cos], axis=-1)
    sin = Tensor.cat([sin, sin], axis=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
