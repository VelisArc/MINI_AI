# project_chimera/nn/vector_quantizer.py
import numpy as np
from .module import Module
from .embedding import Embedding
from ..l1_calculus.tensor import Tensor

class VectorQuantizer(Module):
    """
    Implements the Vector Quantization layer.
    (TypeError फिक्स के साथ)
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, latents: Tensor):
        original_shape = latents.shape
        assert original_shape[1] == self.embedding_dim, "Input channels must match embedding dim"
        
        flat_latents = latents.reshape((-1, self.embedding_dim))
        
        distances = (
            (flat_latents * flat_latents).sum(axis=1, keepdims=True)
            - 2 * flat_latents.matmul(self.embedding.weight.T)
            + (self.embedding.weight * self.embedding.weight).sum(axis=1, keepdims=False).T
        )
        
        encoding_indices = distances.data.argmin(axis=1)
        quantized_latents_flat_data = self.embedding.weight.data[encoding_indices]
        
        quantized_latents_flat = Tensor(quantized_latents_flat_data, requires_grad=False)
        quantized_latents = quantized_latents_flat.reshape(original_shape)
        
        quantized_tensor = latents + (quantized_latents - latents).detach()
        
        # --- THE FIX IS HERE ---
        # `**2` को `* self` से बदला गया
        
        # VQ Loss (कोडबुक लॉस)
        diff1 = latents.detach() - quantized_latents
        vq_loss = (diff1 * diff1).mean()
        
        # Commitment Loss
        diff2 = latents - quantized_latents.detach()
        commitment_loss = (diff2 * diff2).mean()
        
        total_vq_loss = vq_loss + self.commitment_cost * commitment_loss
        
        return quantized_tensor, total_vq_loss, encoding_indices
