# project_chimera/nn/attention.py
from .module import Module
from ..l1_calculus.tensor import Tensor
from .linear import Linear

class Attention(Module):
    def __init__(self, hidden_size):
        self.W_encoder = Linear(hidden_size, hidden_size)
        self.W_decoder = Linear(hidden_size, hidden_size)
        self.V = Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        batch_size, seq_len, hidden_size = encoder_outputs.shape
        encoder_features = self.W_encoder(encoder_outputs)
        decoder_features = self.W_decoder(decoder_hidden)
        expanded_decoder_features = decoder_features.reshape((batch_size, 1, hidden_size))
        scores = self.V( (encoder_features + expanded_decoder_features).tanh() )
        attention_weights = scores.reshape((batch_size, seq_len)).softmax(axis=1)
        reshaped_weights = attention_weights.reshape((batch_size, seq_len, 1))
        context = (encoder_outputs * reshaped_weights).sum(axis=1)
        return context, attention_weights
