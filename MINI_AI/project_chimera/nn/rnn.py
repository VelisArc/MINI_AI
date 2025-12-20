# project_chimera/nn/rnn.py
from .module import Module
from .linear import Linear

class SimpleRNNCell(Module):
    """A single step of a Simple RNN, now in its proper module."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = Linear(input_size, hidden_size)
        self.W_hh = Linear(hidden_size, hidden_size)
    
    def forward(self, x, h):
        # The tanh method is part of our Tensor class, so this works.
        return (self.W_ih(x) + self.W_hh(h)).tanh()
