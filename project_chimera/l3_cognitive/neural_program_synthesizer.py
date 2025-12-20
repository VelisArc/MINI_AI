# project_chimera/l3_cognitive/neural_program_synthesizer.py
from ..nn.module import Module
from ..nn.embedding import Embedding
from ..nn.linear import Linear
from ..nn.activations import ReLU

class ProgramSynthesizer(Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = Embedding(vocab_size, embed_size)
        self.encoder = Linear(embed_size, hidden_size)
        self.decoder = Linear(hidden_size, vocab_size)
        # <META_ENGINE_HOOK_LAYERS>
        
    def forward(self, x_tokens):
        x = self.embedding(x_tokens)
        x = x.mean(axis=1)
        h = self.encoder(x).relu()
        # <META_ENGINE_HOOK_FORWARD>
        logits = self.decoder(h)
        return logits
