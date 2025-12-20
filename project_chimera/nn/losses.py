# project_chimera/nn/losses.py
from .module import Module
from project_chimera.l1_calculus.tensor import Tensor
class MSELoss(Module):
    def forward(self, p, t): diff = p - t; return (diff * diff).sum() * (1.0 / diff.shape[0])
class CrossEntropyLoss(Module):
    def log_softmax(self, x: Tensor) -> Tensor:
        max_val = Tensor(x.data.max(axis=1, keepdims=True))
        x_stable = x - max_val
        log_sum_exp = x_stable.exp().sum(axis=1, keepdims=True).log()
        return x_stable - log_sum_exp
    def forward(self, predictions: Tensor, targets) -> Tensor:
        batch_size = predictions.shape[0]
        log_probs = self.log_softmax(predictions)
        selected_log_probs = log_probs[range(batch_size), targets]
        return selected_log_probs.sum() * (-1.0 / batch_size)
