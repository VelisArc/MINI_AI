# project_chimera/nn/module.py
from project_chimera.l1_calculus.tensor import Tensor
import inspect
import numpy as np

class Module:
  def parameters(self):
    params = []
    for name, value in inspect.getmembers(self):
      if isinstance(value, Tensor) and value.requires_grad:
        params.append(value)
      elif isinstance(value, Module):
        params.extend(value.parameters())
    # डुप्लिकेट्स हटाएं (यदि एक ही टेंसर कई जगहों पर है)
    return list(dict.fromkeys(params))

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def forward(self, *args, **kwargs):
    raise NotImplementedError("Subclasses must implement the forward pass")

  def zero_grad(self):
    for p in self.parameters():
      if p.grad is not None:
        p.grad = None

  def save(self, filepath: str):
    """Saves all model parameters to a file."""
    print(f"Saving model to {filepath}...")
    param_dict = {f"param_{i}": p.data for i, p in enumerate(self.parameters())}
    np.savez(filepath, **param_dict)
    print("Model saved.")

  def load(self, filepath: str):
    """Loads model parameters from a file."""
    print(f"Loading model from {filepath}...")
    try:
        data = np.load(filepath)
        params = self.parameters()
        assert len(data.files) == len(params), "Mismatched number of parameters in file and model."
        for i, p in enumerate(params):
            p.data = data[f"param_{i}"]
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
