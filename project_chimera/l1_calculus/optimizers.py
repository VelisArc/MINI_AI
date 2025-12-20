# project_chimera/l1_calculus/optimizers.py
from ..l0_hal.hardware_abstraction import HAL, ARRAY_LIB

# Base class
class Optimizer:
    def __init__(self, params):
        self.params = list(params) # Ensure it's a list

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                # Use HAL-aware zeros_like to create grad on the correct device (CPU/GPU)
                p.grad = HAL.zeros_like(p.grad)

# SGD Optimizer
class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                # Use HAL's array library (NumPy or CuPy) for the operation
                p.data -= self.lr * p.grad

# Adam Optimizer
class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # Initialize moment vectors on the correct device (CPU/GPU)
        self.m = [HAL.zeros_like(p.data) for p in self.params]
        self.v = [HAL.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Use HAL's array library (NumPy or CuPy) for all operations
            grad = p.grad
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)
            
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            p.data -= self.lr * m_hat / (ARRAY_LIB.sqrt(v_hat) + self.eps)
