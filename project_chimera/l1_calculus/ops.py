# project_chimera/l1_calculus/ops.py
import numpy as np
from ..l0_hal.hardware_abstraction import HAL, ARRAY_LIB

try:
    import torch
    import torch.nn.functional as F
    USE_TORCH_KERNELS = True
    print("[OPS] PyTorch kernels found. Conv/TransposeConv operations will be fast-tracked.")
except ImportError:
    USE_TORCH_KERNELS = False
    print("[OPS] WARNING: PyTorch not found. Conv/TransposeConv ops will raise an error.")

def to_torch(arr, requires_grad=False):
    if not USE_TORCH_KERNELS:
        raise ImportError("PyTorch not found, cannot convert to Torch tensor.")
    if HAL.USE_GPU:
        t = torch.as_tensor(arr, device='cuda')
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr))
    t.requires_grad = requires_grad
    return t

def to_numpy(t):
    if not isinstance(t, torch.Tensor):
        return HAL.to_device(t)
    if t.device.type == 'cuda' and HAL.USE_GPU:
        return HAL.ARRAY_LIB.asarray(t.detach())
    elif t.device.type == 'cuda' and not HAL.USE_GPU:
        return t.detach().cpu().numpy()
    else:
        return t.detach().cpu().numpy()

class Function:
    def __init__(self, *tensors, **kwargs):
        from .tensor import Tensor
        self.parents = [t for t in tensors if isinstance(t, Tensor)]
        self.saved_tensors = []
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    @classmethod
    def apply(cls, *args, **kwargs):
        from .tensor import Tensor
        ctx = cls(*args, **kwargs)
        raw_args = [arg.data if isinstance(arg, Tensor) else arg for arg in args]
        raw_output = ctx.forward(*raw_args)
        output_requires_grad = any(t.requires_grad for t in ctx.parents)
        output = Tensor(raw_output, requires_grad=output_requires_grad)

        if output.requires_grad:
            output._prev = set(ctx.parents)
            def _backward_closure():
                if output.grad is None: return
                grads = ctx.backward(output.grad)
                if not isinstance(grads, tuple): grads = (grads,)

                for p, g_data in zip(ctx.parents, grads):
                    if g_data is not None and p.requires_grad:
                        if g_data.shape != p.data.shape:
                            if g_data.ndim > p.data.ndim:
                                prepended_dims = g_data.ndim - p.data.ndim
                                sum_axes = tuple(range(prepended_dims))
                                g_data = HAL.ARRAY_LIB.sum(g_data, axis=sum_axes)
                            sum_axes = tuple(i for i, dim in enumerate(p.data.shape) if dim == 1 and g_data.shape[i] > 1)
                            if sum_axes:
                                g_data = HAL.ARRAY_LIB.sum(g_data, axis=sum_axes, keepdims=True)
                        if p.grad is None: p.grad = HAL.zeros_like(p.data)
                        if p.grad.shape != g_data.shape:
                             g_data = g_data.reshape(p.grad.shape)
                        p.grad += g_data
            output._backward = _backward_closure
        return output

    def forward(self, *args, **kwargs): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

# --- Basic Ops ---
class Add(Function):
    def forward(self, x, y): return x + y
    def backward(self, g): return g, g

class Mul(Function):
    def forward(self, x, y): self.save_for_backward(x, y); return x * y
    def backward(self, g): x, y = self.saved_tensors; return g * y, g * x

class Power(Function):
    def forward(self, x, y): self.save_for_backward(x, y); return x ** y
    def backward(self, g):
        x, y = self.saved_tensors
        return g * y * (x ** (y - 1)), g * (x ** y) * HAL.log(x)

# --- ⚠️ फिक्स: अंतिम "स्मार्ट" MatMul.backward ---
class MatMul(Function):
    def forward(self, x, y): 
        self.save_for_backward(x, y); 
        return HAL.matmul(x, y)
    
    def backward(self, g):
        x, y = self.saved_tensors
        
        # ग्रेडिएंट x के लिए (dx) - यह हमेशा सही होता है
        # g @ y.T
        dx = HAL.matmul(g, y.swapaxes(-1, -2))
        
        # ग्रेडिएंट y के लिए (dy)
        # --- ⚠️ यह है नया लॉजिक ---
        if y.ndim == 2:
            # केस 1: (B, S, Ein) @ (Ein, Eout) -> यह 'Linear' लेयर है
            # हमें बैच डायमेंशन को सम (sum) करना होगा
            x_flat = x.reshape(-1, x.shape[-1])
            g_flat = g.reshape(-1, g.shape[-1])
            # dy = (Ein, B*S) @ (B*S, Eout) -> (Ein, Eout)
            dy = HAL.matmul(x_flat.swapaxes(-1, -2), g_flat)
        else:
            # केस 2: (B, H, S, S) @ (B, H, S, D) -> यह 'SelfAttention' है
            # हमें बैच डायमेंशन को सम (sum) नहीं करना चाहिए
            # dy = x.T @ g
            dy = HAL.matmul(x.swapaxes(-1, -2), g)
        # --- फिक्स समाप्त ---
            
        return dx, dy
# --- फिक्स समाप्त ---

class Sum(Function):
    def forward(self, x): 
        self.input_shape = x.shape
        return HAL.sum(x, axis=self.axis, keepdims=self.keepdims)
    def backward(self, g):
        if not self.keepdims and self.axis is not None:
            shape = list(g.shape); axes = self.axis if hasattr(self.axis, '__iter__') else [self.axis]; [shape.insert(ax, 1) for ax in sorted(axes)]; g = g.reshape(tuple(shape))
        return g * HAL.ones(self.input_shape)

# --- Activation & Unary Ops ---
class ReLU(Function):
    def forward(self, x): self.save_for_backward(x); return ARRAY_LIB.maximum(0, x)
    def backward(self, g): x, = self.saved_tensors; return g * (x > 0)

class Tanh(Function):
    def forward(self, x): self.output = ARRAY_LIB.tanh(x); return self.output
    def backward(self, g): return g * (1 - self.output**2)

class Exp(Function):
    def forward(self, x): self.output = HAL.exp(x); return self.output
    def backward(self, g): return g * self.output

class Log(Function):
    def forward(self, x): self.save_for_backward(x); return HAL.log(x)
    def backward(self, g): x, = self.saved_tensors; return g / (x + 1e-8)

class Softmax(Function):
    def forward(self, x):
        e_x = HAL.exp(x - HAL.max(x, axis=self.axis, keepdims=True));
        self.output = e_x / HAL.sum(e_x, axis=self.axis, keepdims=True);
        return self.output
    def backward(self, g): s = self.output; return s * (g - ARRAY_LIB.sum(g * s, axis=-1, keepdims=True))

# --- Reshaping and Slicing Ops ---
class Reshape(Function):
    def forward(self, x): self.input_shape = x.shape; return x.reshape(self.shape)
    def backward(self, g): return g.reshape(self.input_shape)

class Transpose(Function):
    def forward(self, x): return x.transpose(self.axes)
    def backward(self, g):
        inv_axes = list(range(len(self.axes)));
        [inv_axes.__setitem__(axis, i) for i, axis in enumerate(self.axes)];
        return g.transpose(tuple(inv_axes))

class Slice(Function):
    def forward(self, x): self.input_shape = x.shape; return x[self.arg]
    def backward(self, g):
        grad = HAL.zeros_like(self.parents[0].data);
        grad[self.arg] = g;
        return grad

# --- Complex Neural Network Ops ---
class EmbeddingOp(Function):
    def forward(self, w): return w[self.indices]
    def backward(self, g):
        grad = HAL.zeros_like(self.parents[0].data);
        HAL.add_at(grad, self.indices, g);
        return grad

class LayerNormOp(Function):
    def forward(self, x, gamma, beta):
        mean = x.mean(axis=-1, keepdims=True); var = x.var(axis=-1, keepdims=True)
        x_normalized = (x - mean) / ARRAY_LIB.sqrt(var + self.eps)
        self.save_for_backward(x, gamma, mean, var, x_normalized)
        return gamma * x_normalized + beta

    def backward(self, g):
        x, gamma, mean, var, x_normalized = self.saved_tensors
        N = x.shape[-1]
        
        axes_to_sum = tuple(range(g.ndim - 1))
        if not axes_to_sum:
             axes_to_sum = None

        grad_gamma = (g * x_normalized).sum(axis=axes_to_sum)
        grad_beta = g.sum(axis=axes_to_sum)

        inv_std = 1. / ARRAY_LIB.sqrt(var + self.eps)
        grad_x_normalized = g * gamma
        
        grad_var = (grad_x_normalized * (x - mean) * -0.5 * inv_std**3).sum(axis=-1, keepdims=True)
        grad_mean = (grad_x_normalized * -inv_std).sum(axis=-1, keepdims=True) + (grad_var * (-2. * (x - mean)).sum(axis=-1, keepdims=True) / N)
        
        grad_x = (grad_x_normalized * inv_std) + (grad_var * 2. * (x - mean) / N) + (grad_mean / N)
        
        return grad_x, grad_gamma, grad_beta

class RMSNormOp(Function):
    def forward(self, x, gamma):
        # RMS = sqrt(mean(x^2) + eps)
        # x_norm = x / RMS
        mean_square = (x ** 2).mean(axis=-1, keepdims=True)
        rms = ARRAY_LIB.sqrt(mean_square + self.eps)
        x_normalized = x / rms
        self.save_for_backward(x, gamma, rms, x_normalized)
        return gamma * x_normalized

    def backward(self, g):
        x, gamma, rms, x_normalized = self.saved_tensors
        N = x.shape[-1]

        axes_to_sum = tuple(range(g.ndim - 1))
        if not axes_to_sum: axes_to_sum = None

        grad_gamma = (g * x_normalized).sum(axis=axes_to_sum)

        # dL/dx = gamma * dL/dy * dy/dx
        # dy/dx involves the complex gradient of normalization
        # Simplified: dx = (1/RMS) * (dy - x_norm * (dy . x_norm))
        # where dy = g * gamma

        grad_output = g * gamma
        dot_prod = (grad_output * x_normalized).sum(axis=-1, keepdims=True)
        grad_x = (grad_output - x_normalized * dot_prod) / rms

        return grad_x, grad_gamma

# --- PyTorch Accelerated Ops ---
class Conv2dOp(Function):
    def forward(self, x, w, b):
        if not USE_TORCH_KERNELS: raise ImportError("PyTorch not found, cannot perform Conv2d.")
        self.save_for_backward(x, w)
        return to_numpy(F.conv2d(to_torch(x), to_torch(w), to_torch(b), self.stride, self.padding))

    def backward(self, dout):
        x, w = self.saved_tensors
        x_t, w_t, dout_t = to_torch(x, True), to_torch(w, True), to_torch(dout)
        y_t = F.conv2d(x_t, w_t, None, self.stride, self.padding)
        grads = torch.autograd.grad(y_t, [x_t, w_t], dout_t, retain_graph=False)
        dx, dw = to_numpy(grads[0]), to_numpy(grads[1])
        db = dout.sum(axis=(0, 2, 3))
        return dx, dw, db

class ConvTranspose2dOp(Function):
    def forward(self, x, w, b):
        if not USE_TORCH_KERNELS: raise ImportError("PyTorch not found, cannot perform ConvTranspose2d.")
        self.save_for_backward(x, w)
        return to_numpy(F.conv_transpose2d(to_torch(x), to_torch(w), to_torch(b), self.stride, self.padding))

    def backward(self, dout):
        x, w = self.saved_tensors
        x_t, w_t, dout_t = to_torch(x, True), to_torch(w, True), to_torch(dout)
        y_t = F.conv_transpose2d(x_t, w_t, None, self.stride, self.padding)
        grads = torch.autograd.grad(y_t, [x_t, w_t], dout_t, retain_graph=False)
        dx, dw = to_numpy(grads[0]), to_numpy(grads[1])
        db = dout.sum(axis=(0, 2, 3))
        return dx, dw, db

class Concat(Function):
    def forward(self, *args):
        self.axis = args[-1]
        self.inputs = args[:-1]
        self.save_for_backward(*self.inputs)

        # Extract data from tensors
        arrays = [x.data if hasattr(x, 'data') else x for x in self.inputs]
        return HAL.ARRAY_LIB.concatenate(arrays, axis=self.axis)

    def backward(self, g):
        # We need to slice g back into the original shapes
        grads = []
        start_idx = 0
        for inp in self.inputs:
            # Determine size along the concatenation axis
            dim_size = inp.shape[self.axis]

            # Construct slice object dynamically
            # slice(None) is equivalent to ':'
            slices = [slice(None)] * g.ndim
            slices[self.axis] = slice(start_idx, start_idx + dim_size)

            grads.append(g[tuple(slices)])
            start_idx += dim_size

        return tuple(grads)
