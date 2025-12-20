import numpy as np
import os
import subprocess

USE_GPU = os.environ.get('PROMETHEUS_USE_GPU', 'false').lower() == 'true'
ARRAY_LIB = np
TORCH_AVAILABLE = False
CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    print("[HAL] ✓ PyTorch found")
except ImportError:
    print("[HAL] ⚠ PyTorch not found")

if USE_GPU:
    try:
        import cupy as cp
        CUPY_AVAILABLE = True
        ARRAY_LIB = cp
        print("[HAL] ✓ CuPy enabled - GPU acceleration active")
    except ImportError:
        print("[HAL] ⚠ CuPy not found - falling back to CPU")
        USE_GPU = False
else:
    print("[HAL] Running in CPU mode")

class HAL_Kernels:
    def __init__(self):
        self.device = "GPU" if USE_GPU else "CPU"
        self.USE_GPU = USE_GPU
        self.ARRAY_LIB = ARRAY_LIB
        self.TORCH_AVAILABLE = TORCH_AVAILABLE
        self.CUPY_AVAILABLE = CUPY_AVAILABLE

    def get_gpu_count(self):
        """Multi-GPU detection with fallbacks"""
        if self.TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    print(f"[HAL] Detected {count} GPU(s) via PyTorch")
                    return count
            except Exception as e:
                print(f"[HAL] PyTorch GPU detection failed: {e}")

        if self.CUPY_AVAILABLE:
            try:
                import cupy as cp
                count = cp.cuda.runtime.getDeviceCount()
                print(f"[HAL] Detected {count} GPU(s) via CuPy")
                return count
            except Exception as e:
                print(f"[HAL] CuPy GPU detection failed: {e}")

        try:
            result = subprocess.run(['nvidia-smi', '-L'], 
                                    capture_output=True, text=True, 
                                    timeout=5, check=True)
            count = len(result.stdout.strip().split('\n'))
            print(f"[HAL] Detected {count} GPU(s) via nvidia-smi")
            return count
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

        return 0

    def get_device_memory(self, device_id=0):
        """Get GPU memory info"""
        if self.TORCH_AVAILABLE and self.USE_GPU:
            try:
                import torch
                props = torch.cuda.get_device_properties(device_id)
                total = props.total_memory / (1024**3)  # GB
                allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                return {'total': total, 'allocated': allocated, 'free': total - allocated}
            except Exception:
                pass
        return {'total': 0, 'allocated': 0, 'free': 0}

    def to_device(self, arr):
        if isinstance(arr, self.ARRAY_LIB.ndarray):
            return arr
        if self.USE_GPU:
            return self.ARRAY_LIB.asarray(arr)
        return np.asarray(arr)

    def as_torch(self, arr, device='auto', requires_grad=False):
        if not self.TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        import torch
        
        if device == 'auto':
            device = 'cuda' if self.USE_GPU else 'cpu'
        
        if self.CUPY_AVAILABLE and isinstance(arr, cp.ndarray) and device == 'cuda':
            t = torch.as_tensor(arr, device=device)
        else:
            if self.CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
                arr = arr.get()
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        
        t.requires_grad = requires_grad
        return t

    def from_torch(self, t):
        if not isinstance(t, torch.Tensor):
            return self.to_device(t)
        
        if t.device.type == 'cuda' and self.USE_GPU:
            return self.ARRAY_LIB.asarray(t.detach())
        return t.detach().cpu().numpy()

    # Essential operations
    def zeros_like(self, arr):
        return self.ARRAY_LIB.zeros_like(arr)

    def ones(self, shape, dtype=np.float32):
        return self.ARRAY_LIB.ones(shape, dtype=dtype)

    def matmul(self, A, B):
        return self.ARRAY_LIB.matmul(A, B)

    def log(self, arr):
        return self.ARRAY_LIB.log(arr + 1e-10)

    def exp(self, arr):
        return self.ARRAY_LIB.exp(arr)

    def max(self, arr, axis=None, keepdims=False):
        return arr.max(axis=axis, keepdims=keepdims)

    def sum(self, arr, axis=None, keepdims=False):
        return arr.sum(axis=axis, keepdims=keepdims)

    def add_at(self, arr, indices, values):
        if self.USE_GPU:
            arr.scatter_add(indices, values)
        else:
            np.add.at(arr, indices, values)
        return arr

HAL = HAL_Kernels()

