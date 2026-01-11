# Google Colab Setup for Project Chimera

import os
import subprocess
import sys

def setup_colab_environment():
    print("[Colab Setup] Setting up environment...")

    # 1. Check GPU
    gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if gpu_info.returncode == 0:
        print("[Colab Setup] GPU detected!")
        os.environ['PROMETHEUS_USE_GPU'] = 'true'
    else:
        print("[Colab Setup] GPU not found. Running in CPU mode (slower).")
        os.environ['PROMETHEUS_USE_GPU'] = 'false'

    # 2. Install Dependencies
    print("[Colab Setup] Installing dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'torch', 'torchvision'])

    if os.environ.get('PROMETHEUS_USE_GPU') == 'true':
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'])
        except Exception:
             print("[Colab Setup] Warning: Could not install CuPy. Falling back to PyTorch/NumPy.")

    # 3. Setup Python Path
    project_path = os.getcwd()
    if project_path not in sys.path:
        sys.path.append(project_path)
    print(f"[Colab Setup] Added {project_path} to sys.path")

    print("[Colab Setup] Setup complete! You can now run the agent.")

if __name__ == "__main__":
    setup_colab_environment()
