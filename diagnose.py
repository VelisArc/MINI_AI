import sys
import os
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def check_hal():
    """Test Hardware Abstraction Layer"""
    print(f"\n{Colors.BOLD}=== HAL CHECK ==={Colors.END}")
    try:
        from project_chimera.l0_hal.hardware_abstraction import HAL
        print_success("HAL imported")
        print_info(f"Device: {HAL.device}")
        print_info(f"GPU Count: {HAL.get_gpu_count()}")
        
        if HAL.get_gpu_count() > 0:
            mem = HAL.get_device_memory(0)
            print_info(f"GPU 0 Memory: {mem['total']:.2f}GB total, {mem['free']:.2f}GB free")
        
        return True
    except Exception as e:
        print_error(f"HAL test failed: {e}")
        return False

def check_multi_gpu():
    """Check multi-GPU support"""
    print(f"\n{Colors.BOLD}=== MULTI-GPU CHECK ==={Colors.END}")
    try:
        from project_chimera.l0_hal.hardware_abstraction import HAL
        gpu_count = HAL.get_gpu_count()
        
        if gpu_count > 1:
            print_success(f"{gpu_count} GPUs detected - Multi-GPU training available!")
            print_info("You can use: torchrun --nproc_per_node=N train_vqvae_multi_gpu.py")
            return True
        elif gpu_count == 1:
            print_warning("Single GPU - Multi-GPU features disabled")
            return True
        else:
            print_warning("No GPUs - CPU only mode")
            return True
    except Exception as e:
        print_error(f"Multi-GPU check failed: {e}")
        return False

def check_dependencies():
    """Check required packages"""
    print(f"\n{Colors.BOLD}=== DEPENDENCIES CHECK ==={Colors.END}")
    
    required = ['numpy', 'torch', 'torchvision']
    optional = ['cupy', 'faiss']
    
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print_success(f"{pkg} installed")
        except ImportError:
            print_error(f"{pkg} NOT installed (REQUIRED)")
            all_ok = False
    
    for pkg in optional:
        try:
            __import__(pkg)
            print_success(f"{pkg} installed")
        except ImportError:
            print_warning(f"{pkg} not installed (optional)")
    
    return all_ok

def check_models():
    """Check trained models"""
    print(f"\n{Colors.BOLD}=== MODELS CHECK ==={Colors.END}")
    
    if Path("vqvae_model.npz").exists():
        print_success("VQ-VAE model found")
    else:
        print_warning("VQ-VAE model not found - run: python3 train_vqvae.py")
    
    if Path("caelonyx_agent_transformer.npz").exists():
        print_success("Agent model found")
    else:
        print_warning("Agent model not found - run: python3 train_agent.py")
    
    return True

def main():
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════╗")
    print("║  CAELONYX SYSTEM DIAGNOSTIC (UPGRADED)    ║")
    print("╚════════════════════════════════════════════╝")
    print(Colors.END)

    results = {
        'HAL': check_hal(),
        'Multi-GPU': check_multi_gpu(),
        'Dependencies': check_dependencies(),
        'Models': check_models()
    }

    print(f"\n{Colors.BOLD}=== SUMMARY ==={Colors.END}")
    passed = sum(results.values())
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED{Colors.END}")
        sys.exit(0)
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ SOME CHECKS FAILED{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()
