"""
#!/bin/bash
# Caelonyx Quick Start Script (UPGRADED)

echo "================================"
echo "  CAELONYX AGI - Quick Start"
echo "================================"

# Check GPU
echo ""
echo "[1/6] Checking hardware..."
python3 -c "from project_chimera.l0_hal.hardware_abstraction import HAL; print(f'GPUs: {HAL.get_gpu_count()}'); print(f'Device: {HAL.device}')"

# Create directories
echo ""
echo "[2/6] Creating directories..."
mkdir -p datasets vqvae_results agent_generations

# Create sample data
echo ""
echo "[3/6] Creating sample dataset..."
cat > datasets/train.txt << EOF
hello world
machine learning is amazing
deep learning transforms data
neural networks learn patterns
artificial intelligence evolves
cognitive systems think deeply
EOF

# Run diagnosis
echo ""
echo "[4/6] Running system diagnosis..."
python3 diagnose.py || echo "Some optional components missing (OK)"

# Train VQ-VAE (Multi-GPU if available)
echo ""
echo "[5/6] Training VQ-VAE..."
if [ $(python3 -c "from project_chimera.l0_hal.hardware_abstraction import HAL; print(HAL.get_gpu_count())") -gt 1 ]; then
    echo "Multi-GPU detected! Using distributed training..."
    torchrun --nproc_per_node=$(python3 -c "from project_chimera.l0_hal.hardware_abstraction import HAL; print(HAL.get_gpu_count())") train_vqvae_multi_gpu.py --epochs 5
else
    python3 train_vqvae.py
fi

# Train Agent
echo ""
echo "[6/6] Training Agent..."
python3 train_agent.py --data_path datasets/train.txt --epochs 100

echo ""
echo "================================"
echo "  ✓ Setup Complete!"
echo "================================"
echo ""
echo "Run interactive agent:"
echo "  python3 run_agent.py"
echo ""
"""

