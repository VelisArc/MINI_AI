import numpy as np
import os
import time
import argparse
from project_chimera.cognitive_models.vq_vae import VQVAE
from project_chimera.l1_calculus.tensor import Tensor
from project_chimera.l1_calculus.optimizers import Adam
from project_chimera.nn.losses import MSELoss
from project_chimera.l0_hal.hardware_abstraction import HAL
from project_chimera.l4_distribution.multi_gpu_trainer import MultiGPUTrainer

try:
    import torch
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
except ImportError:
    print("ERROR: PyTorch/Torchvision required")
    exit(1)

def train_vqvae():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mode', type=str, default='ddp', choices=['ddp', 'fsdp'])
    args = parser.parse_args()

    # Initialize multi-GPU trainer
    trainer = MultiGPUTrainer(mode=args.mode)
    
    if trainer.is_main_process():
        print("="*60)
        print("  CAELONYX VQ-VAE TRAINING (MULTI-GPU FIXED)")
        print("="*60)
        print(f"GPUs: {HAL.get_gpu_count()}")
        print(f"Mode: {args.mode.upper()}")
        print(f"Epochs: {args.epochs}")
        os.makedirs("vqvae_results", exist_ok=True)

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=trainer.is_main_process(),
        transform=transform
    )
    
    # Wait for main process to download
    if trainer.is_distributed:
        torch.distributed.barrier()
    
    train_loader = trainer.prepare_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Create model
    model = VQVAE(
        in_channels=3,
        embedding_dim=64,
        num_embeddings=512
    )
    
    # Wrap for multi-GPU
    # Note: For Caelonyx custom models, we keep them on Tensor
    # But for data loading we use PyTorch's efficient distributed sampler
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0
        
        for i, (images, _) in enumerate(train_loader):
            # Move to GPU efficiently
            if HAL.USE_GPU:
                torch_gpu = images.cuda(non_blocking=True)
                cupy_data = HAL.ARRAY_LIB.asarray(torch_gpu)
                input_tensor = Tensor(cupy_data)
            else:
                input_tensor = Tensor(images.numpy())

            # Forward
            reconstructed, vq_loss, _ = model.forward(input_tensor)
            recon_loss = loss_fn.forward(reconstructed, input_tensor)
            total_loss = recon_loss + vq_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.data.item()

            if trainer.is_main_process() and (i + 1) % 100 == 0:
                print(f"Epoch[{epoch+1}/{args.epochs}] Step[{i+1}] "
                      f"Loss:{total_loss.data.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        if trainer.is_main_process():
            print(f"Epoch {epoch+1} complete | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")

    # Save model (only main process)
    if trainer.is_main_process():
        model.save("vqvae_model.npz")
        print("\n✓ Training complete! Model saved.")

    trainer.cleanup()

if __name__ == "__main__":
    train_vqvae()

