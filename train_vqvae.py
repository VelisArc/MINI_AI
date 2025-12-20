# train_vqvae.py
import numpy as np; import os; import time
from project_chimera.cognitive_models.vq_vae import VQVAE
from project_chimera.l1_calculus.tensor import Tensor
from project_chimera.l1_calculus.optimizers import Adam
from project_chimera.nn.losses import MSELoss
# --- GPU फिक्स के लिए HAL इम्पोर्ट करें ---
from project_chimera.l0_hal.hardware_abstraction import HAL, ARRAY_LIB 
try:
    import torch; from torchvision import datasets, transforms; from torchvision.utils import save_image
except ImportError: print("PyTorch/Torchvision की आवश्यकता है।"); exit()

os.makedirs("vqvae_results", exist_ok=True)

def main():
    print("===== प्रोजेक्ट Caelonyx: VQ-VAE ट्रेनिंग (पूर्ण) =====")
    hidden_channels, res_channels, num_resnet_blocks, num_embeddings, embedding_dim, commitment_cost = 128, 64, 2, 512, 64, 0.25
    batch_size, learning_rate, epochs, log_interval = 64, 2e-4, 20, 100

    print("\n[STEP 1] CIFAR-10 डेटासेट लोड किया जा रहा है..."); transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # --- GPU फिक्स: pin_memory=True डेटा ट्रांसफर को तेज़ करता है ---
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=HAL.USE_GPU)
    print("डेटासेट सफलतापूर्वक लोड हो गया।")

    print("\n[STEP 2] VQ-VAE मॉडल और ऑप्टिमाइज़र बनाया जा रहा है..."); model = VQVAE(in_channels=3, embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    optimizer = Adam(model.parameters(), lr=learning_rate); recon_loss_fn = MSELoss(); print("मॉडल तैयार है।")

    print("\n[STEP 3] ट्रेनिंग शुरू हो रही है..."); fixed_images, _ = next(iter(train_loader)); save_image(fixed_images, "vqvae_results/original_images.png", nrow=8, normalize=True)
    total_steps = len(train_loader)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for i, (images, _) in enumerate(train_loader):
            
            # --- GPU फिक्स: डेटा को कुशलता से डिवाइस पर लाएं ---
            if HAL.USE_GPU:
                # 1. Torch DataLoader 'images' को GPU पर ले जाएं
                torch_gpu_images = images.cuda(non_blocking=True)
                # 2. Torch GPU tensor को CuPy array में बदलें (तेज़)
                cupy_data = ARRAY_LIB.asarray(torch_gpu_images)
                # 3. CuPy array से Caelonyx Tensor बनाएं
                input_tensor = Tensor(cupy_data)
            else:
                # 1. मूल CPU-आधारित तरीका
                input_tensor = Tensor(images.numpy())
            # --- फिक्स समाप्त ---

            reconstructed_images_tensor, vq_loss, _ = model.forward(input_tensor)
            recon_loss = recon_loss_fn.forward(reconstructed_images_tensor, input_tensor)
            total_loss = recon_loss + vq_loss
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()
            if (i + 1) % log_interval == 0:
                print(f"E[{epoch+1}/{epochs}], S[{i+1}/{total_steps}], Loss:{total_loss.data.item():.4f}, Recon:{recon_loss.data.item():.4f}, VQ:{vq_loss.data.item():.4f}")

        with torch.no_grad():
            # --- फिक्स: 'fixed_images' के लिए भी यही लॉजिक लागू करें ---
            if HAL.USE_GPU:
                torch_gpu_fixed = fixed_images.cuda(non_blocking=True)
                cupy_fixed_data = ARRAY_LIB.asarray(torch_gpu_fixed)
                fixed_tensor = Tensor(cupy_fixed_data)
            else:
                fixed_tensor = Tensor(fixed_images.numpy())
            # --- फिक्स समाप्त ---

            reconstructed, _, _ = model.forward(fixed_tensor)
            
            # --- फिक्स: save_image के लिए CuPy/NumPy को Torch में बदलें ---
            if HAL.USE_GPU:
                # CuPy array को Torch GPU tensor में बदलें
                torch_reconstructed = torch.as_tensor(reconstructed.data, device='cuda')
            else:
                # NumPy array को Torch CPU tensor में बदलें
                torch_reconstructed = torch.from_numpy(reconstructed.data)
            
            save_image(torch_reconstructed, f"vqvae_results/reconstructed_epoch_{epoch+1}.png", nrow=8, normalize=True)
        print(f"Epoch {epoch+1} पूरा हुआ। समय: {time.time() - epoch_start_time:.2f}s")

    model.save("vqvae_model.npz")
    print("\nट्रेनिंग सफलतापूर्वक पूरी हुई!"); print("प्रशिक्षित VQ-VAE मॉडल 'vqvae_model.npz' में सहेजा गया है।")

if __name__ == "__main__": main()# train_vqvae.py
import numpy as np; import os; import time
from project_chimera.cognitive_models.vq_vae import VQVAE
from project_chimera.l1_calculus.tensor import Tensor
from project_chimera.l1_calculus.optimizers import Adam
from project_chimera.nn.losses import MSELoss
# --- GPU फिक्स के लिए HAL इम्पोर्ट करें ---
from project_chimera.l0_hal.hardware_abstraction import HAL, ARRAY_LIB 
try:
    import torch; from torchvision import datasets, transforms; from torchvision.utils import save_image
except ImportError: print("PyTorch/Torchvision की आवश्यकता है।"); exit()

os.makedirs("vqvae_results", exist_ok=True)

def main():
    print("===== प्रोजेक्ट Caelonyx: VQ-VAE ट्रेनिंग (पूर्ण) =====")
    hidden_channels, res_channels, num_resnet_blocks, num_embeddings, embedding_dim, commitment_cost = 128, 64, 2, 512, 64, 0.25
    batch_size, learning_rate, epochs, log_interval = 64, 2e-4, 20, 100

    print("\n[STEP 1] CIFAR-10 डेटासेट लोड किया जा रहा है..."); transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # --- GPU फिक्स: pin_memory=True डेटा ट्रांसफर को तेज़ करता है ---
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=HAL.USE_GPU)
    print("डेटासेट सफलतापूर्वक लोड हो गया।")

    print("\n[STEP 2] VQ-VAE मॉडल और ऑप्टिमाइज़र बनाया जा रहा है..."); model = VQVAE(in_channels=3, embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    optimizer = Adam(model.parameters(), lr=learning_rate); recon_loss_fn = MSELoss(); print("मॉडल तैयार है।")

    print("\n[STEP 3] ट्रेनिंग शुरू हो रही है..."); fixed_images, _ = next(iter(train_loader)); save_image(fixed_images, "vqvae_results/original_images.png", nrow=8, normalize=True)
    total_steps = len(train_loader)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for i, (images, _) in enumerate(train_loader):
            
            # --- GPU फिक्स: डेटा को कुशलता से डिवाइस पर लाएं ---
            if HAL.USE_GPU:
                # 1. Torch DataLoader 'images' को GPU पर ले जाएं
                torch_gpu_images = images.cuda(non_blocking=True)
                # 2. Torch GPU tensor को CuPy array में बदलें (तेज़)
                cupy_data = ARRAY_LIB.asarray(torch_gpu_images)
                # 3. CuPy array से Caelonyx Tensor बनाएं
                input_tensor = Tensor(cupy_data)
            else:
                # 1. मूल CPU-आधारित तरीका
                input_tensor = Tensor(images.numpy())
            # --- फिक्स समाप्त ---

            reconstructed_images_tensor, vq_loss, _ = model.forward(input_tensor)
            recon_loss = recon_loss_fn.forward(reconstructed_images_tensor, input_tensor)
            total_loss = recon_loss + vq_loss
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()
            if (i + 1) % log_interval == 0:
                print(f"E[{epoch+1}/{epochs}], S[{i+1}/{total_steps}], Loss:{total_loss.data.item():.4f}, Recon:{recon_loss.data.item():.4f}, VQ:{vq_loss.data.item():.4f}")

        with torch.no_grad():
            # --- फिक्स: 'fixed_images' के लिए भी यही लॉजिक लागू करें ---
            if HAL.USE_GPU:
                torch_gpu_fixed = fixed_images.cuda(non_blocking=True)
                cupy_fixed_data = ARRAY_LIB.asarray(torch_gpu_fixed)
                fixed_tensor = Tensor(cupy_fixed_data)
            else:
                fixed_tensor = Tensor(fixed_images.numpy())
            # --- फिक्स समाप्त ---

            reconstructed, _, _ = model.forward(fixed_tensor)
            
            # --- फिक्स: save_image के लिए CuPy/NumPy को Torch में बदलें ---
            if HAL.USE_GPU:
                # CuPy array को Torch GPU tensor में बदलें
                torch_reconstructed = torch.as_tensor(reconstructed.data, device='cuda')
            else:
                # NumPy array को Torch CPU tensor में बदलें
                torch_reconstructed = torch.from_numpy(reconstructed.data)
            
            save_image(torch_reconstructed, f"vqvae_results/reconstructed_epoch_{epoch+1}.png", nrow=8, normalize=True)
        print(f"Epoch {epoch+1} पूरा हुआ। समय: {time.time() - epoch_start_time:.2f}s")

    model.save("vqvae_model.npz")
    print("\nट्रेनिंग सफलतापूर्वक पूरी हुई!"); print("प्रशिक्षित VQ-VAE मॉडल 'vqvae_model.npz' में सहेजा गया है।")

if __name__ == "__main__": main()
