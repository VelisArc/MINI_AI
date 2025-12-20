#!/usr/bin/env python3
"""
mini_train.py - Quick training test for Caelonyx
यह script पूरे pipeline को 5 minutes में test करेगा
"""

import numpy as np
import os
import sys

print("="*60)
print("  CAELONYX MINI TRAINING TEST (5 Minutes)")
print("="*60)

# ============================================
# STEP 1: Setup Environment
# ============================================
print("\n[1/4] Setting up environment...")
os.makedirs("datasets", exist_ok=True)
os.makedirs("vqvae_results", exist_ok=True)
os.makedirs("agent_generations", exist_ok=True)

# Create minimal training data
with open("datasets/mini_train.txt", "w") as f:
    f.write("hello world\n")
    f.write("machine learning\n")
    f.write("deep learning\n")
    f.write("neural networks\n")
    f.write("artificial intelligence\n")

print("✓ Environment ready")

# ============================================
# STEP 2: Test VQ-VAE (Mini Version)
# ============================================
print("\n[2/4] Testing VQ-VAE (5 epochs only)...")

try:
    from project_chimera.cognitive_models.vq_vae import VQVAE
    from project_chimera.l1_calculus.tensor import Tensor
    from project_chimera.l1_calculus.optimizers import Adam
    from project_chimera.nn.losses import MSELoss
    import torch
    from torchvision import datasets, transforms
    
    # Minimal config
    model = VQVAE(
        in_channels=3,
        hidden_channels=64,  # Reduced from 128
        embedding_dim=32,    # Reduced from 64
        num_embeddings=256   # Reduced from 512
    )
    
    # Load tiny dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Use only 100 samples for speed
    train_subset = torch.utils.data.Subset(train_dataset, range(100))
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=10,
        shuffle=True
    )
    
    optimizer = Adam(model.parameters(), lr=2e-4)
    loss_fn = MSELoss()
    
    print("  Training VQ-VAE for 5 epochs (quick test)...")
    for epoch in range(5):
        total_loss = 0
        for i, (images, _) in enumerate(train_loader):
            if i >= 2:  # Only 2 batches per epoch for speed
                break
            
            input_tensor = Tensor(images.numpy())
            reconstructed, vq_loss, _ = model.forward(input_tensor)
            recon_loss = loss_fn.forward(reconstructed, input_tensor)
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        
        avg_loss = total_loss / 2
        print(f"    Epoch {epoch+1}/5: Loss = {avg_loss:.4f}")
    
    model.save("vqvae_model_mini.npz")
    print("✓ VQ-VAE mini training complete!")
    print("  Model saved: vqvae_model_mini.npz")
    
except Exception as e:
    print(f"✗ VQ-VAE test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# STEP 3: Test Agent Training
# ============================================
print("\n[3/4] Testing Agent Training (50 epochs only)...")

try:
    from project_chimera.p5_agent.caelonyx_agent import CaelonyxAgent
    from project_chimera.l1_calculus.optimizers import Adam
    from project_chimera.nn.losses import CrossEntropyLoss
    
    config = {
        "vq_hidden_channels": 64,
        "vq_embedding_dim": 32,
        "num_visual_tokens": 256,
        "embed_size": 128,      # Reduced from 256
        "num_layers": 2,        # Reduced from 4
        "heads": 2,             # Reduced from 4
        "forward_expansion": 2, # Reduced from 4
        "max_length": 128,      # Reduced from 512
        "text_vocab_size": 500, # Reduced from 2000
        "sos_id": 500,
        "eos_id": 501,
        "image_id": 502,
        "vqvae_path": "vqvae_model_mini.npz",
        "transformer_path": "agent_mini.npz",
        "image_token_count": 4*4  # Reduced from 8*8
    }
    config["total_vocab_size"] = config["text_vocab_size"] + config["num_visual_tokens"] + 3
    
    agent = CaelonyxAgent(config)
    agent.vqvae.load(config["vqvae_path"])
    
    # Train tokenizer on mini corpus
    agent.tokenizer.train([
        "hello world",
        "machine learning",
        "deep learning"
    ])
    
    optimizer = Adam(agent.transformer.parameters(), lr=3e-4)
    loss_fn = CrossEntropyLoss()
    
    # Mini training loop
    print("  Training Agent for 50 epochs (quick test)...")
    for epoch in range(50):
        # Simple text-only training
        text = "hello world"
        tokens = agent.tokenizer.encode(text)
        if not tokens:
            continue
        
        input_seq = [config['sos_id']] + tokens[:-1]
        target_seq = tokens
        
        input_tensor = Tensor(np.array([input_seq]))
        logits = agent.transformer.forward(input_tensor)
        
        # Simplified loss calculation
        loss = loss_fn.forward(
            logits.reshape(-1, config['total_vocab_size']),
            np.array(target_seq)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50: Loss = {loss.data.item():.4f}")
    
    agent.transformer.save(config["transformer_path"])
    print("✓ Agent mini training complete!")
    print("  Model saved: agent_mini.npz")
    
except Exception as e:
    print(f"✗ Agent training test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# STEP 4: Test Inference
# ============================================
print("\n[4/4] Testing Inference...")

try:
    # Reload agent
    agent = CaelonyxAgent(config)
    agent.vqvae.load(config["vqvae_path"])
    agent.transformer.load(config["transformer_path"])
    agent.tokenizer.train(["hello", "world", "test"])
    agent.is_ready = True
    
    print("  Testing chat generation...")
    test_input = "hello"
    tokens = agent.tokenizer.encode(test_input)
    input_seq = [config['sos_id']] + tokens
    
    # Generate 3 tokens
    for _ in range(3):
        input_tensor = Tensor(np.array([input_seq]))
        logits = agent.transformer.forward(input_tensor)
        predicted_id = np.argmax(logits.data[0, -1, :])
        
        if predicted_id >= config['text_vocab_size']:
            break
        
        input_seq.append(int(predicted_id))
    
    print(f"  Input: '{test_input}'")
    print(f"  Generated tokens: {input_seq}")
    print("✓ Inference test complete!")
    
except Exception as e:
    print(f"✗ Inference test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# SUCCESS!
# ============================================
print("\n" + "="*60)
print("  ✅ ALL TESTS PASSED!")
print("="*60)
print("\nMini models created:")
print("  • vqvae_model_mini.npz")
print("  • agent_mini.npz")
print("\nNext steps:")
print("  1. Run full training: python3 train_vqvae.py")
print("  2. Train agent: python3 train_agent.py --data_path datasets/train.txt")
print("  3. Run interactive: python3 run_agent.py")
print("\nNote: Mini models are for testing only. Use full training for real applications.")
print("="*60)
