# train_multimodal.py
import numpy as np; import os; import time
from project_chimera.cognitive_models.caelonyx_model import CaelonyxModel
from project_chimera.cognitive_models.vq_vae import VQVAE
from project_chimera.l1_calculus.tensor import Tensor
from project_chimera.l1_calculus.optimizers import Adam
from project_chimera.nn.losses import CrossEntropyLoss
from project_chimera.l2_data.unigram_tokenizer import UnigramTokenizer
try:
    import torch; from torchvision.utils import save_image
except ImportError: print("PyTorch/Torchvision की आवश्यकता है।"); exit()

# --- एक सरल, सिम्युलेटेड डेटासेट ---
def get_simulated_dataset():
    # एक लाल वर्ग बनाएं
    red_square = np.full((3, 32, 32), -1.0, dtype=np.float32)
    red_square[0, 8:24, 8:24] = 1.0 # लाल चैनल
    # एक नीला वृत्त बनाएं
    blue_circle = np.full((3, 32, 32), -1.0, dtype=np.float32)
    center, radius = 16, 8
    for r in range(32):
        for c in range(32):
            if (r - center)**2 + (c - center)**2 < radius**2:
                blue_circle[2, r, c] = 1.0 # नीला चैनल
    
    dataset = {
        "a red square": red_square,
        "a blue circle": blue_circle
    }
    return dataset

def main():
    print("===== Caelonyx: मल्टी-मोडल ट्रांसफॉर्मर ट्रेनिंग =====")
    # --- 1. कॉन्फ़िगरेशन ---
    text_vocab_size = 50
    num_visual_tokens = 512 # VQ-VAE के num_embeddings के बराबर
    # VQ-VAE हाइपरपैरामीटर्स (train_vqvae.py के समान होने चाहिए)
    vq_hidden_channels=128; vq_embedding_dim=64
    
    # ट्रांसफॉर्मर हाइपरपैरामीटर्स
    embed_size = 128; num_layers = 4; heads = 4; max_length = 200
    
    # ट्रेनिंग
    learning_rate = 3e-4; epochs = 500;
    
    # --- 2. टोकनाइज़र और विशेष टोकन ---
    text_tokenizer = UnigramTokenizer(vocab_size=text_vocab_size)
    dataset = get_simulated_dataset()
    text_tokenizer.train(list(dataset.keys()))
    
    # विशेष टोकन IDs
    SOS_ID = text_tokenizer.vocab_size
    EOS_ID = text_tokenizer.vocab_size + 1
    IMAGE_ID = text_tokenizer.vocab_size + 2
    
    # --- 3. मॉडल्स लोड/बनाएं ---
    print("\n[STEP 1] मॉडल्स लोड/बनाएं...")
    # VQ-VAE को लोड करें (यह प्रशिक्षित होना चाहिए)
    vqvae = VQVAE(hidden_channels=vq_hidden_channels, embedding_dim=vq_embedding_dim, num_embeddings=num_visual_tokens)
    if os.path.exists("vqvae_model.npz"):
        vqvae.load("vqvae_model.npz")
    else:
        print("WARNING: 'vqvae_model.npz' नहीं मिला। सुनिश्चित करें कि आपने पहले train_vqvae.py चलाया है।"); return
    
    # मल्टी-मोडल ट्रांसफॉर्मर बनाएं
    # वोकैबुलरी = टेक्स्ट टोकन + विजुअल टोकन + विशेष टोकन
    total_vocab_size = text_tokenizer.vocab_size + num_visual_tokens + 3
    caelonyx_transformer = CaelonyxModel(vocab_size=total_vocab_size, embed_size=embed_size, num_layers=num_layers, heads=heads, max_length=max_length)
    
    optimizer = Adam(caelonyx_transformer.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()
    print("मॉडल्स तैयार हैं।")
    
    # --- 4. ट्रेनिंग लूप ---
    print("\n[STEP 2] मल्टी-मोडल ट्रेनिंग शुरू हो रही है...")
    captions = list(dataset.keys())
    
    for epoch in range(epochs):
        caption = captions[epoch % len(captions)]
        image = dataset[caption]
        
        # 1. टेक्स्ट को टोकनाइज़ करें
        text_tokens = text_tokenizer.encode(caption)
        
        # 2. VQ-VAE एन्कोडर का उपयोग करके इमेज को टोकनाइज़ करें
        with torch.no_grad(): # हमें यहाँ ग्रेडिएंट की आवश्यकता नहीं है
            img_tensor = Tensor(image[np.newaxis, ...]) # बैच डायमेंशन जोड़ें
            z = vqvae.encoder(img_tensor)
            z = vqvae.pre_vq_conv(z)
            _, _, visual_indices = vqvae.vq_layer(z)
        visual_tokens = visual_indices.flatten()
        # विजुअल टोकन IDs को टेक्स्ट वोकैबुलरी के बाद शुरू करने के लिए ऑफसेट करें
        visual_tokens_offset = visual_tokens + text_tokenizer.vocab_size + 3

        # 3. इनपुट और टारगेट सीक्वेंस बनाएं
        # इनपुट: [SOS] टेक्स्ट [IMAGE] इमेज
        # टारगेट: टेक्स्ट [IMAGE] इमेज [EOS]
        input_seq = [SOS_ID] + text_tokens + [IMAGE_ID] + visual_tokens_offset.tolist()
        target_seq = text_tokens + [IMAGE_ID] + visual_tokens_offset.tolist() + [EOS_ID]
        
        input_tensor = Tensor(np.array([input_seq[:-1]]))
        target_np = np.array(target_seq)
        
        # 4. फॉरवर्ड और बैकवर्ड पास
        logits = caelonyx_transformer.forward(input_tensor)
        logits_reshaped = logits.reshape(-1, total_vocab_size)
        
        loss = loss_fn.forward(logits_reshaped, target_np)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Caption: '{caption}', Loss: {loss.data.item():.4f}")

    # --- 5. प्रशिक्षित ट्रांसफॉर्मर को सहेजें ---
    caelonyx_transformer.save("caelonyx_transformer.npz")
    print("\nमल्टी-मोडल ट्रांसफॉर्मर 'caelonyx_transformer.npz' में सहेजा गया।")


if __name__ == "__main__": main()
