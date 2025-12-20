# run_agent.py
import numpy as np; import os
from project_chimera.p5_agent.caelonyx_agent import CaelonyxAgent

def main():
    print("===== Caelonyx इंटरैक्टिव यूनिफाइड एजेंट =====")

    # यह कॉन्फ़िगरेशन train_agent.py के साथ मेल खाना चाहिए
    config = {
        "vq_hidden_channels": 128, "vq_embedding_dim": 64, "num_visual_tokens": 512,
        "embed_size": 256, "num_layers": 4, "heads": 4, "forward_expansion": 4,
        "max_length": 512,
        "text_vocab_size": 1000, "sos_id": 1000, "eos_id": 1001, "image_id": 1002,
        "vqvae_path": "vqvae_model.npz", 
        "transformer_path": "caelonyx_agent_transformer.npz",
        "image_token_count": 8*8,
        "max_chat_tokens": 100,
        "generation_temp": 0.8,
        "generation_top_k": 40
    }
    config["total_vocab_size"] = config["text_vocab_size"] + config["num_visual_tokens"] + 3

    agent = CaelonyxAgent(config)

    # मॉडल लोड करें और जांचें कि क्या वे मौजूद हैं
    if not agent.load_models():
        print("\nएजेंट शुरू करने में असमर्थ। कृपया पहले ट्रेनिंग स्क्रिप्ट चलाएं।")
        return

    print("\nCaelonyx तैयार है। अपना संदेश टाइप करें या 'generate image of <...>' लिखें।")
    print("बाहर निकलने के लिए 'exit' टाइप करें।")
    
    while True:
        try:
            prompt = input(">>> आप: ")
            if prompt.lower() == 'exit':
                print("Caelonyx को अलविदा!")
                break
            
            # एजेंट से प्रतिक्रिया प्राप्त करें
            agent.process_prompt(prompt)

        except Exception as e:
            print(f"एक अप्रत्याशित त्रुटि हुई: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
