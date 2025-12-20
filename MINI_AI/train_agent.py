# train_agent.py (ULTIMATE version with multiple data format support)

import numpy as np
import os
import time
import argparse
import json # JSONL पढ़ने के लिए इम्पोर्ट करें

from project_chimera.p5_agent.caelonyx_agent import CaelonyxAgent
from project_chimera.l1_calculus.tensor import Tensor
from project_chimera.l1_calculus.optimizers import Adam
from project_chimera.nn.losses import CrossEntropyLoss
try:
  import torch
except ImportError:
  print("PyTorch/Torchvision की आवश्यकता है।"); exit()

# --- NEW: उन्नत DataLoader ---
class UniversalDataLoader:
  def __init__(self, file_path, tokenizer, data_type='auto', text_key='text', batch_size=16, seq_len=64):
    self.batch_size = batch_size
    self.seq_len = seq_len

    # स्वचालित रूप से डेटा प्रकार का पता लगाएं
    if data_type == 'auto':
      if file_path.endswith('.jsonl'):
        data_type = 'jsonl'
      elif file_path.endswith('.txt'):
        data_type = 'text'
      else:
        raise ValueError(f"असमर्थित फ़ाइल प्रकार: {file_path}. कृपया .txt या .jsonl का उपयोग करें।")

    print(f"[DataLoader] '{file_path}' से '{data_type}' प्रारूप में डेटा लोड किया जा रहा है...")
    corpus = []
    if data_type == 'text':
      with open(file_path, 'r', encoding='utf-8') as f:
        corpus = f.read().split('\n')
    elif data_type == 'jsonl':
      with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
          try:
            record = json.loads(line)
            if text_key in record:
              corpus.append(record[text_key])
            else:
              print(f"चेतावनी: JSON लाइन में '{text_key}' की नहीं मिली: {line.strip()}")
          except json.JSONDecodeError:
            print(f"चेतावनी: JSON लाइन को पार्स करने में विफल: {line.strip()}")

    if not corpus:
      raise ValueError(f"डेटा फ़ाइल '{file_path}' से कोई टेक्स्ट लोड नहीं हुआ। कृपया अपनी फ़ाइल और --text_key जांचें।")

    # टोकनाइज़र को लोड किए गए कॉर्पस पर प्रशिक्षित करें
    tokenizer.train(corpus)
    full_text = " ".join(corpus)
    self.tokens = tokenizer.encode(full_text)
    print(f"[DataLoader] डेटा लोड और टोकनाइज़ किया गया। कुल टोकन: {len(self.tokens)}")

  def get_batch(self):
    if len(self.tokens) < self.seq_len + self.batch_size:
      raise ValueError("टोकन की संख्या बैच बनाने के लिए बहुत कम है। कृपया एक बड़ा डेटासेट प्रदान करें।")
    start_indices = np.random.randint(0, len(self.tokens) - self.seq_len - 1, self.batch_size)
    input_seqs = [self.tokens[i : i + self.seq_len] for i in start_indices]
    target_seqs = [self.tokens[i + 1 : i + self.seq_len + 1] for i in start_indices]
    return Tensor(np.array(input_seqs, dtype=np.int32)), np.array(target_seqs, dtype=np.int32).flatten()


def main():
  # --- NEW: अधिक कमांड-लाइन आर्ग्युमेंट्स ---
  parser = argparse.ArgumentParser(description="Caelonyx यूनिफाइड एजेंट को लचीले डेटा इनपुट के साथ प्रशिक्षित करें।")
  parser.add_argument('--data_path', type=str, required=True, help='प्रशिक्षण डेटा फ़ाइल का पाथ (.txt या .jsonl)।')
  parser.add_argument('--data_type', type=str, default='auto', choices=['auto', 'text', 'jsonl'], help='डेटा फ़ाइल का प्रकार।')
  parser.add_argument('--text_key', type=str, default='text', help='JSONL फ़ाइल में टेक्स्ट वाले फ़ील्ड की की (key)।')
  parser.add_argument('--epochs', type=int, default=2000, help='प्रशिक्षण एपोक्स की संख्या।')
  parser.add_argument('--lr', type=float, default=3e-4, help='ऑप्टिमाइज़र के लिए लर्निंग रेट।')
  parser.add_argument('--save_path', type=str, default='caelonyx_agent_transformer.npz', help='प्रशिक्षित ट्रांसफॉर्मर मॉडल को सहेजने का पाथ।')
  parser.add_argument('--batch_size', type=int, default=16, help='ट्रेनिंग के लिए बैच साइज़।')
  args = parser.parse_args()

  print("===== Caelonyx यूनिफाइड एजेंट ट्रेनिंग (अल्टीमेट) =====")
  print(f"आर्ग्युमेंट्स: {vars(args)}")

  config = {
    "vq_hidden_channels": 128, "vq_embedding_dim": 64, "num_visual_tokens": 512,
    "embed_size": 256, "num_layers": 4, "heads": 4, "forward_expansion": 4,
    "max_length": 512, "text_vocab_size": 2000, # बड़ा शब्दकोश
    "sos_id": 2000, "eos_id": 2001, "image_id": 2002,
    "vqvae_path": "vqvae_model.npz",
    "transformer_path": args.save_path,
    "image_token_count": 8*8
  }
  config["total_vocab_size"] = config["text_vocab_size"] + config["num_visual_tokens"] + 3

  # --- एजेंट और डेटा लोडर को इनिशियलाइज़ करें ---
  agent = CaelonyxAgent(config)
  if not os.path.exists(config['vqvae_path']):
    print(f"त्रुटि: '{config['vqvae_path']}' नहीं मिला। कृपया पहले 'train_vqvae.py' चलाएं।"); return

  agent.vqvae.load(config['vqvae_path'])

  try:
    data_loader = UniversalDataLoader(args.data_path, agent.tokenizer, data_type=args.data_type, text_key=args.text_key, batch_size=args.batch_size)
  except (FileNotFoundError, ValueError) as e:
    print(f"त्रुटि: डेटा लोड करने में विफल: {e}")
    return

  optimizer = Adam(agent.transformer.parameters(), lr=args.lr)
  loss_fn = CrossEntropyLoss()

  print("\n[AGENT TRAINING] मल्टी-मोडल ट्रेनिंग शुरू हो रही है...")
  start_time = time.time()
  for epoch in range(args.epochs):
    # अभी के लिए, हम केवल टेक्स्ट डेटा पर प्रशिक्षित करेंगे क्योंकि यह अधिक जटिल है
    optimizer.zero_grad()

    input_tensor, target_np = data_loader.get_batch()
    logits = agent.transformer.forward(input_tensor)
    logits_reshaped = logits.reshape(-1, config['total_vocab_size'])
    total_loss = loss_fn.forward(logits_reshaped, target_np)

    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
      elapsed = time.time() - start_time
      print(f"E[{epoch+1}/{args.epochs}], Loss: {total_loss.data.item():.4f}, Time/100: {elapsed:.2f}s")
      start_time = time.time()

  agent.transformer.save(config['transformer_path'])
  print(f"\nप्रशिक्षण पूरा हुआ। एजेंट का ट्रांसफॉर्मर '{config['transformer_path']}' में सहेजा गया।")

if __name__ == "__main__":
  main()
