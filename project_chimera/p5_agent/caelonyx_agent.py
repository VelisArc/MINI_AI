# project_chimera/p5_agent/caelonyx_agent.py
import numpy as np; import os
from ..cognitive_models.caelonyx_model import CaelonyxModel
from ..cognitive_models.vq_vae import VQVAE
from ..l1_calculus.tensor import Tensor
from ..l1_calculus.optimizers import Adam
from ..nn.losses import CrossEntropyLoss
from ..l2_data.unigram_tokenizer import UnigramTokenizer
try:
 import torch; from torchvision.utils import save_image
except ImportError: print("PyTorch/Torchvision की आवश्यकता है।"); exit()

class CaelonyxAgent:
 def __init__(self, config):
  self.config = config
  self.is_ready = False # NEW: एक फ्लैग यह ट्रैक करने के लिए कि क्या मॉडल लोड हुए हैं
  # Optimizer for real-time learning (initialized later)
  self.optimizer = None
  self.loss_fn = CrossEntropyLoss()

  self.vqvae = VQVAE(
   in_channels=3,
   hidden_channels=config['vq_hidden_channels'],
   embedding_dim=config['vq_embedding_dim'],
   num_embeddings=config['num_visual_tokens']
  )
  self.transformer = CaelonyxModel(
   vocab_size=config['total_vocab_size'],
   embed_size=config['embed_size'],
   num_layers=config['num_layers'],
   heads=config['heads'],
   forward_expansion=config['forward_expansion'],
   max_length=config['max_length']
  )
  self.tokenizer = UnigramTokenizer(vocab_size=config['text_vocab_size'])

 def load_models(self):
  print("[Agent] प्रशिक्षित मॉडल्स को लोड किया जा रहा है...")
  try:
   self.vqvae.load(self.config['vqvae_path'])
   self.transformer.load(self.config['transformer_path'])

   # Initialize optimizer with loaded parameters
   self.optimizer = Adam(self.transformer.parameters(), lr=1e-4) # Lower LR for fine-tuning

   # एक सरल कॉर्पस पर हमेशा टोकनाइज़र को प्रशिक्षित करें
   self.tokenizer.train(["hello world", "a red square", "what is ai", "generate image of a blue circle"])
   print("[Agent] मॉडल्स सफलतापूर्वक लोड हो गए।")
   self.is_ready = True
   return True
  except Exception as e:
   # --- THE FIX IS HERE ---
   print(f"[Agent] त्रुटि: प्रशिक्षित मॉडल लोड करने में विफल: {e}")
   print("[Agent] कृपया सुनिश्चित करें कि 'train_vqvae.py' और 'train_agent.py' सफलतापूर्वक चल चुके हैं।")
   self.is_ready = False
   return False # --- यह बहुत महत्वपूर्ण है ---

 def process_prompt(self, prompt: str):
  if not self.is_ready:
   print(">>> Caelonyx: मैं अभी तैयार नहीं हूँ क्योंकि मेरे मॉडल लोड नहीं हुए हैं। कृपया पहले मुझे प्रशिक्षित करें।")
   return None

  if prompt.lower().startswith("generate image of"):
   self.generate_image(prompt[len("generate image of"):].strip())
   return None
  else:
   return self.chat(prompt)

 def learn_from_interaction(self, user_input, response_text, reward):
    """
    Performs a single training step based on the interaction.
    If reward is positive, we reinforce the (input -> response) pair.
    """
    if reward <= 0: return # Only learn from positive feedback for now

    print("[Agent] Learning from this interaction...")

    # Prepare data
    full_text = user_input + " " + response_text
    tokens = self.tokenizer.encode(full_text)
    # Simple autoregressive task: predict next token
    # input: tokens[:-1], target: tokens[1:]

    if len(tokens) < 2: return

    input_seq = tokens[:-1]
    target_seq = tokens[1:]

    input_tensor = Tensor(np.array([input_seq], dtype=np.int32))
    target_np = np.array(target_seq, dtype=np.int32)

    # Training step
    self.optimizer.zero_grad()
    logits = self.transformer.forward(input_tensor)
    logits_reshaped = logits.reshape(-1, self.config['total_vocab_size'])
    loss = self.loss_fn.forward(logits_reshaped, target_np)
    loss.backward()
    self.optimizer.step()

    print(f"[Agent] Updated weights. Loss: {loss.data.item():.4f}")

 def chat(self, user_input: str):
  print(f">>> आप: {user_input}")
  SOS_ID, EOS_ID = self.config['sos_id'], self.config['eos_id']
  text_tokens = self.tokenizer.encode(user_input)
  input_seq = [SOS_ID] + text_tokens

  response_tokens = []
  for _ in range(self.config['max_chat_tokens']):
   input_tensor = Tensor(np.array([input_seq]))
   logits = self.transformer.forward(input_tensor)
   last_logits = logits.data[0, -1, :]

   top_k = self.config.get('generation_top_k', 40)
   temp = self.config.get('generation_temp', 0.8)

   last_logits = last_logits / temp
   top_indices = np.argsort(last_logits)[-top_k:]
   top_logits = last_logits[top_indices]
   probabilities = np.exp(top_logits) / np.sum(np.exp(top_logits))
   predicted_id = np.random.choice(top_indices, p=probabilities)

   if predicted_id == EOS_ID or predicted_id >= self.config['text_vocab_size']:
    break

   input_seq.append(int(predicted_id))
   response_tokens.append(int(predicted_id))

  response_text = self.tokenizer.decode(response_tokens)
  print(f">>> Caelonyx: {response_text}")
  return response_text

 def generate_image(self, prompt: str):
  print(f"[Agent] इमेज जेनरेट की जा रही है: '{prompt}'")
  SOS_ID, IMAGE_ID = self.config['sos_id'], self.config['image_id']
  num_visual_tokens, text_vocab_size = self.config['num_visual_tokens'], self.config['text_vocab_size']
  text_tokens = self.tokenizer.encode(prompt)
  input_seq = [SOS_ID] + text_tokens + [IMAGE_ID]
  generated_visual_tokens = []
  for _ in range(self.config['image_token_count']):
   input_tensor = Tensor(np.array([input_seq]))
   logits = self.transformer.forward(input_tensor)
   last_logits = logits.data[0, -1, :]
   visual_token_start = self.config['total_vocab_size'] - num_visual_tokens
   visual_token_logits = last_logits[visual_token_start:]
   predicted_offset_id = np.argmax(visual_token_logits)
   predicted_id = predicted_offset_id + visual_token_start
   input_seq.append(predicted_id)
   generated_visual_tokens.append(predicted_offset_id)

  generated_tokens_np = np.array(generated_visual_tokens)
  quantized_vectors = self.vqvae.vq_layer.embedding.weight.data[generated_visual_tokens]
  h = w = int(np.sqrt(self.config['image_token_count']))
  quantized_grid = quantized_vectors.reshape(1, h, w, self.config['vq_embedding_dim']).transpose(0, 3, 1, 2)
  with torch.no_grad(): image_tensor = self.vqvae.decoder(Tensor(quantized_grid))
  os.makedirs("agent_generations", exist_ok=True)
  filename = f"agent_generations/agent_generated_{prompt.replace(' ', '_')}.png"
  save_image(torch.from_numpy(image_tensor.data), filename, normalize=True)
  print(f"सफलता! इमेज '{filename}' में सहेजी गई है।")
