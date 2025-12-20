# 🚀 Project Chimera

**A Self-Evolving, Multi-Modal AGI System with Hive Mind Architecture**

> "Weakness-Free AI through Metacognitive Evolution and Distributed Intelligence"

---

## 🎯 Kya Hai Yeh Project?

Caelonyx ek **revolutionary AI system** hai jo:

1. **खुद को improve करता है** - P3 Metacognitive Engine अपने neural architecture को runtime पर modify करता है
2. **Text और Images दोनों समझता है** - Multi-modal VQ-VAE + Transformer
3. **Multiple agents collaborate करते हैं** - Hive Mind में knowledge sharing
4. **CPU और GPU दोनों पर efficiently चलता है** - Hardware Abstraction Layer (HAL)
5. **Symbolic + Neural reasoning** - System 1 (fast neural) + System 2 (slow symbolic)

---

## 📁 Project Structure

```
Caelonyx/
├── project_chimera/
│   ├── l0_hal/                    # Hardware Abstraction (CPU/GPU)
│   ├── l1_calculus/               # Custom Autograd Engine
│   ├── l2_data/                   # Tokenizers & Data Processing
│   ├── l2_storage/                # Vector Search (HNSW)
│   ├── l3_cognitive/              # Neural + Symbolic Engines
│   ├── l4_distribution/           # Multi-GPU Training
│   ├── nn/                        # Neural Network Layers
│   ├── p3_metacognitive/          # Self-Modifying Code Engine
│   ├── p4_environment/            # Dynamic Problem Generator
│   ├── p5_agent/                  # Unified Agent
│   ├── p5_hive_mind/              # Multi-Agent Coordination
│   ├── cognitive_models/          # VQ-VAE & Transformer
│   └── tasks/                     # Training Tasks
├── datasets/                      # Training Data
├── train_vqvae.py                 # Image Model Training
├── train_agent.py                 # Agent Training
├── run_agent.py                   # Interactive Agent
└── run.py                         # Hive Mind Simulation
```

---

## 🛠️ Installation

### Method 1: Quick Start (Recommended)

```bash
chmod +x quick_start.sh
./quick_start.sh
```

### Method 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 2. Install dependencies
# For GPU:
pip install -r requirements-gpu.txt
export PROMETHEUS_USE_GPU=true

# For CPU:
pip install -r requirements-cpu.txt
export PROMETHEUS_USE_GPU=false

# 3. Create directories
mkdir -p datasets vqvae_results agent_generations
```

---

## 🚀 Quick Start Guide

### Step 1: Train VQ-VAE (Image Understanding)

```bash
python3 train_vqvae.py
```

**Output:**
- `vqvae_model.npz` - Trained image encoder/decoder
- `vqvae_results/` - Reconstruction samples

**Time:** ~20 epochs × 2-3 min = 40-60 minutes (CPU)

---

### Step 2: Train Unified Agent (Multi-Modal)

```bash
# Create sample dataset (if not exists)
echo "hello world
the quick brown fox
machine learning is cool" > datasets/train.txt

# Train the agent
python3 train_agent.py \
    --data_path datasets/train.txt \
    --epochs 100 \
    --lr 3e-4 \
    --batch_size 16
```

**Output:**
- `caelonyx_agent_transformer.npz` - Trained transformer

**Time:** ~100 epochs × 1-2 sec = 2-3 minutes (CPU)

---

### Step 3: Run Interactive Agent

```bash
python3 run_agent.py
```

**Usage:**
```
>>> You: hello
>>> Caelonyx: [generates response]

>>> You: generate image of a red square
>>> Caelonyx: [creates image in agent_generations/]

>>> You: exit
```

---

### Step 4: Run Hive Mind (Advanced)

```bash
python3 run.py
```

**Kya Hota Hai:**
- 2 Prometheus Agents spawn होते हैं
- Dynamic math problems solve करते हैं
- P3 Engine अपना code modify करता है
- Agents knowledge share करते हैं

---

## 🧠 Architecture Details

### L0: Hardware Abstraction Layer (HAL)
- **Purpose:** CPU/GPU को transparently handle करता है
- **Key Feature:** NumPy ↔ CuPy auto-switching
- **File:** `l0_hal/hardware_abstraction.py`

### L1: Custom Autograd Engine
- **Purpose:** PyTorch-style automatic differentiation
- **Key Feature:** Full computational graph tracking
- **Files:** `l1_calculus/tensor.py`, `ops.py`

### L2: Data & Storage
- **Tokenizer:** Unigram Language Model (MDL-optimized)
- **Vector Search:** HNSW (Faiss-powered)
- **Files:** `l2_data/unigram_tokenizer.py`, `l2_storage/hsvi.py`

### L3: Cognitive Engines
- **Neural (System 1):** ProgramSynthesizer - fast, intuitive
- **Symbolic (System 2):** Logic Engine - slow, verifiable
- **Files:** `l3_cognitive/neural_program_synthesizer.py`, `symbolic_engine.py`

### P3: Metacognitive Engine (THE MAGIC!)
- **Purpose:** Self-modifying code evolution
- **How:** Generates gene pool → Evaluates → Best survives
- **Key Feature:** Runtime architecture mutation
- **File:** `p3_metacognitive/engine.py`

**Example Mutation:**
```python
# Before Evolution
class ProgramSynthesizer:
    def __init__(self, vocab_size, embed_size, hidden_size):
        self.encoder = Linear(embed_size, hidden_size)

# After P3 Evolution (Generation 5)
class ProgramSynthesizer:
    def __init__(self, vocab_size, embed_size, hidden_size):
        self.encoder = Linear(embed_size, hidden_size)
        self.meta_layer_5_0 = Linear(2048, 2048)  # <-- ADDED BY P3!
        self.meta_layer_5_1 = Linear(4096, 4096)  # <-- ADDED BY P3!
```

### P5: Hive Mind
- **Purpose:** Multi-agent collaborative learning
- **How:** Agents share best "genes" (architectures)
- **Files:** `p5_hive_mind/hive.py`, `prometheus_agent.py`

---

## 🔬 Advanced Usage

### Custom Dataset Training

```bash
# Text-only dataset
python3 train_agent.py \
    --data_path my_corpus.txt \
    --data_type text \
    --epochs 500

# JSONL dataset
python3 train_agent.py \
    --data_path data.jsonl \
    --data_type jsonl \
    --text_key "content" \
    --epochs 500
```

### Multi-GPU Training

```bash
export PROMETHEUS_USE_GPU=true
# Auto-detects GPUs and uses torchrun
python3 train_vqvae.py  # Will use all available GPUs
```

### Hardware Profiling

```python
from project_chimera.l0_hal.hardware_abstraction import HAL

print(f"Device: {HAL.device}")
print(f"GPU Count: {HAL.get_gpu_count()}")
print(f"CuPy Available: {HAL.CUPY_AVAILABLE}")
```

---

## 📊 Expected Performance

### VQ-VAE (CIFAR-10)
- **Training Time:** 20 epochs = ~1 hour (GPU) / ~6 hours (CPU)
- **Final Loss:** ~0.15 (reconstruction) + ~0.05 (VQ) = 0.20
- **Image Quality:** Blurry but recognizable reconstructions

### Unified Agent (Small Corpus)
- **Training Time:** 100 epochs = ~5 minutes (CPU)
- **Loss:** 2.0-3.0 (CrossEntropy)
- **Text Generation:** Semi-coherent short sentences
- **Image Generation:** Abstract patterns (not realistic yet)

### Hive Mind (Math Task)
- **Success Rate:** 20-30% initially → 60-80% after evolution
- **Evolution Cycles:** 5-10 generations
- **Time:** ~10 minutes for full simulation

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'project_chimera'"
**Solution:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "CUDA out of memory"
**Solution:**
```bash
# Reduce batch size
python3 train_vqvae.py  # Edit batch_size to 16 or 8
python3 train_agent.py --batch_size 4
```

### Issue: "Models not found" in run_agent.py
**Solution:**
```bash
# Train models first
python3 train_vqvae.py
python3 train_agent.py --data_path datasets/train.txt
```

### Issue: P3 Engine crashes with "Loss: inf"
**Reason:** Generated architecture is too large for available RAM/VRAM

**Solution:** P3 Engine automatically rolls back to last known good gene.

---

## 🎓 Key Concepts

### What is "Metacognitive Evolution"?
The system doesn't just learn from data - it learns **how to learn**. The P3 Engine modifies its own neural architecture based on performance feedback.

### What is "Hive Mind"?
Multiple agents work on different problems but share their best solutions (architectures). Like a swarm of scientists publishing papers.

### What is "Weakness-Free AI"?
By combining:
1. Neural networks (fast, flexible)
2. Symbolic reasoning (verifiable, safe)
3. Self-modification (adaptive)
4. Distributed intelligence (robust)

We aim to create an AI system with minimal failure modes.

---

## 📚 Further Reading

1. **VQ-VAE:** "Neural Discrete Representation Learning" (van den Oord et al., 2017)
2. **Transformers:** "Attention Is All You Need" (Vaswani et al., 2017)
3. **Metacognition:** "Thinking About Thinking" (Flavell, 1979)
4. **Program Synthesis:** "Neural Program Synthesis" (Devlin et al., 2017)

---

## 🤝 Contributing

Is project mein contribute karne ke liye:

1. Fork the repo
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Test: `python3 run_all_tests.py`
5. Submit a Pull Request

---

## 📝 License

[License MIT]

---

## 🙏 Acknowledgments

Built with ❤️ using:
- NumPy/CuPy for tensor operations
- PyTorch for Conv2D kernels
- Faiss for vector search
- Pure Python for everything else!

---

## 📧 Contact

[velisarcofficial@gmail.com]

---

**Remember:** "The best AI is the one that improves itself." - Project Chimera Motto
