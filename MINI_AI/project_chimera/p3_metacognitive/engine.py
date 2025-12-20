# project_chimera/p3_metacognitive/engine.py
import os, sys, importlib, random, shutil, filecmp
import numpy as np # <-- NEW: NumPy को इम्पोर्ट करें
try:
    import torch # <-- NEW: VRAM चेक करने के लिए Torch को इम्पोर्ट करें
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class MetacognitiveEngine:
    def __init__(self, agent_id, task, target_file, module_name):
        self.agent_id = agent_id
        self.task = task
        self.base_target_file = target_file
        self.module_name = module_name
        self.gene_pool = {}
        self.generation_counter = 0
        self.work_dir = f"project_chimera/p3_metacognitive/work_agent_{agent_id}"
        self.active_gene = "baseline"
        self.last_best_gene = None
        
        # <-- NEW: क्रैश से बचाने के लिए आखिरी 'अच्छे' जीन को याद रखें
        self.last_known_good_gene = "baseline" 

        os.makedirs(self.work_dir, exist_ok=True)
        print(f"[P3 ENGINE-{self.agent_id}] Initialized. Work dir: '{self.work_dir}'")
        self._persistent_setup()

    # --- NEW FUNCTION: "Power-Aware" Hardware Check ---
    def _get_hardware_profile(self):
        """
        चेक करता है कि कितना VRAM उपलब्ध है और उसके आधार पर एक 'टियर' (tier) लौटाता है।
        Returns: (tier_name, base_embed_size)
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[P3 Hardware Check] GPU VRAM मिली: {vram_gb:.2f} GB")
                if vram_gb > 40:
                    return "Tier_S (1000x)", 8192 # >40GB VRAM (H100/A100)
                elif vram_gb > 16:
                    return "Tier_A (100x)", 4096 # 16-40GB VRAM (RTX 4090)
                elif vram_gb > 8:
                    return "Tier_B (10x)", 1024 # 8-16GB VRAM (RTX 4070)
                else:
                    return "Tier_C (1x)", 256 # <8GB VRAM
            except Exception as e:
                print(f"[P3 Hardware Check] GPU VRAM चेक करने में विफल: {e}")
        
        print("[P3 Hardware Check] कोई GPU नहीं मिला या Torch विफल हुआ। CPU मोड (Tier_CPU) में चल रहा है।")
        return "Tier_CPU (1x)", 128 # CPU-Only

    def run_evolutionary_cycle(self, generations=1, population_size=3):
        if not self.gene_pool:
            print(f"\n--- [P3-{self.agent_id}] BASELINE EVALUATION ---")
            baseline_perf = self._evaluate_candidate("baseline")
            if baseline_perf: 
                self.gene_pool["baseline"] = baseline_perf
                self.last_known_good_gene = "baseline" # <-- NEW: बेसलाइन को 'अच्छा' मानें

        best_gene_name = min(self.gene_pool, key=lambda k: self.gene_pool[k].get('loss', float('inf')))

        for gen in range(generations):
            self.generation_counter += 1
            print(f"\n--- [P3-{self.agent_id}] EVOLUTIONARY GENERATION {self.generation_counter} ---")

            candidates = self._generate_population(population_size)
            self._create_candidate_files(candidates, base_gene_name=best_gene_name)

            for name in candidates.keys():
                perf = self._evaluate_candidate(name)
                if perf: 
                    self.gene_pool[name] = perf
                    # <-- NEW: अगर क्रैश नहीं हुआ, तो यह नया 'अच्छा' जीन है
                    if perf.get('loss', float('inf')) != float('inf'):
                        self.last_known_good_gene = name

            best_gene_name = min(self.gene_pool, key=lambda k: self.gene_pool[k].get('loss', float('inf')))
            best_performance = self.gene_pool[best_gene_name]

            print(f"[P3-{self.agent_id} SELECT] Gen-{self.generation_counter} Best: '{best_gene_name}' Loss: {best_performance.get('loss', float('inf')):.4f}")

        self.last_best_gene = {'name': best_gene_name, 'loss': self.gene_pool[best_gene_name]['loss']}
        print(f"[P3-{self.agent_id}] EVOLUTION COMPLETE. Best overall gene: '{best_gene_name}'")
        return best_gene_name

    # --- UPGRADED FUNCTION: "Crash-Proof" Resiliency ---
    def _evaluate_candidate(self, gene_name):
        print(f"--- [P3-{self.agent_id}] Evaluating gene: {gene_name} ---")
        try:
            self.set_active_gene(gene_name)
            reloaded_module = self.get_reloaded_module()
            if reloaded_module:
                # प्रयास करें कि यह 'रन' हो
                return self.task.run(reloaded_module)
            return {"loss": float('inf')} # मॉड्यूल लोड नहीं हो सका
        except Exception as e:
            # --- CRASH HANDLER ---
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"[P3 CRASH HANDLER] जीन '{gene_name}' क्रैश हो गया: {e}")
            print(f"[P3 CRASH HANDLER] यह संभवतः OutOfMemoryError था।")
            print(f"[P3 CRASH HANDLER] पिछले स्थिर जीन '{self.last_known_good_gene}' पर वापस जा रहे हैं।")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            # क्रैश हुए जीन को अनंत (infinite) लॉस दें
            self.gene_pool[gene_name] = {"loss": float('inf')}
            
            # आखिरी 'अच्छे' जीन पर वापस जाएँ
            self.set_active_gene(self.last_known_good_gene)
            return None # मूल्यांकन विफल रहा

    def set_active_gene(self, gene_name):
        candidate_path = self._get_working_path(gene_name)
        if not os.path.exists(candidate_path):
            print(f"[P3-{self.agent_id}] ERROR: Gene path does not exist: {candidate_path}")
            return
        if not os.path.exists(self.base_target_file) or not filecmp.cmp(candidate_path, self.base_target_file, shallow=False):
            shutil.copy(candidate_path, self.base_target_file)
        self.active_gene = gene_name

    def get_reloaded_module(self):
        try:
            if self.module_name in sys.modules:
                importlib.invalidate_caches()
                module = importlib.reload(sys.modules[self.module_name])
            else:
                module = importlib.import_module(self.module_name)
            return module
        except Exception as e:
            print(f"[P3 LOADER-{self.agent_id}] FATAL: Could not reload module '{self.module_name}'. Error: {e}")
            import traceback
            traceback.print_exc()
            # --- NEW: क्रैश हैंडलर को बताने के लिए एरर को फिर से थ्रो करें
            raise e

    # --- UPGRADED FUNCTION: "Power-Aware" Growth ---
    def _generate_population(self, size):
        candidates = {}
        
        # --- NEW: जाँच करें कि हम कितना पावरफुल बन सकते हैं
        tier_name, base_embed_size = self._get_hardware_profile()
        print(f"[P3 Generate] हार्डवेयर टियर '{tier_name}' का उपयोग करके जीन बनाए जा रहे हैं। बेस साइज़: {base_embed_size}")

        for i in range(size):
            layer_num = self.generation_counter
            
            # --- NEW: 'embed_size' को 'base_embed_size' से बदलें
            # यह आपके 100x/10000x के अनुरोध को पूरा करता है
            # हम बेतरतीब ढंग से 1x से 4x (टियर के भीतर) का साइज़ चुनते हैं
            new_size = base_embed_size * random.randint(1, 4)
            
            # अभी के लिए, हम केवल 'ADD_LAYER' पर ध्यान केंद्रित करते हैं
            gene_name = f"g{layer_num}_add_layer_{tier_name}_size{new_size}"
            
            candidates[gene_name] = { 
                "type": "ADD_LAYER", 
                "layer_code": f"    self.meta_layer_{layer_num}_{i} = Linear({new_size}, {new_size}) # Tier: {tier_name}", 
                "forward_code": f"    x = self.meta_layer_{layer_num}_{i}(x).relu()" 
            }
            
            # (पुराने 'REPLACE_LINE' म्यूटेशन को सरलता के लिए हटा दिया गया है)

        return candidates

    def _create_candidate_files(self, candidates, base_gene_name):
        base_path = self._get_working_path(base_gene_name)
        if not os.path.exists(base_path):
            print(f"[P3-{self.agent_id} MODIFY] Error: Base gene file not found: {base_path}")
            return
        with open(base_path, 'r') as f: base_lines = f.readlines()

        for name, mod in candidates.items():
            candidate_path = self._get_working_path(name)
            new_lines = list(base_lines)
            if mod['type'] == 'ADD_LAYER':
                layer_idx = next((i for i, l in enumerate(new_lines) if "# <META_ENGINE_HOOK_LAYERS>" in l), -1)
                forward_idx = next((i for i, l in enumerate(new_lines) if "# <META_ENGINE_HOOK_FORWARD>" in l), -1)
                
                if layer_idx != -1 and forward_idx != -1:
                    # --- NEW: साइज़ को बेसलाइन से बदलें
                    # यह सुनिश्चित करता है कि आपके 'neural_program_synthesizer.py' में 'embed_size' की परवाह किए बिना
                    # यह नया साइज़ इस्तेमाल किया जाएगा।
                    
                    # 'embed_size' या 'hidden_size' को नई लेयर के साइज़ से बदलें
                    mod['layer_code'] = mod['layer_code'].replace("embed_size", str(random.randint(256, 1024))) # एक डिफ़ॉल्ट साइज़
                    mod['forward_code'] = mod['forward_code'].replace("embed_size", str(random.randint(256, 1024)))

                    new_lines.insert(layer_idx + 1, mod['layer_code'] + '\n')
                    new_lines.insert(forward_idx + 1, mod['forward_code'] + '\n')
            
            with open(candidate_path, 'w') as f: f.writelines(new_lines)

    def _persistent_setup(self):
        baseline_path = self._get_working_path("baseline")
        if not os.path.exists(baseline_path):
            # सुनिश्चित करें कि बेस फ़ाइल मौजूद है
            if os.path.exists(self.base_target_file):
                shutil.copy(self.base_target_file, baseline_path)
            else:
                print(f"WARNING: बेस फ़ाइल {self.base_target_file} नहीं मिली!")

    def _get_working_path(self, gene_name):
        safe_name = "".join(c for c in gene_name if c.isalnum() or c in ('_', '-')).rstrip()
        return os.path.join(self.work_dir, f"{safe_name}.py")
