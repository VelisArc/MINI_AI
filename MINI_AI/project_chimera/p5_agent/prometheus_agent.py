# GOAL-DIRECTED FINAL VERSION
import numpy as np
from project_chimera.p3_metacognitive.engine import MetacognitiveEngine
from project_chimera.l3_cognitive.symbolic_engine import SymbolicLogicEngine
from project_chimera.main_metacognitive_loop import MathTask
from project_chimera.l2_storage.hsvi import HNSWIndex
from project_chimera.l1_calculus.tensor import Tensor
import importlib, sys, time

class PrometheusAgent:
    def __init__(self):
        self.sle = SymbolicLogicEngine()
        self.synthesizer_module_name = "project_chimera.l3_cognitive.neural_program_synthesizer"
        self.synthesizer_file_path = "project_chimera/l3_cognitive/neural_program_synthesizer.py"
        self.task = MathTask()
        self.p3_engine = MetacognitiveEngine(task=self.task, target_file=self.synthesizer_file_path, module_name=self.synthesizer_module_name)
        
        self.embedding_dim = 32
        self.memory = HNSWIndex(dim=self.embedding_dim)
        self.memory_store = {}

        # --- NEW: GOAL and SELF-AWARENESS METRICS ---
        self.goal = {"metric": "success_rate", "target": 0.8}
        self.history = [] # To track success/failure over time
        self.synthesizer_instance = None # To hold the actual trained model

        print("[P5 AGENT] Prometheus Agent v3 (Goal-Directed) initialized.")
        self.reload_and_train_synthesizer()

    def reload_and_train_synthesizer(self):
        """Loads the synthesizer module and trains a fresh instance of it."""
        print("[AGENT] Reloading and training synthesizer brain...")
        try:
            if self.synthesizer_module_name in sys.modules: module = importlib.reload(sys.modules[self.synthesizer_module_name])
            else: module = importlib.import_module(self.synthesizer_module_name)
            self.synthesizer_instance = module.ProgramSynthesizer(len(self.task.tokenizer.vocab) + 10, self.embedding_dim, 32)
            # Train the new brain on some basic examples
            self.task.train_model(self.synthesizer_instance, epochs=100)
        except Exception as e:
            print(f"[AGENT] Error loading/training synthesizer: {e}")
            self.synthesizer_instance = None

    def attempt_problem(self, problem):
        # ... (Memory retrieval part is the same) ...
        # --- THIS IS THE REAL TRAINING/INFERENCE LOOP ---
        question = problem["question"]
        correct_answer = problem["correct_answer"]
        question_tokens = self.task.tokenizer.encode(question)
        input_tensor = Tensor(np.array([question_tokens]))
        
        # 1. System 1 (Neural): Generate a program using the *trained* model
        print(f"[AGENT S1] Agent attempts to generate program for: '{question}'")
        logits = self.synthesizer_instance.forward(input_tensor)
        # For simplicity, we decode greedily from the final logits
        program_tokens = np.argmax(logits.data, axis=1)
        generated_program = self.task.tokenizer.decode(program_tokens)
        print(f"[AGENT S1] Generated program: '{generated_program}'")

        # 2. System 2 (Symbolic): Execute and verify
        result, success, _ = self.sle.execute(generated_program)
        
        # 3. Update self-awareness metrics and memory
        is_correct = success and result == correct_answer
        self.history.append(is_correct)
        episode = {"question": question, "program": generated_program, "result": result, "success": is_correct}
        # self.store_episode_in_memory(...) # Memory part can be added back later

        if is_correct:
            print(f"[AGENT S2] SUCCESS! Correct answer found. Reward: +1")
            return True
        else:
            print(f"[AGENT S2] FAILURE! Program was wrong or invalid. Reward: -1.")
            self.check_goal_and_learn()
            return False

    def check_goal_and_learn(self):
        """Checks if the agent is meeting its goal and triggers learning if not."""
        # Calculate current success rate (last 10 attempts)
        recent_history = self.history[-10:]
        current_rate = sum(recent_history) / len(recent_history) if recent_history else 0
        print(f"[AGENT AWARENESS] Current Success Rate: {current_rate*100:.1f}%. Goal: {self.goal['target']*100:.1f}%.")

        if current_rate < self.goal['target']:
            print("[AGENT AWARENESS] Goal not met. Initiating self-improvement.")
            # --- GOAL-DIRECTED LEARNING ---
            # Tell the P3 engine how aggressively to learn
            urgency = self.goal['target'] - current_rate
            if urgency > 0.5: # Very far from goal
                print("[P3 STRATEGY] High urgency. Using aggressive evolution.")
                self.p3_engine.run_evolutionary_cycle(generations=2, population_size=3)
            else: # Closer to goal
                print("[P3 STRATEGY] Low urgency. Using fine-tuning evolution.")
                self.p3_engine.run_evolutionary_cycle(generations=1, population_size=2)
            
            # After evolution, we must get a new brain
            self.reload_and_train_synthesizer()
        else:
            print("[AGENT AWARENESS] Goal met. Continuing with current architecture.")
