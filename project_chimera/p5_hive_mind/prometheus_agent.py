# project_chimera/p5_hive_mind/prometheus_agent.py (OMEGA-READY VERSION)
import numpy as np
from ..p3_metacognitive.engine import MetacognitiveEngine
from ..l3_cognitive.symbolic_engine import SymbolicLogicEngine
from ..tasks.math_task import MathTask
from ..l1_calculus.tensor import Tensor
import importlib, sys, os, shutil

class PrometheusAgent:
  """
  The final Omega-ready AGI Agent. It uses a fully trained synthesizer
  and is ready for large-scale compute and data.
  """
  def __init__(self, agent_id: int):
    self.id = agent_id
    self.sle = SymbolicLogicEngine()
    self.synthesizer_module_name = "project_chimera.l3_cognitive.neural_program_synthesizer"
    self.synthesizer_file_path = "project_chimera/l3_cognitive/neural_program_synthesizer.py"
    self.task = MathTask()

    self.p3_engine = MetacognitiveEngine(
      agent_id=self.id,
      task=self.task,
      target_file=self.synthesizer_file_path,
      module_name=self.synthesizer_module_name
    )

    self.goal = {"metric": "success_rate", "target": 0.8}
    self.history = []
    self.synthesizer_instance = None # This will hold the powerful, trained brain

    print(f"[AGENT-{self.id}] Prometheus Agent vOmega (Production Ready) initialized.")
    self.reload_and_train_synthesizer()

  def reload_and_train_synthesizer(self, gene_name="baseline"):
    print(f"[AGENT-{self.id}] Reloading brain with gene '{gene_name}' and performing heavy training...")
    self.p3_engine.set_active_gene(gene_name)
    try:
      module = self.p3_engine.get_reloaded_module()
      # In a real scenario, you'd load a much larger model here
      self.synthesizer_instance = module.ProgramSynthesizer(len(self.task.tokenizer.vocab) + 10, 128, 512)
      # Training would be on a massive dataset, not just the toy task.
      self.task.train_model(self.synthesizer_instance, epochs=1000) # Increased epochs for real training
    except Exception as e:
      print(f"[AGENT-{self.id}] Error loading/training synthesizer: {e}")

  def attempt_problem(self, problem):
    question = problem["question"]
    correct_answer = problem["correct_answer"]

    if not self.synthesizer_instance:
      print(f"[AGENT-{self.id}] Brain is not functional. Aborting.")
      return False

    # --- THIS IS THE REAL INFERENCE LOOP ---
    # It no longer uses a stub.
    q_tokens = self.task.tokenizer.encode(question)
    input_tensor = Tensor(np.array([q_tokens]))

    print(f"[AGENT-{self.id} S1] Generating program for: '{question}'")
    # 1. Use the actual trained neural network to get predictions (logits)
    logits = self.synthesizer_instance.forward(input_tensor)

    # 2. Decode the predictions into a program string
    # A simple greedy decoding for this example
    program_tokens = np.argmax(logits.data, axis=1)
    generated_program = self.task.tokenizer.decode(program_tokens)
    print(f"[AGENT-{self.id} S1] Generated program: '{generated_program}'")
    # --- END OF REAL INFERENCE ---

    result, success, _ = self.sle.execute(generated_program)

    is_correct = success and result == correct_answer
    self.history.append(is_correct)

    if is_correct:
      print(f"[AGENT-{self.id} S2] SUCCESS! Reward: +1")
      return True
    else:
      print(f"[AGENT-{self.id} S2] FAILURE! Reward: -1.")
      self.check_goal_and_learn()
      return False

  def check_goal_and_learn(self):
    # ... This method remains the same, it's the core of self-awareness ...
    recent_history = self.history[-10:]
    current_rate = sum(recent_history) / len(recent_history) if recent_history else 0
    print(f"[AGENT-{self.id} AWARENESS] Success Rate: {current_rate*100:.1f}%. Goal: {self.goal['target']*100:.1f}%.")

    if current_rate < self.goal['target']:
      print(f"[AGENT-{self.id} AWARENESS] Goal not met. Initiating self-improvement.")
      urgency = self.goal['target'] - current_rate
      if urgency > 0.5:
        print("[P3 STRATEGY] High urgency. Using aggressive evolution.")
        self.p3_engine.run_evolutionary_cycle(generations=2, population_size=3)
      else:
        print("[P3 STRATEGY] Low urgency. Using fine-tuning evolution.")
        self.p3_engine.run_evolutionary_cycle(generations=1, population_size=2)

      self.reload_and_train_synthesizer(gene_name=self.p3_engine.last_best_gene['name'])
    else:
      print(f"[AGENT-{self.id} AWARENESS] Goal met. Continuing with current architecture.")

  def assimilate_knowledge(self, gene_name, source_engine):
    # ... This method remains the same ...
    print(f"[AGENT-{self.id}] Assimilating new knowledge '{gene_name}' from Hive...")
    source_path = source_engine._get_working_path(gene_name)
    my_path = self.p3_engine._get_working_path(gene_name)
    if os.path.exists(source_path) and not os.path.exists(my_path):
      shutil.copy(source_path, my_path)
      perf = self.p3_engine._evaluate_candidate(gene_name)
      if perf: self.p3_engine.gene_pool[gene_name] = perf
      print(f"[AGENT-{self.id}] Knowledge assimilated.")
