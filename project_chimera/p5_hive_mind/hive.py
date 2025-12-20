# project_chimera/p5_hive_mind/hive.py
from .prometheus_agent import PrometheusAgent
from ..p4_environment.dynamic_math_env import DynamicMathEnvironment
import time

class Hive:
  """
  Manages a collective of Prometheus Agents.
  It provides problems and facilitates knowledge sharing.
  """
  def __init__(self, num_agents: int):
    print(f"[HIVE] Initializing Hive Mind with {num_agents} agents.")
    self.environment = DynamicMathEnvironment()
    self.agents = [PrometheusAgent(agent_id=i) for i in range(num_agents)]
    self.shared_knowledge = {} # Best genes discovered by any agent

  def run_lifecycle(self, num_cycles: int):
    for i in range(num_cycles):
      print(f"\n{'#'*25} HIVE MIND LIFECYCLE STEP {i+1} {'#'*25}")
      problem = self.environment.get_new_problem()
      if not problem: continue

      # Assign the problem to a random agent
      agent_to_task = self.agents[i % len(self.agents)]
      print(f"[HIVE] Assigning problem to Agent-{agent_to_task.id}...")

      success = agent_to_task.attempt_problem(problem)

      # Knowledge Sharing
      if agent_to_task.p3_engine.last_best_gene:
        gene_name = agent_to_task.p3_engine.last_best_gene['name']
        gene_loss = agent_to_task.p3_engine.last_best_gene['loss']

        if gene_name not in self.shared_knowledge or gene_loss < self.shared_knowledge[gene_name]:
          print(f"[HIVE] Agent-{agent_to_task.id} discovered a new superior gene: '{gene_name}'! Sharing with the Hive.")
          self.shared_knowledge[gene_name] = gene_loss
          # Broadcast this new knowledge to other agents
          for agent in self.agents:
            if agent.id != agent_to_task.id:
              agent.assimilate_knowledge(gene_name, agent_to_task.p3_engine)
      time.sleep(1)
