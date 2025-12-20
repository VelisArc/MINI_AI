# FINAL OMEGA BLUEPRINT - RUNNER
import sys
import os
import time

def run_hive_mind_loop():
    """Main entry point for the Project Omega Hive Mind simulation."""
    # Ensure all necessary packages can be found
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Create necessary directories if they don't exist
    for new_dir in ["p4_environment", "p5_hive_mind", "cognitive_models"]:
        path = os.path.join("project_chimera", new_dir)
        os.makedirs(path, exist_ok=True)
        init_path = os.path.join(path, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f: pass

    from project_chimera.p5_hive_mind.hive import Hive

    print("\n--- PROJECT OMEGA: ACTIVATING HIVE MIND ---")
    
    # The Hive will manage the agents and their lifecycle
    hive = Hive(num_agents=2)
    hive.run_lifecycle(num_cycles=5)
    
    print("\n--- PROJECT OMEGA: HIVE MIND SIMULATION COMPLETE ---")

if __name__ == "__main__":
    start_time = time.time()
    
    # Clean up from previous runs for a fresh start
    if os.path.exists("project_chimera/p3_metacognitive/work"):
        import shutil
        shutil.rmtree("project_chimera/p3_metacognitive/work")
        
    run_hive_mind_loop()

    end_time = time.time()
    print(f"\nExecution finished in {end_time - start_time:.2f} seconds.")
