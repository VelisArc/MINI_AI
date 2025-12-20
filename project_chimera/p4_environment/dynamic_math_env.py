# NEW FILE
import random

class DynamicMathEnvironment:
    """
    An environment that generates a stream of novel math problems for the agent.
    This simulates an open-ended world where the agent must continuously adapt.
    """
    def __init__(self):
        self.numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.operators = [("plus", "+"), ("minus", "-"), ("times", "*")]
        print("[P4 ENV] Dynamic Math Environment initialized.")

    def get_new_problem(self):
        """Generates a new, potentially unseen, math problem."""
        n1_val = random.randint(0, 9)
        n2_val = random.randint(0, 9)
        op_str, op_sym = random.choice(self.operators)

        if op_sym == "-" and n1_val < n2_val:
            n1_val, n2_val = n2_val, n1_val # Avoid negative results

        question = f"what is {self.numbers[n1_val]} {op_str} {self.numbers[n2_val]}"
        
        try:
            # The ground truth answer
            correct_answer = eval(f"{n1_val}{op_sym}{n2_val}")
        except:
            return None # Should not happen with current setup

        print(f"[P4 ENV] New Problem: '{question}' (Answer: {correct_answer})")
        return {"question": question, "correct_answer": correct_answer}
