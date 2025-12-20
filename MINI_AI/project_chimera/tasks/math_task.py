# NEW FILE
import numpy as np
from ..l3_cognitive.symbolic_engine import SymbolicLogicEngine
from ..p4_environment.dynamic_math_env import DynamicMathEnvironment
from ..l2_data.unigram_tokenizer import UnigramTokenizer
from ..l1_calculus.tensor import Tensor
from ..l1_calculus.optimizers import Adam
from ..nn.losses import CrossEntropyLoss

class MathTask:
    """
    Defines the math task, including data generation, training, and evaluation logic.
    """
    def __init__(self):
        self.env = DynamicMathEnvironment()
        corpus = self.env.numbers + [op[0] for op in self.env.operators]
        self.tokenizer = UnigramTokenizer(vocab_size=50)
        self.tokenizer.train(corpus)

    def run(self, synthesizer_module):
        """Runs a single evaluation step for the P3 engine."""
        print("--- [TASK] Running Evaluation ---")
        if synthesizer_module is None: return {'loss': float('inf')}
        
        try:
            synthesizer_instance = synthesizer_module.ProgramSynthesizer(
                vocab_size=len(self.tokenizer.vocab) + 10,
                embed_size=32, hidden_size=32
            )
        except Exception as e:
            print(f"[TASK EVAL] Failed to instantiate model from module: {e}")
            return {'loss': float('inf')}

        optimizer = Adam(synthesizer_instance.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()
        
        problem = self.env.get_new_problem()
        if not problem: return {'loss': float('inf')}

        question_text = problem['question']
        target_program_text = f"({problem['correct_answer']})"
        
        q_tokens = self.tokenizer.encode(question_text)
        p_tokens = self.tokenizer.encode(target_program_text)
        if not q_tokens or not p_tokens: return {'loss': float('inf')}

        input_tensor = Tensor(np.array([q_tokens]))
        target_tensor = Tensor(np.array([p_tokens[0]]))
        
        optimizer.zero_grad()
        predicted_logits = synthesizer_instance.forward(input_tensor)
        loss = loss_fn.forward(predicted_logits, target_tensor.data.astype(int))
        loss.backward()
        optimizer.step()
        
        loss_val = loss.data.item()
        print(f"  [TASK] Evalled: '{question_text}' -> '{target_program_text}'. Loss: {loss_val:.4f}")
        return {"loss": loss_val}
    
    def train_model(self, model, epochs=50): # Reduced epochs for CPU
        """A method to actually train a new synthesizer instance from scratch."""
        print(f"[TASK] Training new model instance for {epochs} epochs...")
        optimizer = Adam(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            problem = self.env.get_new_problem()
            if not problem: continue

            q_tokens = self.tokenizer.encode(problem['question'])
            p_str = f"({problem['correct_answer']})"
            p_tokens = self.tokenizer.encode(p_str)
            if not q_tokens or not p_tokens: continue

            input_tensor = Tensor(np.array([q_tokens]))
            logits = model.forward(input_tensor)
            target_tensor = Tensor(np.array([p_tokens[0]])) 
            
            loss = loss_fn.forward(logits, target_tensor.data.astype(int))
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 25 == 0:
                print(f"    [TRAIN] Epoch {epoch+1}, Loss: {loss.data.item():.4f}")
