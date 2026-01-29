# src/agents/reasoner.py
from .base_agent import ClinicalAgent

class ReasonerAgent(ClinicalAgent):
    def propose(self, vignette, evidence):
        history = "\n".join(self.critique_history)
        # Instruction to limit length without truncating logic
        prompt = (
            f"SYSTEM: Be concise. Your clinical reasoning MUST be under 800 words.\n"
            f"VIGNETTE: {vignette}\nEVIDENCE: {evidence}\nPAST_CRITIQUES: {history}\n"
            f"PROPOSE Ct:"
        )
        # We limit Reasoner to 1024 tokens to keep space for others
        return self.generate(prompt, max_new_tokens=1024)