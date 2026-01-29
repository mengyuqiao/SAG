# src/agents/knowledge.py
from .base_agent import ClinicalAgent

class KnowledgeAgent(ClinicalAgent):
    def retrieve_evidence(self, vignette):
        history = "\n".join(self.critique_history)
        prompt = (
            f"SYSTEM: Retrieve only high-relevance evidence. Limit to 500 words.\n"
            f"VIGNETTE: {vignette}\nHISTORY: {history}\nRETRIEVE E:"
        )
        return self.generate(prompt, max_new_tokens=800)