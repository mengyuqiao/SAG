from .base_agent import ClinicalAgent

class CritiqueAgent(ClinicalAgent):
    def audit(self, proposal, evidence):
        prompt = (
            f"SYSTEM: Identify Safety/Logic gaps. Be direct. Your critique MUST be under 300 words.\n"
            f"PROPOSAL: {proposal}\nEVIDENCE: {evidence}\nCRITIQUE Ft:"
        )
        # Critique is limited to ~400 tokens
        return self.generate(prompt, max_new_tokens=400)