from .base_agent import ClinicalAgent
import json

class CoordinatorAgent(ClinicalAgent):
    def __init__(self, model, tokenizer, max_rounds=5):
        super().__init__(model, tokenizer, role="Judge / Final Consensus")
        self.max_rounds = max_rounds
        # Centralized memory for all critiques (Ft) across the PEC protocol
        self.global_critique_history = [] 

    def clear_memory(self):
        """Cleanup memory after ACCEPT/REJECT to prevent context contamination and OOM."""
        self.global_critique_history = []
        print("[🧹] Coordinator memory cleared.")

    def judge(self, proposal, evidence, critique, current_round):
        """
        Main judgment logic:
        1. Accumulate the current critique.
        2. Decide status: ACCEPT, REJECT, or RE-ITERATE.
        3. Enforce word limits (300 words) for the directive.
        """
        # Save current critique to history
        self.global_critique_history.append(critique)
        
        # Build the structured prompt for judgment
        # We explicitly instruct the LLM to provide a concise, structured response
        history_summary = "\n".join([f"Round {i+1} Critique: {c}" for i, c in enumerate(self.global_critique_history)])
        
        prompt = (
            f"SYSTEM: You are the Lead Physician. Evaluate the clinical safety and consistency. "
            f"Your directive MUST be under 300 words. Output in JSON format.\n"
            f"CURRENT_PROPOSAL: {proposal}\n"
            f"CURRENT_CRITIQUE: {critique}\n"
            f"FULL_HISTORY: {history_summary}\n"
            f"TASK: Decide if the proposal is SAFE and CONSISTENT. \n"
            f"If YES: Output status='ACCEPT'. \n"
            f"If NO and round < {self.max_rounds}: Output status='RE-ITERATE' with a concise 'directive'. \n"
            f"If round == {self.max_rounds}: Output status='REJECT'.\n"
            f"FORMAT: {{\"status\": \"...\", \"recommendation\": \"...\", \"directive\": \"...\", \"reason\": \"...\"}}"
        )

        # We limit the Coordinator's output to ~400 tokens to save KV Cache
        raw_output = self.generate(prompt, max_new_tokens=400)
        
        try:
            # Attempt to parse JSON; fallback to simple status if generation is messy
            result = json.loads(raw_output[raw_output.find("{"):raw_output.rfind("}")+1])
        except Exception:
            # Basic fallback logic for robustness
            if "ACCEPT" in raw_output.upper(): result = {"status": "ACCEPT", "recommendation": proposal}
            elif current_round >= self.max_rounds - 1: result = {"status": "REJECT", "reason": "Max iterations reached."}
            else: result = {"status": "RE-ITERATE", "directive": "Refine based on safety check."}
            
        return result