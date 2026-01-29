class ClinicalAgent:
    def __init__(self, model, tokenizer, role):
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        # History is now managed via the Coordinator, but agents store a local copy
        self.critique_history = [] 

    def clear_memory(self):
        """Reset history to avoid OOM in subsequent cases."""
        self.critique_history = []

    def generate(self, prompt, max_new_tokens=512):
        # We enforce a hard generation limit to save KV Cache
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)