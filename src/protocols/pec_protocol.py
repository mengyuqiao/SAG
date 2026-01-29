class PECProtocol:
    def execute(self, vignette):
        t = 0
        feedback = None
        
        while t < 5:
            # 1. Sync & Evidence Retrieval
            evidence = self.knowledge_agent.retrieve_evidence(vignette)
            
            # 2. Proposal Generation
            proposal = self.reasoner_agent.propose(vignette, evidence, feedback)
            
            # 3. Safety & Logic Check
            critique = self.critic_agent.audit(proposal, evidence)
            
            # 4. Coordinator Judgment
            result = self.coordinator_agent.judge(proposal, evidence, critique, t)
            
            if result["status"] == "ACCEPT":
                return result["recommendation"]
            elif result["status"] == "REJECT":
                return "Fallback: Rejection"
            
            # 5. Apply Directives and Re-iterate
            feedback = result["directive"]
            t += 1