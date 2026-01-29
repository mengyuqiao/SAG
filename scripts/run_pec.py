import argparse
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.agents.reasoner import ReasonerAgent
from src.agents.knowledge import KnowledgeAgent
from src.agents.critic import CritiqueAgent
from src.agents.coordinator import CoordinatorAgent

def run_clinical_session(vignette, model_id, max_t=5):
    device = "cuda:0" # Keeping it on GPU 0 as per your previous run
    
    print(f"[🚀] Loading Shared Backbone once: {model_id}")
    
    # 1. Load weights only ONCE
    # Using float16 to fit the 30GB profile promised in your ICML draft
    shared_tokenizer = AutoTokenizer.from_pretrained(model_id)
    shared_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map=device
    )

    # 2. Distribute the same model instance to all agents
    reasoner = ReasonerAgent(shared_model, shared_tokenizer)
    knowledge = KnowledgeAgent(shared_model, shared_tokenizer)
    critic = CritiqueAgent(shared_model, shared_tokenizer)
    coordinator = CoordinatorAgent(shared_model, shared_tokenizer, max_rounds=max_t)

    t = 0
    feedback = None
    
    print(f"[📋] Patient Vignette: {vignette[:100]}...")
    
    while t < max_t:
        print(f"\n--- 🔄 Iteration Round {t+1} ---")
        
        print("[🔍] A_K: Syncing and retrieving clinical evidence...")
        evidence = knowledge.retrieve_evidence(vignette, sync_data=feedback)
        
        print("[🧠] A_R: Generating diagnostic proposal (Ct)...")
        proposal = reasoner.propose(vignette, evidence, feedback)
        
        print("[⚖️] A_C: Auditing proposal for safety and logic gaps...")
        critique = critic.audit(proposal, evidence)
        
        print("[👨‍⚖️] A_Coord: Reviewing consensus and critique strength...")
        result = coordinator.judge(proposal, evidence, critique, t)
        
        if result["status"] == "ACCEPT":
            print("\n[✅] Final Consensus Reached!")
            print(f"Final Clinical Recommendation: \n{result['recommendation']}")
            return
        
        if result["status"] == "REJECT":
            print(f"\n[❌] Rejection: {result['reason']}")
            return
            
        print(f"[⚠️] Critique accepted by Coordinator. Applying directives for Round {t+2}...")
        feedback = result["directive"]
        t += 1

    print("\n[⌛] Reached maximum iterations (T=5). Outputting Fallback/Rejection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAG P_EC Protocol Demo")
    parser.add_argument("--backbone", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model ID or local path")
    parser.add_argument("--rounds", type=int, default=5, help="Max T iterations")
    args = parser.parse_args()

    sample_vignette = (
        "A 45-year-old female presents with acute abdominal pain, peripheral neuropathy, "
        "and psychiatric symptoms after starting a sulfa drug. Single agent initially "
        "suspected appendicitis."
    )
    
    run_clinical_session(sample_vignette, args.backbone, args.rounds)