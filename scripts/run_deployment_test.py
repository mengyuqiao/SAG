import argparse
import sys
import os
import torch
import time

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.agents.reasoner import ReasonerAgent
from src.agents.knowledge import KnowledgeAgent
from src.agents.critic import CritiqueAgent
from src.agents.coordinator import CoordinatorAgent

def run_clinical_session(vignette, model_id, max_t=5):
    dev_id = 0 
    
    print(f"[🚀] Loading Shared Backbone: {model_id}")
    
    # Initialize Shared Model and Tokenizer
    shared_tokenizer = AutoTokenizer.from_pretrained(model_id)
    shared_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map={"": dev_id}
    )
    
    # Setup Deployment Profiling
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats(dev_id)
    except Exception:
        pass
        
    params = sum(p.numel() for p in shared_model.parameters())
    start_time = time.perf_counter()
    total_tokens_processed = 0

    # Initialize Agents (Fixing the missing 'role' argument error)
    reasoner = ReasonerAgent(shared_model, shared_tokenizer, role="Reasoning-Heavy")
    knowledge = KnowledgeAgent(shared_model, shared_tokenizer, role="Knowledge-Heavy")
    critic = CritiqueAgent(shared_model, shared_tokenizer, role="Safety-Check")
    coordinator = CoordinatorAgent(shared_model, shared_tokenizer, max_rounds=max_t, role="Judge")
    
    agents = [reasoner, knowledge, critic, coordinator]

    t = 0
    print(f"[📋] Vignette: {vignette[:80]}...")
    
    while t < max_t:
        print(f"\n--- 🔄 Iteration Round {t+1} ---")
        
        # 1. Knowledge Retrieval (Instruction-based length limit)
        evidence = knowledge.retrieve_evidence(vignette)
        
        # 2. Proposal Generation (Limited to max 1024 tokens for COT)
        proposal = reasoner.propose(vignette, evidence)
        
        # 3. Auditing (Limited to max 400 tokens)
        critique = critic.audit(proposal, evidence)
        
        # 4. Coordinator Judgment & History Accumulation
        result = coordinator.judge(proposal, evidence, critique, t)
        
        # Tracking tokens for FLOPs (2 * P * L)
        total_tokens_processed += len(shared_tokenizer.encode(evidence + proposal + critique))
        
        if result["status"] in ["ACCEPT", "REJECT"]:
            status_tag = "[✅] Recommendation" if result["status"] == "ACCEPT" else "[❌] Rejection"
            output = result.get("recommendation") or result.get("reason")
            print(f"\n{status_tag}: {output}")
            
            # CRITICAL: Clear all critique memory once judgment is final
            for agent in agents:
                agent.clear_memory()
            break
            
        # 5. Iterative Sync: Save directive to history for next round
        directive = result.get("directive", "Continue refinement.")
        knowledge.save_critique(directive)
        reasoner.save_critique(directive)
        
        print(f"[💾] Feedback saved. Round {t+1} context pruned.")
        t += 1

    # --- Metrics Calculation ---
    end_time = time.perf_counter()
    latency = end_time - start_time
    peak_mem = torch.cuda.max_memory_allocated(dev_id) / (1024**3)
    tflops = (2 * params * total_tokens_processed) / 1e12

    print("\n" + "="*40)
    print("📊 MEASURED DEPLOYMENT COSTS")
    print("="*40)
    print(f"End-to-End Latency:   {latency:.2f} s")
    print(f"Peak GPU Memory:      {peak_mem:.2f} GB")
    print(f"Estimated Compute:    {tflops:.2f} TFLOPs")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    sample_vignette = (
        "A 45-year-old female presents with acute abdominal pain and neuropathy "
        "after starting a sulfa drug. Initial differential: Appendicitis."
    )
    
    run_clinical_session(sample_vignette, args.backbone, args.rounds)