

# clinicalnlplab/me-llama
# meta-llama/Meta-Llama-3-70B-Instruct
# Qwen/Qwen2.5-72B-Instruct
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_ID = "epfl-llm/meditron-70b"

def profile_comprehensive(question):
    print(f"[⚙️] Loading {MODEL_ID} across your 6x RTX 6000 Ada...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 自动切分到 6 张 48GB 卡
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    # 获取真实的参数量用于 FLOPs 计算
    # 建议加上 only_trainable=False 以获取完整的模型参数量
    num_params = model.num_parameters(only_trainable=False)
    
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]
    
    # 重置显存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 引入流式输出器，让你不再傻等
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\n[⚡] Starting Inference (Streaming Mode)...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=128, 
            streamer=streamer, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # --- 指标计算 ---
    latency = end_time - start_time
    # 统计所有卡中的峰值显存总和
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) 
    out_tokens = output.shape[1] - input_len
    
    # 理论 FLOPs: 2 * P * (N_in + N_out)
    total_tokens = input_len + out_tokens
    theoretical_flops = 2 * num_params * total_tokens

    print("\n" + "="*40)
    print(f"📊 Final Profiling Results for {MODEL_ID}:")
    print(f"   - Peak Memory (System-wide): {peak_mem_gb:.2f} GB")
    print(f"   - End-to-End Latency: {latency:.2f} s")
    print(f"   - Throughput: {out_tokens/latency:.2f} tokens/s")
    print(f"   - Theoretical Computation: {theoretical_flops / 1e12:.2f} TFLOPs")
    print("="*40)

if __name__ == "__main__":
    sample = "A 46-year-old man presents with progressive shortness of breath. What is the most likely diagnosis?"
    profile_comprehensive(sample)