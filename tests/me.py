import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 脚本：clinicalnlplab/me-llama
MODEL_ID = "clinicalnlplab/me-llama" 

def run_mellama_benchmark(question):
    print(f"\n[🏥] Starting Rigorous Me-LLaMA-70B Benchmark")
    
    # 1. 环境检查：确保能看到 6 张卡
    n_gpus = torch.cuda.device_count()
    print(f"[i] Detected GPUs: {n_gpus}")
    for i in range(n_gpus):
        free_m = torch.cuda.mem_get_info(i)[0]/1024**3
        print(f"    - GPU {i}: {torch.cuda.get_device_name(i)} | Free: {free_m:.2f} GB")

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    
    # 3. 强制 GPU 加载，严禁 CPU Offloading
    print(f"[📂] Loading 140GB weights... Ensuring zero CPU-offload.")
    # 脚本：clinicalnlplab/me-llama
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        # 核心修复：防止 RAM 被撑爆
        low_cpu_mem_usage=True, 
        # 强制 GPU 分片，预留显存余量
        max_memory={i: "42GiB" for i in range(6)}, 
        trust_remote_code=True
    )

    # 检查 Device Map 是否包含 CPU 或 Disk
    if any(v in ['cpu', 'disk'] for v in model.hf_device_map.values()):
        print("⚠️ 警告：检测到 CPU/Disk 卸载！推理速度将极慢。请检查 GPU 0 空间。")
    else:
        print("[✅] All layers successfully mapped to GPUs.")

    num_params = model.num_parameters(only_trainable=False) #
    
    prompt = f"Instruction: Provide a professional clinical analysis.\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\n" + "="*15 + " [🧠 Inference Start] " + "="*15)
    
    # 重置所有卡的显存峰值统计
    for i in range(n_gpus):
        torch.cuda.reset_peak_memory_stats(i)
    
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
    latency = time.perf_counter() - start_time
    print("\n" + "="*15 + " [🧠 Inference End] " + "="*15)
    
    # --- 核心指标修正 ---
    # 1. 汇总 6 张卡的峰值显存总和
    total_peak_mem = sum(torch.cuda.max_memory_allocated(i) for i in range(n_gpus)) / (1024**3)
    
    out_tokens = output.shape[1] - input_len
    flops = 2 * num_params * (input_len + out_tokens)

    print(f"\n📊 Me-LLaMA-70B Rigorous Results:")
    print(f"   - System-wide Peak Memory: {total_peak_mem:.2f} GB") # 解决 21GB 的统计幻觉
    print(f"   - End-to-End Latency: {latency:.2f} s")
    print(f"   - Reasoning Speed: {out_tokens/latency:.2f} tokens/s")
    print(f"   - Total TFLOPs: {flops / 1e12:.2f}")

if __name__ == "__main__":
    # 先清理一遍僵尸进程
    import subprocess
    subprocess.run(["pkill", "-u", "yuqiao", "-9", "python"])
    
    run_mellama_benchmark("A patient with history of heavy smoking has progressive dyspnea and clubbing. Diagnosis?")