import torch
import time
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 目标模型：Qwen-2.5-72B
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

def profile_with_progress(question):
    print(f"\n[🚀] Initializing Deployment Suite for: {MODEL_ID}")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 2. 分布式加载模型（带进度条模拟）
    # 注意：HuggingFace 原生支持 shard 加载进度，我们通过 print 明确阶段
    print(f"[📂] Loading model weights into 6x RTX 6000 Ada VRAM... (Approx. 145GB)")
    start_load = time.perf_counter()
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    load_time = time.perf_counter() - start_load
    print(f"[✅] Model loaded successfully in {load_time:.2f}s")

    # 3. 动态获取参数量
    num_params = model.num_parameters(only_trainable=False)
    
    # 4. 构造 Prompt
    prompt = f"<|im_start|>system\nYou are a medical expert assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]
    
    # 5. 准备推理指标
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 引入流式输出器，解决“等得揪心”的问题
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\n" + "-"*20 + " [🧠 Reasoning Start] " + "-"*20)
    
    torch.cuda.synchronize()
    start_inf = time.perf_counter()

    # 执行生成
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=128, 
            streamer=streamer, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()
    inf_latency = time.perf_counter() - start_inf
    print("\n" + "-"*20 + " [🧠 Reasoning End] " + "-"*20)
    
    # --- 指标计算 (用于 Appendix 数据支撑) ---
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) 
    out_tokens = output.shape[1] - input_len
    
    # 理论计算量公式:
    # $$FLOPs \approx 2 \times P \times (N_{in} + N_{out})$$
    total_tokens = input_len + out_tokens
    theoretical_flops = 2 * num_params * total_tokens

    print(f"\n📊 Deployment Metrics:")
    print(f"   - Peak Memory (Total): {peak_mem_gb:.2f} GB")
    print(f"   - Total Latency: {inf_latency:.2f} s")
    print(f"   - Reasoning Speed: {out_tokens/inf_latency:.2f} tokens/s")
    print(f"   - Computation Cost: {theoretical_flops / 1e12:.2f} TFLOPs")

if __name__ == "__main__":
    # 针对 MedQA 或 NEJM 的典型临床案例
    sample_q = "A 46-year-old male presents with progressive shortness of breath. Physical exam shows decreased breath sounds. Most likely diagnosis?"
    profile_with_progress(sample_q)