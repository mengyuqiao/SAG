import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 脚本 1: meta-llama/Meta-Llama-3-70B-Instruct
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct" 

def run_llama3_benchmark(question):
    print(f"\n[🚀] Starting Dedicated Llama-3-70B Benchmark")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    max_memory = {i: "45GiB" for i in range(torch.cuda.device_count())}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    
    import transformers
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("attn impl:", getattr(model.config, "_attn_implementation", None))
    print(model.hf_device_map)  # 看看有没有 'cpu' 或 'disk'
    if any(v in ['cpu', 'disk'] for v in model.hf_device_map.values()):
        print("⚠️ 警告：检测到 CPU/Disk 卸载！推理速度将极慢。请检查 GPU 0 空间。")
    else:
        print("[✅] All layers successfully mapped to GPUs.")

    num_params = model.num_parameters(only_trainable=False) # 修复 API 报错
    
    # Llama-3 官方对话模板
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a helpful medical expert assistant.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]
    
    # 实时流式输出，拒绝“揪心”等待
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\n" + "="*15 + " [🧠 Llama-3 Inference] " + "="*15)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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
    print("\n" + "="*45)
    
    # 数据汇总
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) 
    out_tokens = output.shape[1] - input_len
    flops = 2 * num_params * (input_len + out_tokens)

    print(f"📊 Llama-3-70B Results:")
    print(f"   - Peak Memory: {peak_mem:.2f} GB")
    print(f"   - Latency: {latency:.2f} s")
    print(f"   - TFLOPs: {flops / 1e12:.2f}")

if __name__ == "__main__":
    run_llama3_benchmark("What are the primary symptoms of chronic obstructive pulmonary disease (COPD)?")