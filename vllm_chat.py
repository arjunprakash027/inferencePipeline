# vllm_chat.py — FIXED: GPTQ 4-bit Quantization + vLLM Load for Llama-3.1-8B-Instruct on T4

import os
import subprocess
import sys
from pathlib import Path

QUANTIZED_PATH = "/app/models/Llama-3.1-8B-Instruct-GPTQ"

def install_gptq():
    """Install optimum[gptq] if not available (compatible with vLLM 0.6.3 + Torch 2.4.1)"""
    try:
        from optimum.gptq import GPTQQuantizer
        return True
    except ImportError:
        print("GPTQ not found — installing (~45 seconds)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "optimum[gptq]==1.21.0",
            "--no-cache-dir", "--force-reinstall", "--no-deps"
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "auto-gptq==0.8.0",
            "--no-cache-dir", "--force-reinstall"
        ])
        return True

if not Path(QUANTIZED_PATH).exists():
    print("Quantized model not found → Starting 4-bit GPTQ quantization (~4-5 minutes)...")
    
    install_gptq()  # Install if needed
    
    from optimum.gptq import GPTQQuantizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)

    quantizer = GPTQQuantizer(
        bits=4,
        dataset=["The capital of France is Paris."] * 128,  # Calibration dataset
        block_size=128,
        desc_act=False,
        group_size=128,
        disable_exllama=False,  # Use ExLlamaV2 for better accuracy
        tokenizer=tokenizer,
    )

    os.makedirs(QUANTIZED_PATH, exist_ok=True)
    quantizer.quantize_model(model, tokenizer, save_directory=QUANTIZED_PATH)
    tokenizer.save_pretrained(QUANTIZED_PATH)
    
    print(f"GPTQ Quantization COMPLETE! Model saved to {QUANTIZED_PATH}")
else:
    print("Quantized model already exists → Skipping quantization")

# Load the quantized model in vLLM
print("Loading 4-bit quantized Llama-3.1-8B-Instruct with vLLM... (~15 seconds)")
from vllm import LLM, SamplingParams

llm = LLM(
    model=QUANTIZED_PATH,
    quantization="gptq",  # Key for GPTQ loading
    dtype="half",
    gpu_memory_utilization=0.95,  # Safe for 4-bit (~5 GB)
    max_model_len=8192,
    enforce_eager=True,
    trust_remote_code=True,
    disable_log_stats=True,
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=1024,
    stop=["<|eot_id|>", "<|end_of_text|>"]
)

print("\n8B MODEL QUANTIZED + LOADED SUCCESSFULLY! Start chatting (type 'quit' to exit)\n")

# Interactive chat
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break
    if not user_input:
        continue

    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    outputs = llm.generate([prompt], sampling_params)
    answer = outputs[0].outputs[0].text.strip()
    print(f"\nLlama-3.1-8B (4-bit GPTQ): {answer}\n")
