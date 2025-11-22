# vllm_chat.py — FULL QUANTIZATION + vLLM LOAD IN ONE FILE (WORKS ON T4 16GB)

import os
import subprocess
import sys
from pathlib import Path

QUANTIZED_PATH = "/app/models/Llama-3.1-8B-Instruct-AWQ"

def install_awq():
    """Install AutoAWQ if not available"""
    try:
        from awq import AutoAWQForCausalLM
        return True
    except ImportError:
        print("AutoAWQ not found — installing (~30 seconds)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "autoawq==0.2.6", "--no-cache-dir"])
        return True

if not Path(QUANTIZED_PATH).exists():
    print("Quantized model not found → Starting 4-bit AWQ quantization (~5-6 minutes)...")
    
    install_awq()  # Install if needed
    
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model = AutoAWQForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)

    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    )

    os.makedirs(QUANTIZED_PATH, exist_ok=True)
    model.save_quantized(QUANTIZED_PATH)
    tokenizer.save_pretrained(QUANTIZED_PATH)
    print(f"Quantization COMPLETE! Model saved to {QUANTIZED_PATH}")
else:
    print("Quantized model already exists → Skipping quantization")

# Load the quantized model in vLLM
print("Loading 4-bit quantized Llama-3.1-8B-Instruct with vLLM... (~15 seconds)")
from vllm import LLM, SamplingParams

llm = LLM(
    model=QUANTIZED_PATH,
    quantization="awq",  # Key for quantized loading
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
    print(f"\nLlama-3.1-8B (4-bit): {answer}\n")
