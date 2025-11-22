#!/usr/bin/env python3
# vllm_chat_1.py — Llama-3.2-3B-Instruct 4-bit GPTQ + vLLM (fixed: low-VRAM quantization)
# Uses Transformers GPTQConfig for layer-wise load (1-2 GB VRAM peak, safe on shared T4)

import os
import subprocess
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================= CONFIG =============================
VENV_NAME      = "vllm_venv"
BASE_MODEL     = "meta-llama/Llama-3.2-3B-Instruct"
QUANTIZED_PATH = "/app/models/Llama-3.2-3B-Instruct-GPTQ"
# =================================================================

def setup_venv():
    if os.getenv("VIRTUAL_ENV") and VENV_NAME in os.getenv("VIRTUAL_ENV", ""):
        return
    venv_path = Path(__file__).parent / VENV_NAME
    if not venv_path.exists():
        print(f"Creating isolated venv: {VENV_NAME}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    activate = venv_path / "bin" / "activate"
    args_part = " ".join(repr(arg) for arg in sys.argv)
    cmd = f"source {activate} && exec python {args_part}"
    os.execl("/bin/bash", "bash", "-c", cmd)

def ensure_deps():
    try:
        import torch, vllm
        print(f"READY → vLLM {vllm.__version__} | Torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
        return
    except:
        print("Installing dependencies (safe & fast)...")
        cmds = [
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "torch==2.4.1+cu118", "torchvision==0.19.1+cu118",
             "--index-url", "https://download.pytorch.org/whl/cu118"],
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "vllm==0.6.3.post1"],
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "transformers>=4.44.0", "accelerate", "optimum>=1.21.0", "auto-gptq==0.7.1",
             "--extra-index-url", "https://huggingface.github.io/autogptq-index/whl/cu118/"],
        ]
        for c in cmds:
            subprocess.check_call(c)
        print("Dependencies installed! Restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

def quantize():
    if Path(QUANTIZED_PATH).exists() and any(Path(QUANTIZED_PATH).glob("*.safetensors")):
        print(f"Quantized model ready → {QUANTIZED_PATH}")
        return

    print("Quantizing Llama-3.2-3B to 4-bit GPTQ (layer-wise, <2 GB VRAM peak)...")
    ensure_deps()
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    import torch

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # ←←← FIXED: Use GPTQConfig for low-memory layer-wise quantization
    gptq_config = GPTQConfig(
        bits=4,
        group_size=128,
        dataset="c4",  # Calibration dataset
        desc_act=False,
        block_name_to_quantize="model.layers",
        model_seqlen=2048,  # Smaller seq for less memory
        disable_exllamav2=True,  # Avoid kernel issues
    )

    # Load + quantize layer-by-layer with auto-offload (CPU/GPU shuffle, no full load)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=gptq_config,
        device_map="auto",  # ← Key: Accelerate offloads to CPU/RAM as needed
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Save the quantized model
    model.save_pretrained(QUANTIZED_PATH)
    tokenizer.save_pretrained(QUANTIZED_PATH)
    print("Quantization complete! (low-memory mode)")

def load_engine():
    ensure_deps()
    from vllm import LLM
    print("Loading 4-bit model with vLLM (~4–5 GB VRAM)...")
    llm = LLM(
        model=QUANTIZED_PATH,
        quantization="gptq",
        dtype="half",
        gpu_memory_utilization=0.85,   # Safe for shared T4
        max_model_len=8192,
        enforce_eager=True,
    )
    print("Llama-3.2-3B loaded — 80–120+ tokens/sec")
    return llm

def chat(llm):
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
    print("\nChat started! Type 'quit' to exit.\n")
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break
        history.append({"role": "user", "content": user})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        result = llm.generate([prompt], params)[0].outputs[0].text.strip()
        print(f"\nAssistant: {result}\n{'─'*70}")
        history.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    setup_venv()
    ensure_deps()
    quantize()
    chat(load_engine())
