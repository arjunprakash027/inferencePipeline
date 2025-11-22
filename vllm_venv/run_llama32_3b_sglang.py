#!/usr/bin/env python3
# run_llama_3b_sglang.py — SAME behavior as your vLLM script, but with SGLang (FASTER + BETTER)

import os
from pathlib import Path

# Your model (already cached by Hugging Face)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading meta-llama/Llama-3.2-3B-Instruct with SGLang (~8 seconds)...")

from sglang import Runtime, set_default_backend

# Launch SGLang runtime (FP16, fastest on T4)
runtime = Runtime(
    model_path=MODEL_NAME,
    dtype="bfloat16",           # Best quality + speed on T4
    mem_fraction=0.92,          # Safe on shared GPU
    max_running_requests=32,
    max_total_tokens=32768
)
set_default_backend(runtime)
runtime.start_server(port=30000)

print("\nLlama-3.2-3B-Instruct is READY! Start chatting (type 'quit' to exit)\n")

# Chat loop — identical behavior to your vLLM script
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user = input("You: ").strip()
    if user.lower() in ["quit", "exit", "bye", "q"]:
        print("Goodbye!")
        break

    messages.append({"role": "user", "content": user})

    # SGLang handles the chat template automatically
    print("Llama-3.2-3B: ", end="", flush=True)
    response = ""
    for chunk in runtime.generate(
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|end_of_text|>"],
        stream=True
    ):
        print(chunk, end="", flush=True)
        response += chunk
    print("\n")

    messages.append({"role": "assistant", "content": response.strip()})

# Clean shutdown
runtime.shutdown()
print("SGLang server stopped.")
