#!/usr/bin/env python3
"""
vLLM Local Chat – Llama-3.2-3B-Instruct
Runs 100% on YOUR GPU (NVIDIA T4, A100, RTX, etc.)
Tested & working on Ubuntu with CUDA 12
"""

from vllm import LLM, SamplingParams
from vllm.model_executor.sampling_metadata import SamplingMetadata
import torch

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading Llama-3.2-3B-Instruct with vLLM on YOUR GPU...")
llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    dtype="half",                    # fp16 = fastest on NVIDIA
    trust_remote_code=True,
    enforce_eager=False,             # Let vLLM use its fast CUDA kernels
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024,
    stop=["<|eot_id|>", "<|end_of_text|>"]
)

# Fix for vLLM ≥0.6.0 – use the correct way to apply chat template
tokenizer = llm.get_tokenizer()

print("Model loaded on YOUR GPU! Ready!\n")

history = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    try:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "q", "bye"}:
            print("Goodbye!")
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})

        # THIS IS THE FIXED LINE (works in vLLM 0.6.3+)
        prompt = tokenizer.apply_chat_template(
            history,
            chat_template=None,      # uses model's default template
            add_generation_prompt=True,
            tokenize=False,
            return_dict=False
        )

        print("Assistant: ", end="", flush=True)
        outputs = llm.generate([prompt], sampling_params)

        reply = outputs[0].outputs[0].text
        print(reply)
        print()

        history.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
