#!/usr/bin/env python3
from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
print("Loading Llama-3.2-3B-Instruct with vLLM (BitsAndBytes 4-bit)...")

llm = LLM(
    model=MODEL,
    dtype="half",
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    trust_remote_code=True,
    # BitsAndBytes 4-bit quantization
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    # Optional: use 8-bit instead of 4-bit
    # quantization_param_path="nf4"  # default is nf4 (4-bit)
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)

print("\nReady! Type 'quit' to exit\n")

history = []

while True:
    user = input("You: ").strip()
    if user.lower() in {"quit", "exit", "q"}:
        break
    
    history.append({"role": "user", "content": user})
    
    prompt = llm.llm_engine.tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    
    outputs = llm.generate([prompt], sampling_params)
    reply = outputs[0].outputs[0].text.strip()
    
    print(f"Assistant: {reply}\n")
    history.append({"role": "assistant", "content": reply})
