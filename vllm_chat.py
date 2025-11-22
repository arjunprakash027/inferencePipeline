# vllm_chat.py — Save this file and run with: python vllm_chat.py

from vllm import LLM, SamplingParams

# This is the EXACT path that exists on your VM
MODEL_PATH = "/app/models/models--Qwen--Qwen3-8B"

print("Loading Qwen3-8B (FP16) on Tesla T4... (~25 seconds)")
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    enforce_eager=True,
    dtype="half",                   # FP16 — fits perfectly in 16GB
    disable_log_stats=True,
)

# Greedy = max accuracy (use 0.7 if you want more creative replies)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    stop=["<|im_end|>", "<|endoftext|>"],
)

print("\nQwen3-8B is ready! Start chatting — type 'quit' to exit\n")

history = [
    {"role": "system", "content": "You are a helpful, accurate, and concise educational assistant."}
]

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye", "q"]:
        print("Goodbye!")
        break

    history.append({"role": "user", "content": user_input})

    # Build correct Qwen3 chat template
    prompt = ""
    for msg in history:
        if msg["role"] == "system":
            prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    outputs = llm.generate([prompt], sampling_params)
    reply = outputs[0].outputs[0].text.strip()

    print(f"\nQwen3-8B: {reply}\n")

    history.append({"role": "assistant", "content": reply})
