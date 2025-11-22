#!/usr/bin/env python3
"""
vLLM Chat with Llama 3.2-3B-Instruct using BitsAndBytes 4-bit Quantization
"""
from vllm import LLM, SamplingParams

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading Llama-3.2-3B-Instruct with vLLM (BitsAndBytes 4-bit)...")
print("This will save significant GPU memory!\n")

llm = LLM(
    model=MODEL_NAME,
    dtype="half",
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    trust_remote_code=True,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)

print("✓ Model loaded successfully!")
print("Type 'quit' to exit\n")

history = []

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in {"quit", "exit", "q"}:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    history.append({"role": "user", "content": user_input})
    
    # Format conversation history as chat template
    prompt = llm.llm_engine.tokenizer.apply_chat_template(
        history, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate response
    outputs = llm.generate([prompt], sampling_params)
    reply = outputs[0].outputs[0].text.strip()
    
    print(f"Assistant: {reply}\n")
    
    history.append({"role": "assistant", "content": reply})
