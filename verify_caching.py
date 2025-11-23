
import time
import asyncio
import aiohttp
import sys
import os
import random
import string
import json

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from inferencePipeline.server_pipeline import ServerInferencePipeline, SETTINGS

async def measure_latency(pipeline, prompt, name, max_tokens=100):
    print(f"\n--- Testing {name} ---")
    print(f"Prompt length: {len(prompt)} chars")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        data = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0
        }
        
        async with session.post(f"{pipeline.server.base_url}/completion", json=data) as response:
            if response.status == 200:
                result = await response.json()
                # print(f"Response preview: {result['content'][:50]}...")
            else:
                text = await response.text()
                print(f"Error: {response.status} - {text}")
                return None

    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency:.4f} seconds")
    return latency

def generate_dummy_kb(length):
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

async def main():
    print("Initializing pipeline (this includes warmup)...")
    # This starts the server with settings.json (should be 16384 context)
    pipeline = ServerInferencePipeline()
    
    # Allow server to fully settle
    await asyncio.sleep(2)
    
    # 1. Test Algebra (Context Size Check)
    print("\n=== 1. Algebra Context Test ===")
    algebra_q = "Solve the system: x + y = 10, x - y = 2"
    algebra_prompt = pipeline.create_expert_prompt(algebra_q, "algebra")
    
    # This prompt is > 5000 tokens. If context is 2048, it will fail.
    # If context is 16384, it should pass.
    latency_alg = await measure_latency(pipeline, algebra_prompt, "Algebra Prompt (Large Context)")
    
    if latency_alg is not None:
        print("✅ Algebra Test Passed! Context size is sufficient.")
    else:
        print("❌ Algebra Test Failed! Context size likely too small.")
        return

    # 2. Test Caching (CAG Verification)
    print("\n=== 2. Caching Performance Test ===")
    
    # Cached Case: Use Chinese KB (which was warmed up)
    # We use a NEW question to ensure we are testing prefix caching, not exact response caching.
    cached_prompt = pipeline.create_expert_prompt("中国古代四大发明是什么？", "chinese")
    
    # Uncached Case: Use a Dummy KB of same length
    chinese_kb_len = len(pipeline.chinese_kb)
    dummy_kb = generate_dummy_kb(chinese_kb_len)
    
    uncached_system = f"""你是中国文化和语言专家。使用以下参考资料回答问题。

参考资料：
{dummy_kb}

基于以上参考资料，直接回答以下问题："""
    uncached_user = "问题: 中国古代四大发明是什么？\n答案:"
    uncached_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{uncached_system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{uncached_user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Measure Latency
    # We expect Cached to be much faster because it skips processing the huge KB
    t_cached = await measure_latency(pipeline, cached_prompt, "Cached Context (Chinese KB)")
    t_uncached = await measure_latency(pipeline, uncached_prompt, "Uncached Context (Random KB)")
    
    if t_cached and t_uncached:
        print("\n--- Results ---")
        print(f"Cached Latency:   {t_cached:.4f}s")
        print(f"Uncached Latency: {t_uncached:.4f}s")
        
        speedup = t_uncached / t_cached
        print(f"Speedup: {speedup:.2f}x")
        
        if t_uncached > t_cached * 1.5:
            print("✅ SUCCESS: Caching is working effectively!")
        else:
            print("⚠️  WARNING: Caching speedup is low. Check if warmup actually worked.")

if __name__ == "__main__":
    asyncio.run(main())
