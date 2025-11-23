
import time
import os
from llama_cpp import Llama

# Create a dummy model if not exists (or use existing one)
# We will use the one defined in settings if possible, or just download a small one?
# For this test, we assume the model exists as per previous steps.
# We will try to load the model from the path in settings.

import json
from pathlib import Path

def get_model_path():
    try:
        with open("/tmp/inferencePipeline/inferencePipeline/settings.json", "r") as f:
            settings = json.load(f)
        model_cfg = settings['model']
        server_cfg = settings['server']
        gguf_cache = Path(model_cfg['gguf_cache_dir'])
        return str(gguf_cache / server_cfg['model_file'])
    except Exception as e:
        print(f"Error reading settings: {e}")
        return None

def test_caching():
    model_path = get_model_path()
    if not model_path or not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Cannot run test.")
        return

    print(f"Loading model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0, # Use CPU for consistent timing comparison or small GPU usage
        verbose=False
    )

    # Define a long prefix (simulating a Knowledge Base)
    prefix = "This is a long prefix that represents a knowledge base. " * 50 # ~500 tokens
    
    prompt1 = prefix + "Question 1: What is the first letter of the alphabet? Answer:"
    prompt2 = prefix + "Question 2: What is the second letter of the alphabet? Answer:"

    print("\n--- Run 1 (Cold Cache) ---")
    start = time.perf_counter()
    llm(prompt1, max_tokens=5)
    end = time.perf_counter()
    time1 = end - start
    print(f"Time 1: {time1:.4f}s")

    print("\n--- Run 2 (Should be Warm Cache) ---")
    start = time.perf_counter()
    llm(prompt2, max_tokens=5)
    end = time.perf_counter()
    time2 = end - start
    print(f"Time 2: {time2:.4f}s")

    if time2 < time1 * 0.5:
        print("\n✅ SUCCESS: Caching is working! Run 2 was significantly faster.")
    else:
        print("\n❌ FAILURE: Caching might not be working. Times are too similar.")
        print("Note: If the model is very small or CPU is fast, the difference might be small.")

if __name__ == "__main__":
    test_caching()
