import subprocess
import sys
from pathlib import Path
from llama_cpp import Llama


def find_snapshot_path(hf_cache_dir):
    """
    Find the actual model snapshot path in HuggingFace cache structure.
    HF cache has structure: models--org--name/snapshots/<hash>/
    """
    cache_path = Path(hf_cache_dir)
    snapshots_dir = cache_path / "snapshots"
    
    if snapshots_dir.exists():
        # Get the first (usually most recent) snapshot
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            return snapshots[0]
    
    # If no snapshots directory, assume it's already the correct path
    return cache_path


def convert_to_gguf(hf_model_path, output_gguf_path):
    """
    Convert HuggingFace model to GGUF format.
    Uses the llama.cpp conversion script.
    """
    # Point to llama.cpp/convert_hf_to_gguf.py in your project
    convert_script = Path(__file__).parent / "llama.cpp" / "convert_hf_to_gguf.py"
    
    # Find the actual snapshot path
    model_path = find_snapshot_path(hf_model_path)
    print(f"Using model path: {model_path}")
    
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_gguf_path),
        "--outtype", "f16"  # or "q8_0" for 8-bit quantization
    ]
    
    print(f"Running conversion: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Conversion complete: {output_gguf_path}")


def run_inference(gguf_model_path, prompt):
    """
    Run inference using llama-cpp-python bindings.
    """
    print(f"Loading model from: {gguf_model_path}")
    
    # Load model
    llm = Llama(
        model_path=str(gguf_model_path),
        n_ctx=2048,          # Context window
        n_threads=16,        # Use all 16 cores on your AMD CPU
        n_gpu_layers=0,      # 0 for CPU-only inference
        verbose=False        # Set to True for debugging
    )
    
    print(f"Generating response for prompt: '{prompt}'")
    
    # Generate
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        echo=False
    )
    
    return output['choices'][0]['text']


# Main workflow
if __name__ == "__main__":
    # Your HF model cache (parent directory)
    hf_cache_path = "./app/model/models--meta-llama--Llama-3.2-1B-Instruct"
    gguf_output = "model.gguf"
    
    # Convert once before submission
    convert_to_gguf(hf_cache_path, gguf_output)
    
    # Run inference (works offline)
    prompt = "The capital of Ireland is"
    result = run_inference(gguf_output, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result}")
