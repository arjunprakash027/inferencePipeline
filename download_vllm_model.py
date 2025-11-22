"""
Model Download Script for vLLM Pipeline - Tech Arena 2025
Downloads Qwen2.5 models for offline evaluation
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def download_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct", cache_dir: str = "/app/models"):
    """Download model and tokenizer"""
    print("=" * 80)
    print(f"📥 DOWNLOADING: {model_name}")
    print("=" * 80)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Cache: {cache_path.absolute()}")

    try:
        print("\n🔤 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✅ Tokenizer OK")

        print("\n🤖 Downloading model (this takes 5-10 min)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype="float16",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model OK")

        print(f"\n✅ DOWNLOAD COMPLETE!")
        print(f"📦 Model ready at: {cache_path.absolute()}")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
    cache = sys.argv[2] if len(sys.argv) > 2 else "/app/models"

    download_model(model, cache)
