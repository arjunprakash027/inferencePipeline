"""
System Diagnostics for Tech Arena 2025 Pipeline
Checks GPU, CUDA, dependencies, and model availability
"""

import sys
import os
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80)


def check_python():
    """Check Python version"""
    print_section("PYTHON VERSION")

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("✅ Python version compatible (3.10+)")
        return True
    else:
        print("❌ Python 3.10+ required")
        return False


def check_cuda():
    """Check CUDA availability"""
    print_section("CUDA & GPU")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"\nGPU Devices:")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  [{i}] {torch.cuda.get_device_name(i)}")
                print(f"      Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"      Compute capability: {props.major}.{props.minor}")
                print(f"      Multi-processors: {props.multi_processor_count}")

            # Check if T4
            name = torch.cuda.get_device_name(0)
            if "T4" in name:
                print(f"\n✅ Tesla T4 detected - optimal for competition!")
            else:
                print(f"\n⚠️  Not a T4 GPU - performance may differ")

            return True
        else:
            print("❌ CUDA not available - GPU inference will not work!")
            print("   Solution: Install PyTorch with CUDA support:")
            print("   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118")
            return False

    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_dependencies():
    """Check required packages"""
    print_section("DEPENDENCIES")

    required = {
        "torch": "2.5.1",
        "transformers": "4.49.0",
        "vllm": "0.6.3",
    }

    all_ok = True

    for package, expected_version in required.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")

            # Check version match (major.minor)
            version_parts = version.split(".")[:2]
            expected_parts = expected_version.split(".")[:2]

            if version_parts == expected_parts or "unknown" in version:
                print(f"✅ {package}: {version}")
            else:
                print(f"⚠️  {package}: {version} (expected {expected_version})")
                all_ok = False

        except ImportError:
            print(f"❌ {package}: NOT INSTALLED")
            all_ok = False

    if all_ok:
        print("\n✅ All core dependencies installed")
    else:
        print("\n⚠️  Some dependencies missing or mismatched")
        print("   Install with: pip install -r requirements_vllm.txt")

    return all_ok


def check_vllm_import():
    """Test vLLM import and basic functionality"""
    print_section("VLLM FUNCTIONALITY")

    try:
        from vllm import LLM, SamplingParams

        print("✅ vLLM imports successfully")

        # Check if FP8 support available
        try:
            import torch

            if torch.cuda.is_available():
                # Check compute capability for FP8
                cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor

                if cc >= 70:  # Volta and newer
                    print(f"✅ FP8 quantization supported (compute capability {cc/10:.1f})")
                else:
                    print(f"⚠️  FP8 may not be optimal (compute capability {cc/10:.1f})")
                    print("   Will automatically fall back to FP16")
        except Exception as e:
            print(f"⚠️  Could not check FP8 support: {e}")

        return True

    except ImportError as e:
        print(f"❌ vLLM import failed: {e}")
        print("   Solution: pip install vllm==0.6.3.post1")
        return False


def check_model_cache():
    """Check if model is downloaded"""
    print_section("MODEL CACHE")

    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "/app/models"))
    print(f"Cache directory: {cache_dir}")

    if not cache_dir.exists():
        print(f"⚠️  Cache directory does not exist")
        print(f"   Will be created on first run")
        return False

    # Check for Qwen model
    qwen_patterns = [
        "models--Qwen--Qwen2.5-3B-Instruct",
        "Qwen2.5-3B-Instruct",
        "qwen2.5-3b",
    ]

    found = False
    for pattern in qwen_patterns:
        matches = list(cache_dir.rglob(f"*{pattern}*"))
        if matches:
            print(f"✅ Found model: {matches[0].name}")
            found = True
            break

    if not found:
        print("⚠️  Model not found in cache")
        print("   Download with: python download_model.py")
        print("   Or it will download automatically on first run (requires internet)")
        return False

    return True


def check_gpu_memory():
    """Check available GPU memory"""
    print_section("GPU MEMORY")

    try:
        import torch

        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            free = total_mem - reserved

            print(f"Total VRAM: {total_mem:.2f} GB")
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Reserved: {reserved:.2f} GB")
            print(f"Free: {free:.2f} GB")

            # Check if enough for competition
            if total_mem >= 15:
                print(f"\n✅ Sufficient VRAM for T4 (16GB)")
            elif total_mem >= 10:
                print(f"\n⚠️  Limited VRAM - may need to reduce batch size")
            else:
                print(f"\n❌ Insufficient VRAM - need 16GB T4 GPU")
                return False

            return True
        else:
            print("❌ No GPU available")
            return False

    except Exception as e:
        print(f"❌ Error checking GPU memory: {e}")
        return False


def test_inference():
    """Quick inference test"""
    print_section("INFERENCE TEST")

    try:
        print("Loading pipeline...")

        # Try to import and initialize
        from vllm_pipeline import WinningVLLMPipeline

        # Use small model for quick test
        pipeline = WinningVLLMPipeline(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            cache_dir=os.environ.get("MODEL_CACHE_DIR", "/app/models")
        )

        print("✅ Pipeline loaded successfully")

        # Test inference
        print("\nRunning test inference...")
        test_q = [{"questionID": "test", "question": "What is 2+2?"}]

        import time
        start = time.time()
        result = pipeline(test_q)
        elapsed = time.time() - start

        print(f"✅ Inference completed in {elapsed:.2f}s")
        print(f"   Answer: {result[0]['answer'][:100]}")

        return True

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_checks(skip_inference=False):
    """Run all diagnostic checks"""
    print("\n" + "=" * 80)
    print("🏥 TECH ARENA 2025 - SYSTEM DIAGNOSTICS")
    print("=" * 80)

    results = {}

    # Run all checks
    results["Python"] = check_python()
    results["CUDA"] = check_cuda()
    results["Dependencies"] = check_dependencies()
    results["vLLM"] = check_vllm_import()
    results["Model Cache"] = check_model_cache()
    results["GPU Memory"] = check_gpu_memory()

    if not skip_inference:
        results["Inference Test"] = test_inference()

    # Summary
    print("\n" + "=" * 80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 80)

    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")

    passed = sum(results.values())
    total = len(results)

    print(f"\nPassed: {passed}/{total} checks ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🏆 SYSTEM READY FOR COMPETITION!")
    elif passed >= total * 0.8:
        print("\n✅ System mostly ready - review warnings above")
    else:
        print("\n⚠️  SYSTEM NOT READY - Fix critical issues above")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="System diagnostics for Tech Arena 2025")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference test")
    args = parser.parse_args()

    run_all_checks(skip_inference=args.skip_inference)
