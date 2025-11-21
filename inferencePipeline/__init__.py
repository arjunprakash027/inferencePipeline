"""
Ultra-Optimized Inference Pipeline for Tech Arena 2025
Entry point and shared utilities
"""

import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def find_snapshot_path(hf_cache_dir: str) -> Path:
    """Find the actual model snapshot path in HuggingFace cache"""
    cache_path = Path(hf_cache_dir)
    snapshots_dir = cache_path / "snapshots"
    
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            snapshot = sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            print(f"[SNAPSHOT] Found: {snapshot}")
            return snapshot
    
    print(f"[SNAPSHOT] Using direct path: {cache_path}")
    return cache_path


def convert_to_gguf(hf_model_path: str, output_path: str, target_q: str = "q4_1") -> bool:
    """
    Convert HuggingFace model -> f16 GGUF -> quantize to target quantization
    
    Args:
        hf_model_path: Path to HuggingFace model cache
        output_path: Output path for quantized GGUF
        target_q: Target quantization (q4_0, q4_1, q8_0, etc.)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        model_path = find_snapshot_path(hf_model_path)
        tmp_f16 = Path(output_path).with_suffix(".f16.gguf")

        # Step 1: Convert to f16
        print(f"[CONVERT] Step 1: Converting HF -> f16 GGUF")
        
        # Find conversion script
        base_dir = Path(__file__).parent
        possible_scripts = [
            base_dir / "llama.cpp" / "convert_hf_to_gguf.py",
            base_dir / "llama.cpp" / "convert.py",
        ]
        convert_script = next((p for p in possible_scripts if p.exists()), None)
        if convert_script is None:
            raise FileNotFoundError("convert_hf_to_gguf.py not found in llama.cpp")

        cmd_f16 = [
            sys.executable,
            str(convert_script),
            str(model_path),
            "--outfile", str(tmp_f16),
            "--outtype", "f16"
        ]
        subprocess.run(cmd_f16, check=True)
        print(f"[CONVERT] ✓ f16 GGUF created: {tmp_f16}")

        # Step 2: Detect OS and find quantizer binary
        current_os = platform.system().lower()
        if "darwin" in current_os:
            bin_dir = "bin_mac"
        elif "linux" in current_os:
            bin_dir = "bin_linux"
        else:
            raise RuntimeError(f"Unsupported OS detected: {current_os}")

        quant_bin = base_dir / "llama.cpp" / bin_dir / "llama-quantize"
        if not quant_bin.exists():
            raise FileNotFoundError(f"Quantizer not found at {quant_bin}")

        # Ensure binary is executable
        import os
        if not os.access(quant_bin, os.X_OK):
            os.chmod(quant_bin, 0o755)

        # Step 3: Quantize
        cmd_quant = [
            str(quant_bin),
            str(tmp_f16),
            str(output_path),
            target_q
        ]
        print(f"[CONVERT] Step 2: Running quantizer ({current_os}) → {target_q}")
        subprocess.run(cmd_quant, check=True)
        print(f"[CONVERT] ✓ Quantization to {target_q} complete")

        return True

    except subprocess.CalledProcessError as e:
        print(f"[CONVERT] ✗ Subprocess failed: {e}")
        print(f"stderr:\n{getattr(e, 'stderr', '')}")
        return False
    except Exception as e:
        print(f"[CONVERT] ✗ Unexpected error: {e}")
        return False


def ensure_gguf_models_exist():
    """
    Ensure GGUF models exist for server use.
    Converts from HuggingFace if needed.
    """
    cache_dir = './app/model'
    gguf_cache = Path("./gguf_cache")
    gguf_cache.mkdir(exist_ok=True)
    
    # Models to ensure exist
    models_to_check = [
        ("model_q8.gguf", "q8_0"),
        ("model_q4_0_draft.gguf", "q4_0"),
    ]
    
    for model_name, quant in models_to_check:
        model_path = gguf_cache / model_name
        if not model_path.exists():
            print(f"[GGUF] {model_name} not found, converting (one-time, 2-5 min)...")
            
            hf_path = Path(cache_dir) / "models--meta-llama--Llama-3.2-1B-Instruct"
            if not hf_path.exists():
                hf_path = Path(cache_dir)
            
            success = convert_to_gguf(str(hf_path), str(model_path), target_q=quant)
            if success:
                print(f"[GGUF] ✓ {model_name} created successfully")
            else:
                print(f"[GGUF] ✗ Failed to create {model_name}")
        else:
            print(f"[GGUF] ✓ Found cached {model_name}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def loadPipeline(method: str = "server"):
    """
    Factory function - Routes to the appropriate pipeline implementation
    
    Args:
        method: "server" (default, fast) or "local" (transformers fallback)
    
    Returns:
        Ready-to-use pipeline instance
    """
    # Ensure GGUF models exist (for server use)
    ensure_gguf_models_exist()
    
    if method == "server":
        print("[FACTORY] Initializing Server Pipeline (fast mode)...")
        from .server_pipeline import loadServerPipeline
        return loadServerPipeline()
    else:
        print("[FACTORY] Initializing Local Pipeline (transformers fallback)...")
        from .local_pipeline import loadLocalPipeline
        return loadLocalPipeline()


__all__ = ['loadPipeline', 'convert_to_gguf', 'ensure_gguf_models_exist']
