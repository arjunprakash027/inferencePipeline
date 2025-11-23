"""
Ultra-Optimized Inference Pipeline for Tech Arena 2025
Entry point and shared utilities
"""

import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Optional


# ============================================================================
# SETTINGS LOADER (Loads once at module import time - before timing starts)
# ============================================================================

_SETTINGS = None

def get_settings():
    """Load settings from settings.json (cached after first load)"""
    global _SETTINGS
    if _SETTINGS is None:
        settings_path = Path(__file__).parent / "settings.json"
        with open(settings_path, 'r') as f:
            _SETTINGS = json.load(f)
    return _SETTINGS


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
    Converts from HuggingFace if needed, using settings.json configuration.
    """
    settings = get_settings()
    model_cfg = settings['model']
    gguf_cfg = settings['gguf_conversion']
    
    cache_dir = model_cfg['cache_dir']
    gguf_cache = Path(model_cfg['gguf_cache_dir'])
    gguf_cache.mkdir(exist_ok=True)
    
    # Get model name and construct HF cache path
    model_name = model_cfg['name']
    # Convert model name to HF cache format: "meta-llama/Llama-3.2-1B" -> "models--meta-llama--Llama-3.2-1B"
    hf_cache_name = "models--" + model_name.replace("/", "--")
    hf_path = Path(cache_dir) / hf_cache_name
    
    # Fallback to cache_dir if specific model path doesn't exist
    if not hf_path.exists():
        print(f"[GGUF] Looking for model in {cache_dir}")
        hf_path = Path(cache_dir)
    
    # Models to check based on settings
    if 'output_filename' in gguf_cfg:
        # Simplified: single model conversion
        model_name = gguf_cfg['output_filename']
        quant = gguf_cfg.get('quantization', 'q4_0')
        models_to_check = [(model_name, quant)]
    elif 'target_filenames' in gguf_cfg:
        # Legacy: multiple models (draft/main)
        models_to_check = []
        for key, filename in gguf_cfg['target_filenames'].items():
            quant = gguf_cfg['target_quantizations'].get(key, 'q4_0')
            models_to_check.append((filename, quant))
    else:
        # Fallback to default naming
        models_to_check = []
        for key, quant in gguf_cfg.get('target_quantizations', {}).items():
            filename = f"model_{quant}.gguf"
            models_to_check.append((filename, quant))
    
    # Convert each model if it doesn't exist
    for model_name, quant in models_to_check:
        model_path = gguf_cache / model_name
        if not model_path.exists():
            print(f"[GGUF] {model_name} not found, converting (one-time, 2-5 min)...")
            
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
    from .llama_cpp_pipeline import loadLlamaCppPipeline
    
    return loadLlamaCppPipeline()
    
    # if method == "server":
    #     print("[FACTORY] Initializing Server Pipeline (fast mode)...")
    #     from .server_pipeline import loadServerPipeline
    #     return loadServerPipeline()
    # else:
    #     print("[FACTORY] Initializing Local Pipeline (transformers fallback)...")
    #     from .local_pipeline import loadLocalPipeline
    #     return loadLocalPipeline()


__all__ = ['loadPipeline', 'convert_to_gguf', 'ensure_gguf_models_exist']
