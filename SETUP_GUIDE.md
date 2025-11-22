# Tech Arena 2025 - vLLM Pipeline Setup Guide

Complete guide to set up, test, and optimize the winning vLLM inference pipeline on T4 GPU.

---

## 🎯 Quick Start (5 minutes)

### Prerequisites
- **GPU**: Tesla T4 (16GB VRAM) with CUDA 11.8 or 12.1
- **Python**: 3.10 or 3.11 (recommended)
- **System**: Ubuntu 20.04+ or similar Linux
- **RAM**: 32GB+ recommended

### Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install PyTorch with CUDA support (CRITICAL!)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# 3. Install vLLM and dependencies
pip install -r requirements_vllm.txt

# 4. Verify CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# Should print: CUDA available: True
```

---

## 📥 Download Model

```bash
# Create download script
cat > download_model.py << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set cache directory
cache_dir = "/app/models"  # Or your preferred path
os.makedirs(cache_dir, exist_ok=True)

model_name = "Qwen/Qwen2.5-3B-Instruct"

print(f"Downloading {model_name}...")
print("This will take 5-10 minutes depending on your connection...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True
)

# Download model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype="float16",
    trust_remote_code=True
)

print(f"✅ Model downloaded to {cache_dir}")
print(f"Total size: ~6GB")
EOF

# Run download
python download_model.py
```

**Alternative models to try:**
```python
# For higher accuracy (slower):
model_name = "Qwen/Qwen2.5-7B-Instruct"  # ~14GB

# For maximum speed (lower accuracy):
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # ~3GB

# Llama alternative:
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # ~6GB
```

---

## 🧪 Testing Pipeline

### Basic Functionality Test

```bash
# Run comprehensive test suite
python test_vllm_pipeline.py
```

This will test:
- ✅ Pipeline loading
- ✅ Subject detection
- ✅ Batch processing
- ✅ Answer quality
- ✅ Performance targets
- ✅ Error handling

**Expected output:**
```
🧪 VLLM PIPELINE TEST SUITE
================================================================================

TEST 1: BASIC FUNCTIONALITY
✅ Pipeline loaded successfully
✅ Single question test passed
✅ Empty input test passed

...

📊 TEST SUMMARY
✅ PASS - Basic Functionality
✅ PASS - Subject Detection
✅ PASS - Batch Processing
✅ PASS - Answer Quality
✅ PASS - Performance Targets
✅ PASS - Error Handling

Total: 6/6 tests passed (100.0%)
🏆 ALL TESTS PASSED - PIPELINE READY FOR COMPETITION!
```

### Performance Benchmark

```bash
# Run full benchmark (500 questions)
python benchmark_vllm.py
```

**What it measures:**
- ⏱️ Total latency for 500 questions
- 📊 Throughput (questions/sec)
- 💾 GPU memory usage
- 📏 Answer length distribution
- 🎯 Subject-specific performance
- 🏆 Estimated competition score

**Target metrics:**
- **Total time**: < 90s (competitive), < 60s (top tier)
- **Throughput**: > 5.5 Q/s (competitive), > 8 Q/s (top tier)
- **Peak memory**: < 12GB (safe for T4 16GB)
- **Score**: > 70% (qualifies for presentation)

---

## 🔧 Optimization Guide

### 1. FP8 Quantization (Default)

The pipeline uses **FP8 dynamic quantization** by default:

```python
# In vllm_pipeline.py
self.llm = LLM(
    quantization="fp8",  # FP8 quantization
    kv_cache_dtype="fp8",  # FP8 KV cache
)
```

**Benefits:**
- 2x memory reduction vs FP16
- 1.5-2x faster inference
- ~1-2% accuracy loss (acceptable)

**If FP8 fails** (older CUDA), it automatically falls back to FP16.

### 2. Batch Size Tuning

Adjust `max_num_seqs` based on your VRAM:

```python
# For 16GB T4 (default)
max_num_seqs=32  # Aggressive batching

# If OOM errors occur
max_num_seqs=24  # More conservative

# For 8GB GPU
max_num_seqs=16  # Minimal batching
```

### 3. Model Selection

Trade-off between speed and accuracy:

```python
# BALANCED (recommended)
model_name = "Qwen/Qwen2.5-3B-Instruct"
# Speed: ★★★★☆  Accuracy: ★★★★☆

# MAXIMUM SPEED
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# Speed: ★★★★★  Accuracy: ★★★☆☆

# MAXIMUM ACCURACY
model_name = "Qwen/Qwen2.5-7B-Instruct"
# Speed: ★★★☆☆  Accuracy: ★★★★★
```

### 4. Token Limits

Reduce `max_tokens` for faster inference:

```python
# Current settings (in _setup_sampling_params)
self.base_sampling = SamplingParams(
    max_tokens=200,  # Default
)

# For speed optimization
self.base_sampling = SamplingParams(
    max_tokens=150,  # Faster, shorter answers
)

# For accuracy
self.base_sampling = SamplingParams(
    max_tokens=300,  # Slower, more detailed
)
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce `gpu_memory_utilization`:
   ```python
   gpu_memory_utilization=0.85  # From 0.92
   ```

2. Reduce batch size:
   ```python
   max_num_seqs=16  # From 32
   ```

3. Reduce context length:
   ```python
   max_model_len=1536  # From 2048
   ```

4. Switch to smaller model:
   ```python
   model_name = "Qwen/Qwen2.5-1.5B-Instruct"
   ```

### Issue: "vLLM import error"

**Solution:**
```bash
# Reinstall vLLM with CUDA support
pip uninstall vllm -y
pip install vllm==0.6.3.post1 --no-cache-dir

# Verify CUDA
nvcc --version  # Should show CUDA 11.8 or 12.1
```

### Issue: "FP8 not supported"

**Solution:**
The pipeline automatically falls back to FP16. To force FP16:

```python
# In vllm_pipeline.py, remove FP8 lines:
self.llm = LLM(
    model=self.model_name,
    # quantization="fp8",  # Remove this
    # kv_cache_dtype="fp8",  # Remove this
    dtype="float16",  # Keep this
    ...
)
```

### Issue: Slow performance

**Diagnostic steps:**

1. **Check GPU utilization:**
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   # Should show 90-100% GPU utilization during inference
   ```

2. **Profile with benchmark:**
   ```bash
   python benchmark_vllm.py
   # Check "Throughput" metric
   ```

3. **Common fixes:**
   - Increase batch size if GPU not fully utilized
   - Reduce max_tokens to generate shorter answers
   - Enable CUDA graphs (should be default)
   - Check if warmup completed (should happen automatically)

### Issue: Poor accuracy

**Solutions:**

1. **Switch to better model:**
   ```python
   model_name = "Qwen/Qwen2.5-7B-Instruct"  # Higher accuracy
   ```

2. **Adjust temperature:**
   ```python
   # In _setup_sampling_params
   temperature=0.1  # More deterministic (from 0.3)
   ```

3. **Improve prompts:**
   - Add few-shot examples (already included)
   - Adjust system prompts in `_create_prompt()`

---

## 📊 Testing on Real Data

### Prepare Test Dataset

```python
# Create test_data.json
cat > test_data.json << 'EOF'
[
  {
    "questionID": "1",
    "question": "What is the capital of France?",
    "subject": "geography"
  },
  {
    "questionID": "2",
    "question": "Solve for x: 2x + 5 = 15",
    "subject": "algebra"
  }
]
EOF
```

### Run Inference

```python
# test_real_data.py
import json
from vllm_pipeline import loadPipeline

# Load questions
with open("test_data.json", "r") as f:
    questions = json.load(f)

# Load pipeline
pipeline = loadPipeline()

# Run inference
results = pipeline(questions)

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Processed {len(results)} questions")
print(f"💾 Results saved to results.json")
```

---

## 🏆 Competition Submission Checklist

Before submitting:

- [ ] **All tests pass**: `python test_vllm_pipeline.py`
- [ ] **Benchmark under 120s**: `python benchmark_vllm.py`
- [ ] **Model downloaded**: Check `/app/models` directory
- [ ] **Dependencies complete**: All packages in `requirements_vllm.txt`
- [ ] **Error handling tested**: Try edge cases
- [ ] **Memory usage safe**: < 14GB peak on T4
- [ ] **Offline mode works**: Set `HF_HUB_OFFLINE=1`

### Final Validation

```bash
# Test offline mode (competition environment)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run full test
python test_vllm_pipeline.py

# Should complete without internet access
```

---

## 📈 Expected Performance

### On Tesla T4 (16GB)

| Metric | Expected Value |
|--------|---------------|
| **Load time** | 15-30s (one-time) |
| **500Q latency** | 60-90s |
| **Throughput** | 5.5-8.3 Q/s |
| **Peak VRAM** | 8-12GB |
| **Accuracy** | 75-82% |
| **Final score** | 70-78% |

### Competitive Tiers

- 🥇 **Top 3**: < 60s latency, > 80% accuracy, > 75% score
- 🥈 **Top 5**: < 90s latency, > 75% accuracy, > 70% score
- ✅ **Qualifies**: < 120s latency, > 65% accuracy, > 60% score

---

## 🔬 Advanced: Model Comparison

Test different models on your hardware:

```bash
# Test all models and compare
python - << 'EOF'
from vllm_pipeline import WinningVLLMPipeline
import time

models = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

test_q = [{"questionID": "1", "question": "What is 2+2?"}] * 100

for model_name in models:
    print(f"\nTesting {model_name}...")
    pipeline = WinningVLLMPipeline(model_name=model_name)

    start = time.time()
    _ = pipeline(test_q)
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {len(test_q)/elapsed:.2f} Q/s")
EOF
```

---

## 💡 Tips for Winning

1. **Optimize for the 60/40 split**: Balance accuracy and speed
   - Don't sacrifice 10% accuracy for 5% speed gain
   - But don't optimize accuracy if you're over time limit

2. **Test on actual T4 hardware**: Performance varies by GPU
   - Cloud options: AWS g4dn.xlarge, GCP T4
   - Colab: T4 available on free tier

3. **Monitor during evaluation**:
   ```bash
   watch -n 1 nvidia-smi  # Monitor GPU usage
   ```

4. **Have fallback ready**: If FP8 fails, FP16 is automatic

5. **Conservative estimates**: Target 75% accuracy (achievable) not 90% (risky)

---

## 📞 Support

If you encounter issues:

1. Check `test_results.json` for detailed test output
2. Check `benchmark_results.json` for performance metrics
3. Run with debug logging:
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   python test_vllm_pipeline.py
   ```

---

## 🚀 Final Deployment

For competition submission:

```bash
# 1. Create submission package
mkdir -p submission/inferencePipeline
cp vllm_pipeline.py submission/inferencePipeline/
cp requirements_vllm.txt submission/requirements.txt

# 2. Add __init__.py entry point
cat > submission/inferencePipeline/__init__.py << 'EOF'
from .vllm_pipeline import loadPipeline
__all__ = ['loadPipeline']
EOF

# 3. Test submission package
cd submission
python -c "from inferencePipeline import loadPipeline; p = loadPipeline()"

# 4. Create tarball (if required)
tar -czf submission.tar.gz inferencePipeline/ requirements.txt
```

---

**Good luck! 🏆**
