# VM Deployment and Testing Guide

## Current Implementation Status

### ✅ Completed Features
1. **Speculative Decoding**: Qwen3-4B (main) + Qwen3-1.7B (draft) both quantized with AWQ
2. **Enhanced Algebra Reasoning**: Step-by-step prompting with low temperature (0.1)
3. **Quantization**: Both models automatically quantized during setup
4. **Optimized Batching**: Subject-wise batching with prefix caching
5. **LLM-as-a-Judge**: Evaluation script for quality assessment

### 📋 Key Parameters
- **Qwen3-4B**: Main model, AWQ 4-bit quantized
- **Qwen3-1.7B**: Draft model, AWQ 4-bit quantized
- **Speculative Tokens**: 5 tokens lookahead
- **Algebra Temperature**: 0.1 (very deterministic for accuracy)
- **Algebra Max Tokens**: 800 (for detailed reasoning)
- **Chinese/General Temperature**: 0.18-0.2

---

## Step 1: Access the VM

```bash
chmod 600 user.pem
ssh -i user.pem user@63.35.184.44
```

---

## Step 2: Setup the Repository

```bash
# Navigate to your working directory
cd ~

# Clone or update your repository
git clone <your-repo-url> jaadu
# OR if already cloned:
cd jaadu
git fetch origin
git checkout speculative-decoding-implementation
git pull origin speculative-decoding-implementation

# Navigate to project directory
cd jaadu
```

---

## Step 3: Verify Models are Downloaded

```bash
# Check if models exist in cache
ls -la /app/models/

# You should see:
# - models--Qwen--Qwen3-4B/
# - models--Qwen--Qwen3-1.7B/
# - (possibly other models)

# If models are missing, download them:
python3 download_model.py
```

---

## Step 4: Install Dependencies

```bash
cd inferencePipeline

# Create virtual environment if needed
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# If autoawq fails, try:
pip install autoawq --no-build-isolation
```

---

## Step 5: Test the Pipeline Locally

```bash
# Make sure you're in the jaadu directory
cd ~/jaadu

# Run with sample questions
python3 load.py --name "speculative_test_1"

# This will:
# 1. Load the pipeline (untimed setup)
# 2. Quantize both Qwen3-4B and Qwen3-1.7B to AWQ (first time only, ~10-15 min total)
# 3. Load both models with speculative decoding
# 4. Process all questions
# 5. Save answers to answers_output.json
# 6. Log performance to experiments.csv
```

Expected output:
```
🚀 Loading Qwen3-4B with vLLM (awq)...
🚀 Preparing AWQ quantized model for Qwen/Qwen3-4B...
✅ AWQ quantized model found at /app/models/qwen3_4b_awq
🚀 Setting up speculative decoding with draft model: Qwen/Qwen3-1.7B
🚀 Preparing AWQ quantized draft model for Qwen/Qwen3-1.7B...
✅ Draft model quantized and ready: /app/models/qwen3_1.7b_awq
✅ Speculative decoding enabled for all subjects!
🔥 Warming up cache with Knowledge Bases...
✅ Cache warmed up!
✅ Pipeline ready for inference

🔢 Processing X Algebra questions (reasoning mode)...
📖 Processing X general questions...
🇨🇳 Processing X Chinese questions...
✅ Completed X questions
```

---

## Step 6: Evaluate the Results

```bash
# Run the LLM-as-a-judge evaluator
python3 evaluate_answers.py

# This will show:
# - Subject-wise performance scores
# - Overall average score
# - Detailed feedback on low-scoring answers
# - Recommendations for improvement
```

---

## Step 7: Iterative Refinement

Based on the evaluation feedback:

### If Algebra accuracy is low (< 13/100):

**Option A: Increase reasoning depth**
```python
# Edit inferencePipeline/pipeline.py
# Line ~343: Increase max_tokens
max_tokens=1000,  # Was 800

# Line ~341: Even lower temperature
temperature=0.05,  # Was 0.1
```

**Option B: Add more examples to prompt**
- Edit the algebra prompt section (~408-486) to add more diverse examples
- Focus on the types of questions that failed

**Option C: Disable speculative decoding completely**
```python
# Set environment variable before running:
export ENABLE_SPECULATIVE_DECODING=false
python3 load.py --name "no_spec_test"
```

### If Chinese/Geography/History accuracy is low:

**Option A: Adjust temperature**
```python
# Edit inferencePipeline/pipeline.py
# Line ~332: For Chinese
temperature=0.15,  # Was 0.18

# Line ~350: For general
temperature=0.15,  # Was 0.2
```

**Option B: Enhance knowledge bases**
- Edit `inferencePipeline/chinese_kb.txt` to add more relevant information
- Ensure knowledge bases are comprehensive

### If latency is too high:

**Option A: Reduce max_tokens**
```python
# Algebra: 800 → 600
# Chinese: 300 → 250
# General: 448 → 350
```

**Option B: Increase batch size**
```python
# Line ~284:
"max_num_seqs": 24,  # Was 16
```

**Option C: Increase speculative tokens**
```python
# Line ~298:
num_speculative_tokens=7,  # Was 5
```

---

## Step 8: Re-test After Changes

```bash
# Run again with a different name to track experiments
python3 load.py --name "iteration_2_temp_adjustment"

# Evaluate
python3 evaluate_answers.py

# Check experiments log
cat experiments.csv
```

---

## Step 9: Prepare Submission

Once you're satisfied with accuracy and latency:

```bash
# Navigate to project root
cd ~/jaadu

# Create submission package
zip -r submission.zip inferencePipeline/ -x "*.pyc" -x "__pycache__/*" -x "*.git/*" -x ".venv/*"

# Verify size (must be < 1GB)
ls -lh submission.zip

# Download to your local machine (from your local terminal):
scp -i user.pem user@63.35.184.44:~/jaadu/submission.zip ./
```

---

## Debugging Tips

### Check GPU usage:
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```

### Check memory:
```bash
free -h
htop
```

### View logs in detail:
```bash
python3 load.py --name "debug_run" 2>&1 | tee debug.log
```

### Test quantization only:
```python
# In Python shell:
from inferencePipeline.pipeline import quantize_model_awq, find_model_path

model_path = find_model_path("Qwen/Qwen3-4B", "/app/models")
quant_path = quantize_model_awq(model_path, "/app/models", "Qwen/Qwen3-4B")
print(f"Quantized model at: {quant_path}")
```

### If you get OOM errors:
```python
# Reduce GPU memory utilization in pipeline.py line ~281:
"gpu_memory_utilization": 0.75,  # Was 0.88
```

---

## Quick Reference: File Locations

- **Main pipeline**: `inferencePipeline/pipeline.py`
- **Entry point**: `inferencePipeline/__init__.py`
- **Test script**: `load.py`
- **Evaluation**: `evaluate_answers.py`
- **Dependencies**: `inferencePipeline/requirements.txt`
- **Sample questions**: `sample_questions.xlsx`
- **Output**: `answers_output.json`
- **Metrics**: `experiments.csv`
- **Chinese KB**: `inferencePipeline/chinese_kb.txt`
- **Algebra KB**: `inferencePipeline/algebra_kb.txt`

---

## Expected Performance Targets

### Latency (500 questions, 2-hour limit):
- **Target**: < 7200 seconds total
- **Per question**: < 14.4 seconds average
- **With speculative decoding**: ~2-3x speedup expected

### Accuracy:
- **Minimum**: 10% (to enter leaderboard)
- **Target**: > 60% (competitive)
- **Algebra focus**: Improve from 13/100 baseline

### Current Issues to Address:
1. **Algebra accuracy**: 13/100 → need significant improvement
2. **Reasoning**: Ensure step-by-step solutions are correct
3. **Answer extraction**: Verify "Final Answer:" is extracted properly

---

## Environment Variables

```bash
# Disable speculative decoding
export ENABLE_SPECULATIVE_DECODING=false

# Use different model (not recommended per requirements)
export MODEL_NAME="Qwen/Qwen3-4B"
export DRAFT_MODEL_NAME="Qwen/Qwen3-1.7B"

# Offline mode (should be default on VM)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## Next Steps

1. ✅ Test basic functionality on VM
2. ✅ Run evaluation and identify weak areas
3. 🔄 Iteratively refine prompts and parameters
4. 🔄 Focus on improving algebra accuracy (critical!)
5. 🔄 Balance accuracy vs latency
6. ✅ Create final submission zip
7. ✅ Upload to competition platform

Good luck! 🚀
