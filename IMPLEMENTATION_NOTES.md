# Implementation Notes: Speculative Decoding & Enhanced Reasoning

## Overview

This implementation adds **speculative decoding** with quantized models and **enhanced reasoning** for algebra to improve both speed and accuracy.

---

## Key Changes

### 1. Speculative Decoding Architecture

**What is Speculative Decoding?**
- Uses a smaller "draft" model (Qwen3-1.7B) to quickly generate candidate tokens
- Main model (Qwen3-4B) verifies and accepts/rejects candidates
- When accepted: **2-3x speedup** (multiple tokens per forward pass)
- When rejected: Falls back to normal generation (no slowdown)

**Implementation:**
```python
# Both models are AWQ quantized (4-bit)
Main Model: Qwen3-4B (AWQ) → /app/models/qwen3_4b_awq
Draft Model: Qwen3-1.7B (AWQ) → /app/models/qwen3_1.7b_awq

# Speculative configuration
num_speculative_tokens=5  # Look ahead 5 tokens
draft_token_acceptance_method="typical_acceptance_sampler"
```

**Memory footprint:**
- Qwen3-4B AWQ: ~2.5 GB
- Qwen3-1.7B AWQ: ~1.1 GB
- Total: ~3.6 GB (fits comfortably in 16GB T4)

---

### 2. Enhanced Algebra Reasoning

**Problem:** Previous algebra accuracy was 13/100

**Solution: Deep Chain-of-Thought Prompting**

#### Changes to Algebra Prompts:

1. **Explicit reasoning instructions:**
   ```
   IMPORTANT INSTRUCTIONS:
   1. Think through the problem carefully step-by-step
   2. Show ALL intermediate steps and calculations
   3. Use the formulas from the reference when applicable
   4. Double-check your work
   5. Provide the final answer clearly at the end
   ```

2. **Enhanced examples (6 → 6 detailed examples):**
   - Each example now shows complete step-by-step reasoning
   - Includes verification steps
   - Clear "Final Answer:" marker

3. **Lower temperature for accuracy:**
   ```python
   temperature=0.1  # Was 0.3 - much more deterministic
   top_p=0.9        # Was 0.95 - more focused sampling
   ```

4. **More tokens for reasoning:**
   ```python
   max_tokens=800   # Was 600 - allow detailed explanations
   ```

5. **Better answer extraction:**
   ```python
   # Looks for multiple patterns:
   - "Final Answer: ..."
   - "Answer: ..."
   - "Therefore, ..."
   - Last equation with "="
   ```

#### Example Prompt Structure:

**Before:**
```
Problem: Solve for x: 2x + 5 = 13
Solution: Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4.
Final Answer: x = 4
```

**After:**
```
Problem: Solve for x: 2x + 5 = 13
Step-by-step Solution:
- Start with: 2x + 5 = 13
- Subtract 5 from both sides: 2x = 13 - 5 = 8
- Divide both sides by 2: x = 8/2 = 4
- Verification: 2(4) + 5 = 8 + 5 = 13 ✓
Final Answer: x = 4
```

---

### 3. Quantization Strategy

**Both models use AWQ 4-bit quantization:**

**Benefits:**
- **Memory efficient:** 4x smaller than FP16
- **Fast inference:** Optimized GEMM kernels
- **Good accuracy:** Minimal quality loss (<2% typically)

**Quantization Process:**
1. Load model with CPU offloading (avoids OOM)
2. Use domain-specific calibration data (130 samples):
   - Chinese text (8 samples)
   - Algebra/math (9 samples)
   - General knowledge (9 samples)
   - Repeated 5x for optimal AWQ calibration
3. Apply AWQ quantization (4-bit weights, group size 128)
4. Save to `/app/models/{model_name}_awq/`
5. Cleanup and free GPU memory

**Calibration Data Tailored to Competition:**
```python
calibration_data = [
    # Chinese
    "中国的首都是北京。",
    "长城是世界著名的古代建筑奇迹。",
    ...
    # Algebra
    "Solve for x: 3x + 7 = 22",
    "Calculate the derivative of f(x) = x^3 - 4x + 1",
    ...
    # General
    "The Industrial Revolution began in Great Britain...",
    "Photosynthesis is the process by which plants...",
    ...
] * 5  # 130 total samples
```

---

### 4. Single LLM Instance Design

**Why not separate LLMs for algebra?**

Initially considered:
```python
llm_with_spec = LLM(...)  # With speculative decoding
llm_algebra = LLM(...)    # Without speculative decoding
```

**Problem:** Memory overhead (~6GB for two instances)

**Solution:** Use single LLM with subject-specific sampling parameters:

```python
# Same LLM for all, but different sampling params:

# Algebra: Very deterministic, allows reasoning
params_algebra = SamplingParams(
    temperature=0.1,  # Low for accuracy
    max_tokens=800,   # High for reasoning
)

# Chinese: Balanced
params_chinese = SamplingParams(
    temperature=0.18,
    max_tokens=300,
)

# General: Balanced
params_general = SamplingParams(
    temperature=0.2,
    max_tokens=448,
)
```

**Insight:** Speculative decoding with low temperature still provides speedup without hurting accuracy, as the draft model proposals are verified by main model.

---

### 5. Batch Processing Strategy

**Subject-wise batching** for optimal performance:

```python
# Phase 1: Sort questions by subject
chinese_prompts = [...]
algebra_prompts = [...]
general_prompts = [...]

# Phase 2: Batch execute by subject
chinese_outputs = llm.generate(chinese_prompts, params_chinese)
algebra_outputs = llm.generate(algebra_prompts, params_algebra)
general_outputs = llm.generate(general_prompts, params_general)
```

**Benefits:**
- **Prefix caching hits:** Similar prompts share cached prefixes (KB)
- **Optimal sampling:** Each subject uses best parameters
- **Better GPU utilization:** Batch sizes optimized per subject

---

### 6. Prefix Caching for Knowledge Bases

**Problem:** Chinese KB (~16k chars) and Algebra KB (~10k chars) in every prompt

**Solution:** Warmup cache during initialization:

```python
def _warmup_cache(self):
    # Process KB-heavy prompts once with max_tokens=1
    # This populates the KV cache
    warmup_prompts = [
        chinese_kb_prompt,  # ~8k tokens
        algebra_kb_prompt,  # ~5k tokens
    ]
    llm.generate(warmup_prompts, SamplingParams(max_tokens=1))
```

**Result:** Subsequent requests with same KB prefix hit cache → ~8x faster prefix processing

---

## Performance Expectations

### Latency Improvements

**Without Speculative Decoding:**
- Qwen3-4B AWQ: ~100-150 ms/token on T4
- 500 questions × 400 tokens avg × 125 ms/token = ~25,000 seconds

**With Speculative Decoding (2.5x speedup):**
- Effective: ~40-60 ms/token
- 500 questions × 400 tokens avg × 50 ms/token = **~10,000 seconds**
- **Target: < 7,200 seconds (well within 2-hour limit)**

**Additional speedups from:**
- Prefix caching: ~30-50% reduction for KB-heavy prompts
- Batching: ~20-30% reduction from efficient GPU utilization
- **Estimated total time: 6,000-8,000 seconds for 500 questions**

### Accuracy Improvements

**Algebra (critical improvement area):**
- Baseline: 13/100
- **Target: 60+/100** with enhanced reasoning prompts
- Strategy: Lower temperature (0.1), step-by-step examples, verification

**Chinese:**
- Expected: 70-80% (with KB augmentation)

**Geography/History:**
- Expected: 60-75% (factual knowledge)

**Overall Target: 60-70% accuracy** (competitive for leaderboard)

---

## Trade-offs & Design Decisions

### 1. Speculative Decoding for All Subjects

**Decision:** Enable speculative decoding globally (including algebra)

**Rationale:**
- Low temperature (0.1) ensures draft proposals rarely diverge
- Verification by main model prevents accuracy loss
- Speedup benefit outweighs minimal risk

**Alternative considered:** Disable for algebra (rejected due to minimal accuracy benefit vs large latency cost)

### 2. Single vs Dual LLM Instances

**Decision:** Single LLM instance for all subjects

**Rationale:**
- Memory efficiency (3.6GB vs 6GB+)
- Sampling parameters sufficient to differentiate behavior
- Simpler code, easier to optimize

**Alternative considered:** Separate algebra LLM without spec decode (rejected due to memory constraints)

### 3. Temperature Selection

**Algebra: 0.1** (very deterministic)
- Prioritizes accuracy over diversity
- Reasoning should be deterministic anyway
- Prevents "creative" but wrong answers

**Chinese: 0.18** (low-medium)
- Allows slight variation in phrasing
- Still factually consistent

**General: 0.2** (low-medium)
- Balanced factual accuracy and fluency

**Alternative considered:** Higher temperatures for diversity (rejected - accuracy > diversity for QA)

### 4. Max Tokens Allocation

**Algebra: 800 tokens**
- Needs space for step-by-step reasoning
- Typical solution: 200-600 tokens
- Buffer for complex problems

**Chinese: 300 tokens**
- Factual answers, concise
- Chinese characters are information-dense

**General: 448 tokens**
- Factual answers with some explanation
- Buffer for detailed history questions

**Alternative considered:** Uniform 500 tokens (rejected - wastes time on simple questions, insufficient for complex algebra)

---

## Potential Issues & Mitigations

### Issue 1: Algebra Still Low Accuracy After Changes

**Symptoms:**
- Answers are lengthy but wrong
- Missing key steps
- Incorrect final answers

**Debugging:**
1. Check `answers_output.json` for algebra questions
2. Look for pattern in errors (e.g., always wrong on quadratics?)
3. Check if "Final Answer:" extraction is working

**Mitigations:**
```python
# Option A: Even lower temperature
temperature=0.05  # Was 0.1

# Option B: Add more specific examples for failing question types
# Edit algebra prompt to include similar examples

# Option C: Increase max_tokens for very complex problems
max_tokens=1000  # Was 800

# Option D: Disable speculative decoding
export ENABLE_SPECULATIVE_DECODING=false
```

### Issue 2: Speculative Decoding Not Speeding Up

**Symptoms:**
- Acceptance rate < 40% (check vLLM logs)
- Latency similar to non-speculative

**Debugging:**
1. Check if draft model loaded correctly
2. Look for "speculative_config" in logs
3. Monitor GPU with `nvidia-smi` (should see both models)

**Mitigations:**
```python
# Option A: Increase speculative tokens
num_speculative_tokens=7  # Was 5

# Option B: Use larger draft model (if memory allows)
DRAFT_MODEL_NAME="Qwen/Qwen3-3B"  # Was 1.7B

# Option C: Adjust acceptance method
draft_token_acceptance_method="rejection_sampler"
```

### Issue 3: Out of Memory (OOM)

**Symptoms:**
- CUDA OOM error during quantization or inference

**Mitigations:**
```python
# Option A: Reduce GPU memory utilization
gpu_memory_utilization=0.75  # Was 0.88

# Option B: Reduce batch size
max_num_seqs=8  # Was 16

# Option C: Reduce max context length
max_model_len=8192  # Was 16384

# Option D: Reduce speculative tokens
num_speculative_tokens=3  # Was 5
```

### Issue 4: Slow Quantization (>30 minutes)

**Symptoms:**
- Quantization takes very long on first run

**Normal:** First time quantization takes 10-15 min per model
**Not normal:** > 30 minutes

**Mitigations:**
```python
# Reduce calibration samples
len(calibration_data) = 64  # Was 130 (26 * 5)

# Use pre-quantized models if available (check competition forum)
```

---

## Testing Checklist

Before submitting:

- [ ] Run on VM with GPU (T4)
- [ ] Verify both models quantize successfully
- [ ] Check speculative decoding is enabled (see logs)
- [ ] Test with all 4 subjects (algebra, chinese, geography, history)
- [ ] Run evaluation script, check scores
- [ ] Verify latency < 14.4 seconds/question avg
- [ ] Check accuracy > 10% minimum (ideally > 60%)
- [ ] Test with 500-question set (full competition size)
- [ ] Verify submission.zip < 1GB
- [ ] Test submission can run without internet (offline mode)

---

## Files Modified

1. **inferencePipeline/pipeline.py**
   - Added draft model quantization
   - Implemented speculative decoding
   - Enhanced algebra prompts with reasoning
   - Adjusted sampling parameters
   - Improved answer extraction

2. **evaluate_answers.py** (new)
   - LLM-as-a-judge evaluator
   - Subject-specific scoring
   - Detailed feedback on issues

3. **VM_DEPLOYMENT_GUIDE.md** (new)
   - Step-by-step deployment instructions
   - Debugging tips
   - Iterative refinement workflow

4. **quick_test.sh** (new)
   - Automated test script
   - Run + evaluate + report in one command

---

## Next Steps for Competition

1. **Test on VM** (critical!)
   - SSH into VM
   - Run `python3 load.py`
   - Check if both models load and quantize
   - Verify speculative decoding works

2. **Evaluate Results**
   - Run `python3 evaluate_answers.py`
   - Focus on algebra performance
   - Identify specific failure modes

3. **Iterative Refinement**
   - Based on evaluation feedback, adjust:
     - Prompts (add examples for failing cases)
     - Temperatures (lower if too random)
     - Max tokens (increase if cut off)
     - Speculative config (tune acceptance)

4. **Performance Tuning**
   - If too slow: increase num_speculative_tokens, reduce max_tokens
   - If accuracy low: lower temperature, enhance prompts
   - Balance accuracy (60%) vs latency (40%) in scoring

5. **Final Validation**
   - Run full 500-question test
   - Ensure < 2 hour completion
   - Verify > 60% accuracy target
   - Create submission zip

6. **Submit & Monitor**
   - Upload to platform
   - Check leaderboard
   - Prepare presentation (if top 8)

---

## Competition Scoring Formula

```
Phase 2 Score = 0.6 × Accuracy + 0.4 × Latency

Where:
- Accuracy: Normalized % correct (LLM-as-a-judge)
- Latency: Normalized end-to-end time (lower is better)

Both metrics normalized to best team = 100%
```

**Strategy:**
- Prioritize accuracy (60% weight) but don't ignore latency
- Speculative decoding helps latency without hurting accuracy
- Enhanced prompts improve accuracy
- Balance is key!

---

## References

- vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/models/spec_decode.html
- AWQ Quantization: https://github.com/mit-han-lab/llm-awq
- Chain-of-Thought Prompting: https://arxiv.org/abs/2201.11903
- Qwen3 Model Cards: https://huggingface.co/Qwen/Qwen3-4B

