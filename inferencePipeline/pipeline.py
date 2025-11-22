"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline with vLLM

OPTIMIZATIONS:
✅ vLLM for fast inference
✅ FP16 precision for T4 GPU
✅ Batched processing
✅ Python calculator for math
"""

import os
import re
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path


# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = "/app/models"


def find_model_path(model_name: str, cache_dir: str) -> str:
    """Find the actual snapshot path in HuggingFace cache"""
    cache_path = Path(cache_dir)
    
    # Convert model name to HF cache format: meta-llama/Llama-3.2-3B-Instruct -> models--meta-llama--Llama-3.2-3B-Instruct
    hf_cache_name = "models--" + model_name.replace("/", "--")
    model_cache = cache_path / hf_cache_name
    
    if not model_cache.exists():
        raise FileNotFoundError(f"Model cache not found at {model_cache}")
    
    # Find the snapshot directory (usually only one)
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Snapshots directory not found at {snapshots_dir}")
    
    # Get the latest (or only) snapshot
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    
    # Use the most recent snapshot
    snapshot = sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return str(snapshot)


RAW_MODEL_PATH = find_model_path(MODEL_NAME, CACHE_DIR)


class InferencePipeline:
    """
    Runtime-quantized inference pipeline for T4 GPU

    Strategy:
    1. Quantize model during untimed setup (loadPipeline)
    2. Batch ALL math questions together
    3. Batch ALL text questions together
    4. Maximum throughput with vLLM
    """

    def __init__(self):
        """
        Initialize pipeline with vLLM (FP16)
        Simplified: no quantization needed for 3B model on T4
        """

        print("🚀 Loading model with vLLM (FP16)...")
        self.llm = LLM(
            model=RAW_MODEL_PATH,             # Load directly from HF cache
            dtype="half",                     # FP16 for compute
            gpu_memory_utilization=0.90,      # Conservative for stability
            #quantization="bitsandbytes",           # Dynamic quantization
            max_model_len=2048,               # Reduced for speed/memory
            enforce_eager=False,              # Enable CUDA graphs for speed
            max_num_seqs=64,                  # Increased batch size
            max_num_batched_tokens=8192,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters
        self.params_math = SamplingParams(
            temperature=0.0,
            max_tokens=128,
            stop=["```", "\n\n", ";"],
        )

        self.params_text = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            stop=["<|eot_id|>", "\n\nQuestion:"],
        )

        print("✅ Pipeline ready for inference\n")

    def _safe_eval(self, code: str) -> str:
        """Execute Python math expression safely"""
        try:
            # Remove any non-math characters
            clean = re.sub(r"[^0-9\.\+\-\*\/\(\)\%\s]", "", code).strip()
            if not clean:
                return None

            # Execute safely
            result = eval(clean, {"__builtins__": None}, {})

            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                return f"{result:.6g}"
            return str(result)

        except:
            return None

    def _create_chat_prompt(self, question: str) -> str:
        """Create prompt using Llama chat template"""
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Main inference method with DOUBLE BATCHING

        Critical fix: Batch ALL math, then batch ALL text
        NO sequential processing!
        """

        if not questions:
            return []

        results = [None] * len(questions)

        # =====================================================================
        # PHASE 1: BATCH SORTING
        # =====================================================================

        math_indices = []
        math_prompts = []

        text_indices = []
        text_prompts = []

        for i, q in enumerate(questions):
            subject = q.get('subject', 'general')
            question_text = q['question'].lower()

            # Detect math questions
            is_math = (
                subject == 'algebra' or
                'calculate' in question_text or
                'solve' in question_text or
                re.search(r'\d+\s*[\+\-\*×÷\/]\s*\d+', question_text)
            )

            if is_math:
                math_indices.append(i)
                # Prompt for code generation
                prompt = f"""Convert to Python expression. Output ONLY code.

Q: What is 50 + 50?
A: 50 + 50

Q: {q['question']}
A: """
                math_prompts.append(prompt)
            else:
                text_indices.append(i)
                text_prompts.append(self._create_chat_prompt(q['question']))

        # =====================================================================
        # PHASE 2: BATCH EXECUTION (CRITICAL FIX!)
        # =====================================================================

        # Batch A: ALL math questions in ONE call
        if math_prompts:
            print(f"🧮 Processing {len(math_prompts)} math questions (batched)...")

            # CRITICAL: Single batch call for ALL math
            math_outputs = self.llm.generate(math_prompts, self.params_math, use_tqdm=False)

            for idx, output in zip(math_indices, math_outputs):
                code = output.outputs[0].text.strip()
                answer = self._safe_eval(code)

                if answer:
                    # Success! Use calculator result
                    results[idx] = {
                        "questionID": questions[idx]["questionID"],
                        "answer": answer
                    }
                else:
                    # Calculator failed, add to text batch
                    text_indices.append(idx)
                    text_prompts.append(self._create_chat_prompt(questions[idx]['question']))

        # Batch B: ALL text questions in ONE call
        if text_prompts:
            print(f"📖 Processing {len(text_prompts)} text questions (batched)...")

            # CRITICAL: Single batch call for ALL text
            text_outputs = self.llm.generate(text_prompts, self.params_text, use_tqdm=False)

            for idx, output in zip(text_indices, text_outputs):
                answer = output.outputs[0].text.strip()

                # Enforce 5000 char limit
                if len(answer) > 5000:
                    answer = answer[:5000].rsplit('. ', 1)[0] + '.'

                results[idx] = {
                    "questionID": questions[idx]["questionID"],
                    "answer": answer
                }

        print(f"✅ Completed {len(results)} questions\n")
        return results


def loadPipeline():
    """
    Entry point for evaluation system

    This function is called ONCE before timing starts.
    All quantization happens here (untimed).
    """
    return InferencePipeline()
