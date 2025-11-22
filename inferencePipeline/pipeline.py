"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline with vLLM

OPTIMIZATIONS:
✅ vLLM for fast inference
✅ FP16 precision for T4 GPU
✅ Batched processing
✅ Reasoning enabled for Chinese and Algebra
"""

import os
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path


# Configuration
MODEL_NAME = "Qwen/Qwen2.5-4B-Instruct"
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
    2. Batch questions by subject with reasoning control
    3. Enable reasoning for Chinese and Algebra
    4. Maximum throughput with vLLM
    """

    def __init__(self):
        """
        Initialize pipeline with vLLM (FP16)
        Simplified: no quantization needed for 4B model on T4
        """

        print("🚀 Loading Qwen 4B model with vLLM (FP16)...")
        self.llm = LLM(
            model=RAW_MODEL_PATH,             # Load directly from HF cache
            dtype="half",                     # FP16 for compute
            gpu_memory_utilization=0.90,      # Conservative for stability
            max_model_len=2048,               # Reduced for speed/memory
            enforce_eager=False,              # Enable CUDA graphs for speed
            max_num_seqs=64,                  # Increased batch size
            max_num_batched_tokens=8192,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters with reasoning enabled
        self.params_reasoning = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024,
            stop=["<|im_end|>", "\n\nQuestion:"],
        )

        # Sampling parameters without reasoning
        self.params_no_reasoning = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            stop=["<|im_end|>", "\n\nQuestion:"],
        )

        print("✅ Pipeline ready for inference\n")

    def _create_chat_prompt(self, question: str, enable_reasoning: bool = False) -> str:
        """Create prompt using Qwen chat template with optional reasoning"""
        if enable_reasoning:
            # Add reasoning instruction for Chinese and Algebra
            prompt = f"Think step-by-step and provide a detailed answer.\n\n{question}"
        else:
            prompt = question

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Main inference method with reasoning-based batching

        Reasoning enabled for: Chinese, Algebra
        No reasoning for: History, Geography
        """

        if not questions:
            return []

        results = [None] * len(questions)

        # =====================================================================
        # PHASE 1: BATCH SORTING BY REASONING REQUIREMENT
        # =====================================================================

        reasoning_indices = []
        reasoning_prompts = []

        no_reasoning_indices = []
        no_reasoning_prompts = []

        for i, q in enumerate(questions):
            subject = q.get('subject', 'general').lower()

            # Enable reasoning for Chinese and Algebra
            needs_reasoning = subject in ['chinese', 'algebra']

            if needs_reasoning:
                reasoning_indices.append(i)
                reasoning_prompts.append(self._create_chat_prompt(q['question'], enable_reasoning=True))
            else:
                no_reasoning_indices.append(i)
                no_reasoning_prompts.append(self._create_chat_prompt(q['question'], enable_reasoning=False))

        # =====================================================================
        # PHASE 2: BATCH EXECUTION
        # =====================================================================

        # Batch A: Questions WITH reasoning (Chinese, Algebra)
        if reasoning_prompts:
            print(f"🧠 Processing {len(reasoning_prompts)} questions with reasoning (batched)...")

            reasoning_outputs = self.llm.generate(reasoning_prompts, self.params_reasoning, use_tqdm=False)

            for idx, output in zip(reasoning_indices, reasoning_outputs):
                answer = output.outputs[0].text.strip()

                # Enforce 5000 char limit
                if len(answer) > 5000:
                    answer = answer[:5000].rsplit('. ', 1)[0] + '.'

                results[idx] = {
                    "questionID": questions[idx]["questionID"],
                    "answer": answer
                }

        # Batch B: Questions WITHOUT reasoning (History, Geography)
        if no_reasoning_prompts:
            print(f"📖 Processing {len(no_reasoning_prompts)} questions without reasoning (batched)...")

            no_reasoning_outputs = self.llm.generate(no_reasoning_prompts, self.params_no_reasoning, use_tqdm=False)

            for idx, output in zip(no_reasoning_indices, no_reasoning_outputs):
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
