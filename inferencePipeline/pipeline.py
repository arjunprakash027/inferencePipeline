"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline with vLLM

OPTIMIZATIONS:
✅ vLLM for fast inference with Qwen 4B
✅ FP16 precision for T4 GPU
✅ Batched processing by subject
✅ Few-shot prompting for Chinese and Algebra
✅ Answer extraction (returns only final answers, not reasoning)
"""

import os
import re
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path


# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"
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
            gpu_memory_utilization=0.93,      # Slightly higher for better throughput
            max_model_len=1792,               # Reduced for faster processing (still plenty of room)
            enforce_eager=False,              # Enable CUDA graphs for speed
            max_num_seqs=64,                  # Increased batch size
            max_num_batched_tokens=8192,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters with reasoning enabled
        self.params_reasoning = SamplingParams(
            temperature=0.25,  # Slightly lower for efficiency, still allows creativity
            top_p=0.9,
            max_tokens=768,  # Reduced but still allows thinking space
            stop=["<|im_end|>", "\n\nQuestion:", "\n\n\n"],  # Added triple newline to catch end of thought
            skip_special_tokens=True,  # Filter special tokens during generation
        )

        # Sampling parameters without reasoning
        self.params_no_reasoning = SamplingParams(
            temperature=0.25,  # Slightly lower for efficiency
            top_p=0.9,
            max_tokens=400,  # Reduced but enough for complete answers
            stop=["<|im_end|>", "\n\nQuestion:", "\n\n"],  # Added double newline stop
            skip_special_tokens=True,  # Filter special tokens during generation
        )

        print("✅ Pipeline ready for inference\n")

    def _strip_thinking(self, text: str) -> str:
        """Remove <think> tags and their content from the output"""
        import re
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove standalone <think> or </think> tags
        text = re.sub(r'</?think>', '', text)
        return text.strip()


    def _create_chat_prompt(self, question: str, subject: str = "general") -> str:
        """Create prompt using Qwen chat template with few-shot examples"""

        if subject == "chinese":
            # Few-shot prompting for Chinese questions
            prompt = f"""Answer the Chinese question clearly and directly.

Example:
Question: 下列词语中，加点字的读音完全正确的一项是？
Answer: 选项B的读音完全正确。

Question: {question}
Answer:"""

        elif subject == "algebra":
            # Few-shot prompting for Algebra questions
            prompt = f"""Solve the math problem step by step, then provide the final answer.

Example:
Question: Solve for x: 2x + 5 = 13
Answer: x = 4

Question: {question}
Answer:"""

        else:
            # Standard prompt for other subjects
            prompt = question

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_final_answer(self, text: str, subject: str) -> str:
        """Extract only the final answer from reasoning output"""

        # For Chinese and Algebra, try to extract the final answer
        if subject in ['chinese', 'algebra']:
            # Look for patterns like "Answer: ..." or "答案：..." or "Final answer: ..."
            answer_patterns = [
                r'(?:Final [Aa]nswer|ANSWER|Answer):\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)',
                r'(?:答案|最终答案)[:：]\s*(.+?)(?:\n\n|\n|$)',
                r'(?:Therefore|Thus|So),?\s+(.+?)(?:\n\n|\n|$)',
                r'(?:结论|因此)[:：,，]\s*(.+?)(?:\n\n|\n|$)',
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    # Clean up the answer
                    answer = re.sub(r'\n+', ' ', answer)
                    return answer

            # If no pattern found, try to get the last substantive line
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                # Return the last line as the answer
                return lines[-1]

        # For other subjects or if extraction fails, return the full text
        return text

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
        reasoning_subjects = []

        no_reasoning_indices = []
        no_reasoning_prompts = []

        for i, q in enumerate(questions):
            subject = q.get('subject', 'general').lower()

            # Enable reasoning for Chinese and Algebra
            needs_reasoning = subject in ['chinese', 'algebra']

            if needs_reasoning:
                reasoning_indices.append(i)
                reasoning_prompts.append(self._create_chat_prompt(q['question'], subject=subject))
                reasoning_subjects.append(subject)
            else:
                no_reasoning_indices.append(i)
                no_reasoning_prompts.append(self._create_chat_prompt(q['question'], subject=subject))

        # =====================================================================
        # PHASE 2: BATCH EXECUTION
        # =====================================================================

        # Batch A: Questions WITH reasoning (Chinese, Algebra)
        if reasoning_prompts:
            print(f"🧠 Processing {len(reasoning_prompts)} questions with reasoning (batched)...")

            reasoning_outputs = self.llm.generate(reasoning_prompts, self.params_reasoning, use_tqdm=False)

            for idx, output, subject in zip(reasoning_indices, reasoning_outputs, reasoning_subjects):
                raw_answer = output.outputs[0].text.strip()
                
                # Remove thinking tags first
                raw_answer = self._strip_thinking(raw_answer)

                # Extract final answer (remove reasoning steps)
                answer = self._extract_final_answer(raw_answer, subject)

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
                raw_answer = output.outputs[0].text.strip()
                
                # Remove thinking tags
                answer = self._strip_thinking(raw_answer)

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
