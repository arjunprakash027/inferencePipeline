"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline with vLLM

OPTIMIZATIONS:
✅ vLLM for fast inference with Qwen 4B
✅ FP16 precision for T4 GPU (16GB)
✅ Memory-optimized batch processing for T4 constraints
✅ Batched processing by subject
✅ Few-shot prompting for Chinese and Algebra
✅ Answer extraction (returns only final answers, not reasoning)
✅ Prefix caching for few-shot examples
✅ CPU swap space for memory overflow handling
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

        # Optimized configuration for T4 16GB GPU (Log 3 working config)
        print("🚀 Loading Qwen 4B model with vLLM (T4 optimized)...")

        self.llm = LLM(
            model=RAW_MODEL_PATH,             # Load directly from HF cache
            dtype="half",                     # FP16 for compute
            gpu_memory_utilization=0.90,      # Higher utilization for reasoning tasks
            max_model_len=2048,               # Longer context for reasoning support
            enforce_eager=False,              # Enable CUDA graphs for speed
            max_num_seqs=28,                  # Increased (can handle 36x)
            max_num_batched_tokens=3584,      # Increased throughput
            enable_prefix_caching=False,      # Disabled (Log 3 working config)
            trust_remote_code=True,
            tensor_parallel_size=1,
            swap_space=4,                     # CPU swap space for overflow
            disable_log_stats=True,           # Reduce logging overhead
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters for Chinese (optimized)
        self.params_chinese = SamplingParams(
            temperature=0.18,  # Slightly lower for speed
            top_p=0.88,
            max_tokens=300,    # Slightly reduced
            stop=["<|im_end|>", "\n\n问题", "答案:", "\n\n\n"],
            skip_special_tokens=True,
        )

        # Sampling parameters for Algebra (self-consistency for accuracy)
        self.params_algebra = SamplingParams(
            temperature=0.3,   # Higher for diverse solutions
            top_p=0.95,        # High for complex reasoning
            max_tokens=600,    # Optimized for algebra complexity
            n=3,               # Generate 3 candidate solutions
            stop=["<|im_end|>", "\n\nProblem:", "\n\nExample", "\n\n\n"],
            skip_special_tokens=True,
        )

        # Sampling parameters for History/Geography/Finance
        self.params_general = SamplingParams(
            temperature=0.2,   # Slightly higher to avoid getting stuck
            top_p=0.92,
            max_tokens=448,    # Increased for complete answers (especially History)
            stop=["<|im_end|>", "\n\nQuestion:", "\n\nExample", "\n\n\n"],
            skip_special_tokens=True,
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

    def _majority_vote(self, candidates: List[str]) -> str:
        """Pick most common answer from multiple candidates (self-consistency)"""
        from collections import Counter

        # Normalize candidates (strip whitespace, lowercase for comparison)
        normalized = [c.strip().lower() for c in candidates if c.strip()]

        if not normalized:
            return candidates[0] if candidates else ""

        # Count occurrences
        counter = Counter(normalized)
        most_common = counter.most_common(1)[0][0]

        # Return original (non-normalized) version
        for i, norm in enumerate(normalized):
            if norm == most_common:
                return candidates[i].strip()

        return candidates[0]


    def _create_chat_prompt(self, question: str, subject: str = "general") -> str:
        """Create prompt using Qwen chat template with few-shot examples"""

        if subject == "chinese":
            # Streamlined Chinese prompt - faster, more direct
            prompt = f"""直接回答以下中文问题。

问题: {question}
答案:"""

        elif subject == "algebra":
            # Direct answer format - NO reasoning, just answer
            prompt = f"""You are a math expert. Solve the problem and provide ONLY the final answer. Do NOT explain your work.

Example 1:
Problem: Solve for x: 2x + 5 = 13
Answer: x = 4

Example 2:
Problem: Simplify (x+3)(x-3)
Answer: x² - 9

Example 3:
Problem: If f(x) = 3x² - 2x + 1, find f(2)
Answer: 9

Example 4:
Problem: What is the derivative of x² + 3x?
Answer: 2x + 3

Example 5:
Problem: Solve the system: x + y = 5, x - y = 1
Answer: x = 3, y = 2

Now solve this problem. Give ONLY the final answer, NO explanation:

Problem: {question}
Answer:"""

        else:
            # History/Geography/Finance - Direct answer format
            prompt = f"""Answer this question directly with just the factual answer.

Question: Who discovered America?
Answer: Christopher Columbus in 1492.

Question: What is the highest point in South America?
Answer: Mount Aconcagua in Argentina (6,961 meters).

Question: Which country is known as the 'Land of the Thunder Dragon'?
Answer: Bhutan.

Question: Who won World War 2?
Answer: The Allied Powers (United States, Soviet Union, United Kingdom, and France).

Question: {question}
Answer:"""

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_final_answer(self, text: str, subject: str) -> str:
        """Extract only the final answer from reasoning output"""

        if subject == 'algebra':
            # Strip any reasoning/thinking prefixes
            # Remove lines starting with "Okay", "So", "Let me", etc.
            reasoning_prefixes = [
                r'^(?:Okay|So|Let me|First|Now|Here|The user|To solve|We need).*?\n',
                r'^(?:I need to|I will|Let\'s|Step \d+).*?\n',
            ]
            for prefix in reasoning_prefixes:
                text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE)

            # Aggressive extraction - prioritize "Answer:" pattern
            answer_patterns = [
                r'Answer:\s*(.+?)(?:\n\n|\n(?=Problem)|$)',  # Most specific
                r'(?:Final [Aa]nswer|ANSWER):\s*(.+?)(?:\n|$)',
                r'(?:Therefore|Thus|So),?\s*(.+?)(?:\n|$)',
                r'=\s*([^=\n]+)$',  # Last equation
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    # Clean up artifacts
                    answer = answer.rstrip('.').strip()
                    # Remove any trailing explanations
                    answer = answer.split('\n')[0]
                    return answer

            # Fallback: return last non-empty line that looks like an answer
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in reversed(lines):
                # Skip lines that look like reasoning
                if not any(line.lower().startswith(word) for word in ['so', 'therefore', 'thus', 'let', 'we', 'the']):
                    return line

            if lines:
                return lines[-1]

        elif subject == 'chinese':
            # Chinese-specific extraction
            answer_patterns = [
                r'(?:答案|最终答案)[:：]\s*(.+?)(?:\n\n|\n|$)',
                r'(?:Answer|answer):\s*(.+?)(?:\n\n|\n|$)',
                r'(?:结论|因此)[:：,，]\s*(.+?)(?:\n\n|\n|$)',
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    answer = re.sub(r'\n+', ' ', answer)
                    return answer

            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
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
        # PHASE 1: BATCH SORTING BY SUBJECT
        # =====================================================================

        chinese_indices = []
        chinese_prompts = []

        algebra_indices = []
        algebra_prompts = []

        general_indices = []
        general_prompts = []

        for i, q in enumerate(questions):
            subject = q.get('subject', 'general').lower()
            prompt = self._create_chat_prompt(q['question'], subject=subject)

            if subject == 'chinese':
                chinese_indices.append(i)
                chinese_prompts.append(prompt)
            elif subject == 'algebra':
                algebra_indices.append(i)
                algebra_prompts.append(prompt)
            else:
                general_indices.append(i)
                general_prompts.append(prompt)

        # =====================================================================
        # PHASE 2: BATCH EXECUTION BY SUBJECT
        # =====================================================================

        # Batch A: Chinese questions
        if chinese_prompts:
            print(f"🇨🇳 Processing {len(chinese_prompts)} Chinese questions...")

            chinese_outputs = self.llm.generate(chinese_prompts, self.params_chinese, use_tqdm=False)

            for idx, output in zip(chinese_indices, chinese_outputs):
                raw_answer = output.outputs[0].text.strip()
                raw_answer = self._strip_thinking(raw_answer)
                answer = self._extract_final_answer(raw_answer, 'chinese')

                if len(answer) > 5000:
                    answer = answer[:5000].rsplit('. ', 1)[0] + '.'

                results[idx] = {
                    "questionID": questions[idx]["questionID"],
                    "answer": answer
                }

        # Batch B: Algebra questions (self-consistency with majority voting)
        if algebra_prompts:
            print(f"🔢 Processing {len(algebra_prompts)} Algebra questions (3x for accuracy)...")

            algebra_outputs = self.llm.generate(algebra_prompts, self.params_algebra, use_tqdm=False)

            for idx, output in zip(algebra_indices, algebra_outputs):
                # Get all 3 candidate solutions (n=3 in params)
                candidates = []
                for candidate_output in output.outputs:
                    raw = candidate_output.text.strip()
                    raw = self._strip_thinking(raw)
                    extracted = self._extract_final_answer(raw, 'algebra')
                    candidates.append(extracted)

                # Majority vote among the 3 solutions
                answer = self._majority_vote(candidates)

                if len(answer) > 5000:
                    answer = answer[:5000].rsplit('. ', 1)[0] + '.'

                results[idx] = {
                    "questionID": questions[idx]["questionID"],
                    "answer": answer
                }

        # Batch C: General questions (History, Geography, Finance, etc.)
        if general_prompts:
            print(f"📖 Processing {len(general_prompts)} general questions...")

            general_outputs = self.llm.generate(general_prompts, self.params_general, use_tqdm=False)

            for idx, output in zip(general_indices, general_outputs):
                raw_answer = output.outputs[0].text.strip()

                # Remove "Answer:" prefix if present
                if raw_answer.lower().startswith('answer:'):
                    raw_answer = raw_answer[7:].strip()

                # AGGRESSIVE cleanup for reasoning
                # If output starts with "Okay" and is long (>200 chars), it's all reasoning
                if raw_answer.lower().startswith('okay') and len(raw_answer) > 200:
                    # This is likely full reasoning - extract nothing, mark as failed
                    answer = "Unable to generate answer"
                else:
                    # Split into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', raw_answer)

                    # If first sentence starts with "Okay" or similar, skip it
                    bad_starts = ['okay', 'well', 'alright', 'let me', 'the user', 'i need', 'i will']
                    if sentences and len(sentences) > 1:
                        first_sent_lower = sentences[0].lower()
                        if any(first_sent_lower.startswith(bad) for bad in bad_starts):
                            # Remove first sentence
                            raw_answer = ' '.join(sentences[1:])

                    answer = self._strip_thinking(raw_answer.strip())

                    # Clean up any remaining meta phrases at start
                    if answer.lower().startswith('from what i'):
                        # Extract just the factual part
                        parts = answer.split(',', 1)
                        if len(parts) > 1:
                            answer = parts[1].strip()

                    # If answer is empty, fallback
                    if not answer or len(answer) < 5:
                        answer = "Unable to generate answer"

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
