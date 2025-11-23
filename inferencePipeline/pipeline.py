"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline

Strategy:
- Llama-3.2-3B-Instruct for all subjects
- Simple prompts with ANSWER: marker
- Subject-optimized sampling parameters
- Batched processing for efficiency
"""

import os
import re
from typing import List, Dict
from vllm import LLM, SamplingParams
from pathlib import Path

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
CACHE_DIR = "/app/models"

def find_model_path(model_name: str, cache_dir: str) -> str:
    """Find the actual snapshot path in HuggingFace cache"""
    cache_path = Path(cache_dir)
    hf_cache_name = "models--" + model_name.replace("/", "--")
    model_cache = cache_path / hf_cache_name

    if not model_cache.exists():
        raise FileNotFoundError(f"Model cache not found at {model_cache}")

    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Snapshots directory not found at {snapshots_dir}")

    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

    snapshot = sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return str(snapshot)


class InferencePipeline:
    """Simple, efficient inference pipeline"""

    def __init__(self):
        """Initialize pipeline with vLLM"""

        model_path = find_model_path(MODEL_NAME, CACHE_DIR)
        print(f"🚀 Loading {MODEL_NAME.split('/')[-1]} with vLLM...")

        # Optimized vLLM configuration for T4 GPU
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=False,
            max_num_seqs=32,
            trust_remote_code=True,
            tensor_parallel_size=1,
            disable_log_stats=True,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters optimized by subject
        self.params = {
            'algebra': SamplingParams(
                temperature=0.1,   # Low temp for accuracy
                top_p=0.9,
                max_tokens=700,
                stop=["<|eot_id|>", "\n\nQuestion:", "\n\nProblem:"],
            ),
            'chinese': SamplingParams(
                temperature=0.2,
                top_p=0.9,
                max_tokens=400,
                stop=["<|eot_id|>", "\n\n问题"],
            ),
            'default': SamplingParams(
                temperature=0.2,
                top_p=0.9,
                max_tokens=350,
                stop=["<|eot_id|>", "\n\nQuestion:"],
            ),
        }

        print("✅ Pipeline ready\n")

    def _create_prompt(self, question: str, subject: str) -> str:
        """Create simple prompts with ANSWER: marker"""

        if subject == "algebra":
            prompt = f"""Solve this algebra problem step by step. At the end, clearly mark your final answer with "ANSWER:".

Problem: {question}

Solution:"""

        elif subject == "chinese":
            prompt = f"""你是中国文化和语言专家。请回答以下问题，最后用"答案："标记你的最终答案。

问题：{question}

回答："""

        elif subject == "geography":
            prompt = f"""Answer this geography question concisely. Mark your final answer with "ANSWER:".

Question: {question}

Response:"""

        elif subject == "history":
            prompt = f"""Answer this history question accurately. Mark your final answer with "ANSWER:".

Question: {question}

Response:"""

        else:
            prompt = f"""Answer this question clearly. Mark your final answer with "ANSWER:".

Question: {question}

Response:"""

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_answer(self, text: str, subject: str) -> str:
        """Extract final answer from response"""

        # Strip <think> tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?think>', '', text).strip()

        # Look for answer markers
        if "ANSWER:" in text:
            answer = text.split("ANSWER:")[-1].strip()
        elif "答案：" in text or "答案:" in text:
            answer = re.split(r'答案[：:]', text)[-1].strip()
        else:
            # Fallback: take last paragraph
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if lines:
                answer = lines[-1]
            else:
                answer = text

        # Clean up and limit length
        answer = answer.strip()
        if len(answer) > 5000:
            answer = answer[:5000]

        return answer

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Main inference method with batched processing"""

        if not questions:
            return []

        # Group by subject for batched processing
        subject_batches = {}
        for i, q in enumerate(questions):
            subject = q.get('subject', 'default')
            if subject not in subject_batches:
                subject_batches[subject] = []
            subject_batches[subject].append((i, q))

        # Process each subject batch
        results = [None] * len(questions)

        for subject, batch in subject_batches.items():
            indices, qs = zip(*batch)
            prompts = [self._create_prompt(q['question'], subject) for q in qs]

            print(f"Processing {len(prompts)} {subject} questions...")

            # Get sampling params for this subject
            params = self.params.get(subject, self.params['default'])

            # Generate
            outputs = self.llm.generate(prompts, params, use_tqdm=False)

            # Extract answers
            for idx, output, q in zip(indices, outputs, qs):
                raw_answer = output.outputs[0].text.strip()
                answer = self._extract_answer(raw_answer, subject)

                results[idx] = {
                    "questionID": q["questionID"],
                    "answer": answer
                }

        print(f"✅ Completed {len(results)} questions\n")
        return results


def loadPipeline():
    """Entry point for evaluation system"""
    return InferencePipeline()
