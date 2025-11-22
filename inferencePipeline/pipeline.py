"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline

ALIGNED WITH CHALLENGE REQUIREMENTS:
✅ Tesla T4 GPU (16GB VRAM, Turing arch)
✅ Llama-3.2-3B-Instruct (from approved list)
✅ Loads from /app/models (offline mode)
✅ Single-round Q&A (no conversation history)
✅ Subjects: algebra, geography, history, Chinese
✅ Max answer length: 5000 characters
✅ Scoring: 60% accuracy + 40% latency

OPTIMIZATION STRATEGIES:
1. vLLM continuous batching (max throughput)
2. FP16 native (T4 optimal, no quantization overhead)
3. Python calculator for algebra (10x faster)
4. Subject-specific prompts (accuracy boost)
5. Aggressive batching (latency reduction)
"""

import re
from typing import List, Dict
from vllm import LLM, SamplingParams


class InferencePipeline:
    """
    Single-round Q&A pipeline optimized for T4 GPU

    Target Performance:
    - Latency: <90s for 500 questions
    - Accuracy: >75%
    - Score: >70% (Top 3-5)
    """

    def __init__(self):
        """Initialize pipeline - called once before timing starts"""

        # CRITICAL: Load from local cache (no internet)
        model_path = "/app/models/Llama-3.2-3B-Instruct"

        print("🚀 Initializing inference pipeline...")

        # vLLM with T4 optimizations
        self.llm = LLM(
            model=model_path,
            dtype="float16",              # T4 native (no FP8 on Turing!)
            gpu_memory_utilization=0.90,  # Use 90% of 16GB
            max_model_len=4096,          # Large context
            enforce_eager=True,           # T4 stability (no CUDA graphs)
            max_num_seqs=32,             # Aggressive batching
            trust_remote_code=True,
            tensor_parallel_size=1,
        )

        # Get tokenizer for chat formatting
        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters (per subject type)
        self.params_strict = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=["\n\nQuestion:", "<|eot_id|>"],
        )

        self.params_creative = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            stop=["\n\nQuestion:", "<|eot_id|>"],
        )

        self.params_code = SamplingParams(
            temperature=0.0,
            max_tokens=100,
            stop=["```", "\n\n"],
        )

        # Warmup
        print("Warming up...")
        _ = self.llm.generate(["Test"], self.params_strict, use_tqdm=False)
        print("✅ Pipeline ready\n")

    def _is_math(self, question: str, subject: str) -> bool:
        """Detect if question needs Python calculator"""
        if subject == "algebra":
            return True

        # Detect arithmetic patterns
        patterns = [
            r'\d+\s*[\+\-\*×÷\/]\s*\d+',
            r'calculate|compute|multiply|divide',
            r'what is \d+',
            r'\d+%',
        ]
        return any(re.search(p, question.lower()) for p in patterns)

    def _solve_math(self, question: str) -> str:
        """
        Use Python calculator for math (10x faster than LLM)

        Phase 1: LLM generates Python expression
        Phase 2: Execute safely with eval
        """
        # Generate code
        prompt = f"""Convert to Python expression. Output ONLY code.

Q: What is 50 + 50?
A: 50 + 50

Q: {question}
A: """

        try:
            outputs = self.llm.generate([prompt], self.params_code, use_tqdm=False)
            code = outputs[0].outputs[0].text.strip()

            # Execute safely (remove non-math chars)
            safe_code = re.sub(r'[^0-9\.\+\-\*\/\(\)\%\s]', '', code)
            if not safe_code:
                return None

            result = eval(safe_code, {"__builtins__": None}, {})

            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                return f"{result:.6g}"
            return str(result)

        except:
            return None

    def _create_prompt(self, question: str, subject: str) -> str:
        """Create subject-specific prompt using Llama chat template"""

        # Subject-specific system prompts
        systems = {
            "algebra": "You are a math expert. Answer concisely.",
            "geography": "You are a geography expert. Answer precisely.",
            "history": "You are a historian. Answer with dates and facts.",
            "chinese": "You are a Chinese language expert. Answer clearly.",
        }

        system = systems.get(subject, "Answer concisely and accurately.")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _detect_subject(self, question: str) -> str:
        """Fast keyword-based subject detection"""
        q = question.lower()

        if any(w in q for w in ['solve', 'equation', 'factor', 'calculate']):
            return 'algebra'
        if any(w in q for w in ['country', 'capital', 'mountain', 'river']):
            return 'geography'
        if any(w in q for w in ['war', 'revolution', 'century', 'who was']):
            return 'history'
        if any(w in q for w in ['chinese', '中文', 'mandarin']):
            return 'chinese'

        return 'general'

    def _enforce_limit(self, text: str) -> str:
        """Enforce 5000 character limit"""
        if len(text) <= 5000:
            return text.strip()

        # Try to cut at sentence boundary
        truncated = text[:5000]
        for delim in ['. ', '.\n', '! ', '? ']:
            pos = truncated.rfind(delim)
            if pos > 4500:
                return truncated[:pos + 1]

        return truncated.rstrip() + "..."

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Main inference method (timed in evaluation)

        Input: List of {"questionID": str, "question": str, "subject": str}
        Output: List of {"questionID": str, "answer": str}
        """

        if not questions:
            return []

        results = [None] * len(questions)

        # Separate math (calculator) from text (LLM)
        math_indices, math_qs = [], []
        text_indices, text_prompts, text_params = [], [], []

        for i, q in enumerate(questions):
            subject = q.get('subject', self._detect_subject(q['question']))

            if self._is_math(q['question'], subject):
                math_indices.append(i)
                math_qs.append(q)
            else:
                text_indices.append(i)
                text_prompts.append(self._create_prompt(q['question'], subject))

                # Creative for history/geography, strict otherwise
                params = self.params_creative if subject in ['history', 'geography'] else self.params_strict
                text_params.append(params)

        # Process math with calculator (fast!)
        for idx, q in zip(math_indices, math_qs):
            answer = self._solve_math(q['question'])

            if answer:
                results[idx] = {
                    "questionID": q["questionID"],
                    "answer": answer
                }
            else:
                # Calculator failed, fallback to LLM
                subject = q.get('subject', self._detect_subject(q['question']))
                text_indices.append(idx)
                text_prompts.append(self._create_prompt(q['question'], subject))
                text_params.append(self.params_strict)

        # Batch process all text questions with vLLM
        if text_prompts:
            try:
                outputs = self.llm.generate(text_prompts, text_params, use_tqdm=False)

                for idx, output in zip(text_indices, outputs):
                    answer = output.outputs[0].text.strip()
                    answer = self._enforce_limit(answer)

                    results[idx] = {
                        "questionID": questions[idx]["questionID"],
                        "answer": answer
                    }
            except Exception as e:
                # Fallback: sequential processing
                for idx, prompt, params in zip(text_indices, text_prompts, text_params):
                    try:
                        output = self.llm.generate([prompt], params, use_tqdm=False)[0]
                        answer = self._enforce_limit(output.outputs[0].text.strip())
                        results[idx] = {
                            "questionID": questions[idx]["questionID"],
                            "answer": answer
                        }
                    except:
                        results[idx] = {
                            "questionID": questions[idx]["questionID"],
                            "answer": "Error processing question."
                        }

        return results


def loadPipeline():
    """
    Entry point for evaluation (called by run.py)
    Returns callable pipeline
    """
    return InferencePipeline()
