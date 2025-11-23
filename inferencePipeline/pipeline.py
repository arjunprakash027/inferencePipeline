"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline

Strategy:
- Use Llama-3.2-3B for fast, accurate inference (FP16 precision)
- Enhanced prompts with knowledge bases for better accuracy
- Batched processing by subject
- Optimized parameters for accuracy and speed balance
"""

import os
import re
from typing import List, Dict
from vllm import LLM, SamplingParams
from pathlib import Path

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct")  # Switched to Qwen for better math reasoning
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
    """Optimized inference pipeline with knowledge base integration"""

    def __init__(self):
        """Initialize pipeline with vLLM using FP16 for optimal T4 performance"""

        model_path = find_model_path(MODEL_NAME, CACHE_DIR)
        print(f"🚀 Loading {MODEL_NAME.split('/')[-1]} with vLLM (FP16)...")

        # Optimized vLLM configuration for T4 GPU with FP16
        self.llm = LLM(
            model=model_path,
            dtype="float16",  # Optimal for T4 performance
            gpu_memory_utilization=0.90,  # Use most of T4 memory efficiently
            max_model_len=4096,
            enforce_eager=False,  # Enable CUDA graphs for better performance
            max_num_seqs=32,  # Maximize batch size for throughput
            trust_remote_code=True,
            tensor_parallel_size=1,
            disable_log_stats=True,
            # Additional optimizations for T4
            quantization=None,  # Using native FP16, no quantization
            # Enable graph capture for better performance on repeated prompts
            enable_prefix_caching=True,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Load knowledge bases
        script_dir = Path(__file__).resolve().parent
        self.algebra_kb = self._load_knowledge_base(script_dir / "algebra_kb.txt")
        self.chinese_kb = self._load_knowledge_base(script_dir / "chinese_kb.txt")
        
        # Sampling parameters optimized for each subject
        self.params = {
            'algebra': SamplingParams(
                temperature=0.05,  # Very low temperature for deterministic accuracy
                top_p=0.9,  # Focus on likely tokens
                top_k=20,   # Narrower selection for consistency
                max_tokens=1000,  # Allow space for step-by-step reasoning
                stop=["Question:", "\n\nQuestion:"],  # Stop at next question, let extraction handle ANSWER
            ),
            'chinese': SamplingParams(
                temperature=0.1,  # Low for cultural accuracy
                top_p=0.92,
                top_k=30,
                max_tokens=600,  # Appropriate for Chinese cultural questions
                stop=["答案：", "答案:", "END", "\n\n问题"],
            ),
            'geography': SamplingParams(
                temperature=0.08,  # Low for factual accuracy
                top_p=0.90,
                top_k=25,
                max_tokens=400,
                stop=["ANSWER:", "END", "\n\nQuestion:"],
            ),
            'history': SamplingParams(
                temperature=0.12,  # Moderate for narrative flow
                top_p=0.90,
                top_k=35,
                max_tokens=700,  # Allow for detailed historical context
                stop=["ANSWER:", "END", "\n\nQuestion:"],
            ),
            'default': SamplingParams(
                temperature=0.15,  # Balanced for general questions
                top_p=0.9,
                top_k=40,
                max_tokens=500,
                stop=["ANSWER:", "END", "\n\nQuestion:"],
            ),
        }

        print("✅ Pipeline initialized with knowledge bases and optimized parameters\n")

    def _is_algebra_question(self, question: str) -> bool:
        """
        Determine if a question is algebraic in nature.
        """
        question_lower = question.lower()

        # Check for common algebra indicators
        algebra_indicators = [
            'x =', 'y =', 'z =',
            'equation', 'solve', 'find x', 'find y', 'find z',
            'algebra', 'polynomial', 'quadratic', 'linear',
            'variable', 'unknown', 'calculate', 'compute',
            # Common algebra words
            'equation', 'formula', 'function', 'graph',
            # Mathematical operators in context
            '=', 'equals', 'plus', 'minus', 'times', 'divided by'
        ]

        # Check if question contains numbers and mathematical operations
        has_numbers = any(c.isdigit() for c in question)
        has_math_ops = any(op in question_lower for op in ['+', '-', '*', '/', '^', '**'])

        # Check for algebraic language
        has_indicators = any(indicator in question_lower for indicator in algebra_indicators)

        # Additional check for common algebra problem formats
        # e.g. "If 2x + 3 = 7, what is x?"
        algebraic_pattern = any(pattern in question_lower for pattern in [
            'what is x', 'what is y', 'what is the value', 'find the value',
            'if x', 'when x', 'x equals', 'y equals'
        ])

        return has_numbers or has_math_ops or has_indicators or algebraic_pattern

    def _is_chinese_question(self, question: str) -> bool:
        """
        Determine if a question is related to Chinese language or culture.
        """
        # Check for Chinese characters in the question
        has_chinese_chars = any('\u4e00' <= char <= '\u9fff' for char in question)

        # Check for keywords related to Chinese language/culture
        chinese_keywords = [
            'chinese', 'china', 'chinese language', 'mandarin', 'cantonese',
            'confucius', 'confucian', 'tao', 'dao', 'buddhism', 'buddhist',
            'dynasty', 'dynasties', 'tang', 'song', 'ming', 'qing', 'han', 'tang',
            'pinyin', 'characters', 'character', 'simplified', 'traditional',
            'culture', 'cultural', 'history', 'philosophy', 'poetry', 'poem',
            'calligraphy', 'painting', 'art', 'temple', 'palace', 'forbidden city',
            'great wall', 'kung fu', 'wushu', 'tai chi', 'martial arts', 'taoism', 'daoism',
            'feng shui', 'dragon', 'phoenix', 'zodiac', 'chinese zodiac',
            'festival', 'spring festival', 'lantern festival', 'mid-autumn',
            'the capital of china', 'beijing', 'shanghai', 'guangzhou', 'chongqing',
            'sichuan', 'shandong', 'cuisine', 'chinese food', 'dim sum'
        ]

        question_lower = question.lower()
        has_chinese_keywords = any(keyword in question_lower for keyword in chinese_keywords)

        return has_chinese_chars or has_chinese_keywords

    def _load_knowledge_base(self, kb_path: str) -> str:
        """Load knowledge base content from file"""
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            print(f"⚠️  Knowledge base not found: {kb_path}, using empty content")
            return ""

    def _create_prompt(self, question: str, subject: str) -> str:
        """Create optimized prompts with knowledge base integration"""

        if subject == "algebra":
            # Enhanced algebra prompt with simple thinking approach
            kb_context = self.algebra_kb[:3000] if self.algebra_kb else ""
            prompt = f"""You are an expert mathematician. Solve this algebra problem step by step.

{kb_context}

Question: {question}

Think through this step by step, and at the end put your final answer after "ANSWER:"

Thinking:"""

        elif subject == "chinese":
            # Use Chinese knowledge base for cultural and language expertise
            kb_context = self.chinese_kb[:3000] if self.chinese_kb else ""
            prompt = f"""你是中国文化和语言专家。参考以下知识库回答问题：

{kb_context}

请回答以下问题，展示你的推理过程，并最后用"答案："标记你的最终答案。

问题：{question}

回答："""

        elif subject == "geography":
            prompt = f"""You are a geography expert. Answer this question accurately based on geographical facts and data. Focus on factual information like locations, capitals, landmarks, and geographical features. Mark your final answer with "ANSWER:".

Question: {question}

Response:"""

        elif subject == "history":
            prompt = f"""You are a history expert. Answer this question accurately based on historical facts, events, dates, and figures. Provide context and details where relevant. Mark your final answer with "ANSWER:".

Question: {question}

Response:"""

        else:
            prompt = f"""Answer this question clearly and concisely based on your knowledge. Mark your final answer with "ANSWER:".

Question: {question}

Response:"""

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_answer(self, text: str, subject: str) -> str:
        """Extract final answer from response with improved accuracy"""

        # Strip special tokens and formatting
        text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
        text = text.strip()

        # Look for answer markers
        answer = ""
        if "ANSWER:" in text:
            # For algebra, extract everything after ANSWER:
            answer_parts = text.split("ANSWER:")
            if len(answer_parts) > 1:
                answer = answer_parts[-1].strip()
                # Get just the first line after ANSWER: to get the final answer
                answer_lines = answer.split('\n')
                if len(answer_lines) > 0:
                    answer = answer_lines[0].strip()
            else:
                answer = text.split("ANSWER:")[-1].strip()
        elif "答案：" in text or "答案:" in text:
            # Handle Chinese answer markers
            match = re.search(r'(?:答案[：:])\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            else:
                answer = text.split("答案：")[-1].split("答案:")[-1].strip()
        else:
            # Fallback: extract the last coherent sentence/paragraph
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                # Look for the most complete answer at the end of response
                for line in reversed(lines):
                    if len(line) > 10:  # Not too short
                        answer = line
                        break
                if not answer:
                    answer = lines[-1]
            else:
                answer = text

        # Clean up the answer
        answer = answer.strip()
        
        # Remove any trailing text that looks like follow-up questions or instructions
        answer = re.split(r'\n\s*\n|Question:|Problem:', answer)[0]
        
        # Ensure it doesn't end with incomplete phrases
        answer = re.sub(r'\s*$', '', answer)  # Remove trailing whitespace
        
        # Limit length to 5000 characters as required
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