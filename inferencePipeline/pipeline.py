"""
Tech Arena 2025 - Phase 2
Accuracy-Optimized Inference Pipeline with Qwen3-4B

Strategy:
- Qwen3-4B FP16 (superior model, better reasoning)
- Advanced prompt engineering for algebra & Chinese
- Enable <think> reasoning for algebra & Chinese (accuracy boost)
- Sophisticated answer extraction and post-processing
- Optimized sampling parameters per subject
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
    """Accuracy-focused inference pipeline with Qwen3-4B"""

    def __init__(self):
        """Initialize pipeline with vLLM"""

        model_path = find_model_path(MODEL_NAME, CACHE_DIR)
        print(f"🚀 Loading {MODEL_NAME.split('/')[-1]} (FP16) with vLLM...")

        # Optimized vLLM configuration for accuracy
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,  # Balanced context size
            enforce_eager=False,
            max_num_seqs=32,
            trust_remote_code=True,
            tensor_parallel_size=1,
            disable_log_stats=True,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Subject-specific sampling parameters (accuracy-optimized)
        self.params = {
            'algebra': SamplingParams(
                temperature=0.05,  # Very low for maximum accuracy
                top_p=0.95,
                max_tokens=1200,   # Long for detailed step-by-step reasoning
                repetition_penalty=1.05,
                stop=["<|im_end|>", "<|endoftext|>"],
            ),
            'chinese': SamplingParams(
                temperature=0.1,   # Low for factual accuracy
                top_p=0.95,
                max_tokens=800,
                repetition_penalty=1.05,
                stop=["<|im_end|>", "<|endoftext|>"],
            ),
            'geography': SamplingParams(
                temperature=0.15,
                top_p=0.95,
                max_tokens=600,
                stop=["<|im_end|>", "<|endoftext|>"],
            ),
            'history': SamplingParams(
                temperature=0.15,
                top_p=0.95,
                max_tokens=700,
                stop=["<|im_end|>", "<|endoftext|>"],
            ),
        }

        print("✅ Pipeline ready for high-accuracy inference\n")

    def _create_prompt(self, question: str, subject: str) -> str:
        """Advanced prompt engineering with reasoning support"""

        if subject == "algebra":
            # Advanced algebra prompt with structured thinking
            prompt = f"""You are an expert mathematics teacher. Solve this algebra problem with clear, systematic reasoning.

**Instructions:**
1. Think through the problem step-by-step inside <think> tags
2. Show all mathematical steps and intermediate calculations
3. Verify your answer
4. Provide the final answer after "ANSWER:"

**Problem:** {question}

Let me solve this systematically:

<think>"""

        elif subject == "chinese":
            # Advanced Chinese prompt with cultural context
            prompt = f"""你是中国文化和语言的资深专家，拥有深厚的知识储备。请准确、全面地回答以下问题。

**要求：**
1. 在<think>标签内进行深入思考
2. 确保答案准确、完整
3. 提供充分的细节和背景信息
4. 最后用"答案："标记最终答案

**问题：** {question}

让我仔细思考：

<think>"""

        elif subject == "geography":
            # Geography prompt with factual focus
            prompt = f"""You are a geography expert. Provide an accurate, well-structured answer to this question.

**Question:** {question}

Think carefully about the key facts, then provide a clear answer marked with "ANSWER:".

Response:"""

        elif subject == "history":
            # History prompt with context and accuracy
            prompt = f"""You are a history scholar. Provide an accurate, comprehensive answer with relevant context.

**Question:** {question}

Provide a well-reasoned answer marked with "ANSWER:".

Response:"""

        else:
            prompt = f"""Answer this question accurately and completely. Mark your final answer with "ANSWER:".

**Question:** {question}

Response:"""

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_answer(self, text: str, subject: str) -> str:
        """Advanced answer extraction with post-processing"""

        original_text = text

        # Step 1: Extract reasoning and answer sections
        thinking_content = ""
        answer_content = text

        # Extract <think> content if present
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            # Remove <think> tags but keep the content for potential fallback
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = text.strip()

        # Step 2: Look for explicit answer markers
        answer = None

        if subject == "chinese":
            # Look for Chinese answer markers
            if "答案：" in text or "答案:" in text:
                answer = re.split(r'答案[：:]', text)[-1].strip()
            elif "最终答案" in text:
                answer = text.split("最终答案")[-1].strip()
                answer = re.sub(r'^[：:\s]+', '', answer)
        else:
            # Look for English answer markers
            if "ANSWER:" in text:
                answer = text.split("ANSWER:")[-1].strip()
            elif "Final Answer:" in text or "Final answer:" in text:
                answer = re.split(r'[Ff]inal [Aa]nswer:', text)[-1].strip()

        # Step 3: Subject-specific extraction if no marker found
        if not answer or len(answer.strip()) < 3:
            if subject == "algebra":
                # Look for mathematical conclusion patterns
                lines = [l.strip() for l in text.split('\n') if l.strip()]

                # Prioritize lines with equations
                for line in reversed(lines):
                    # Look for conclusion patterns
                    if any(pattern in line.lower() for pattern in ['therefore', 'thus', 'so', '因此', '所以']):
                        if '=' in line or any(char.isdigit() for char in line):
                            answer = line
                            break
                    # Direct equation answers
                    if re.search(r'[a-z]\s*=\s*[\d\-\.]+', line, re.IGNORECASE):
                        answer = line
                        break

                # Fallback: last line with mathematical content
                if not answer:
                    for line in reversed(lines):
                        if '=' in line or re.search(r'\d+', line):
                            answer = line
                            break

            elif subject == "chinese":
                # Extract last substantial paragraph
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                if paragraphs:
                    answer = paragraphs[-1]
                else:
                    # Fallback to last sentence
                    sentences = re.split(r'[。！？]', text)
                    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
                    if sentences:
                        answer = sentences[-1] + '。'

            else:
                # For geography/history: extract last coherent paragraph
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                if paragraphs:
                    answer = paragraphs[-1]
                else:
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    if lines:
                        answer = lines[-1]

        # Step 4: Final fallback
        if not answer or len(answer.strip()) < 3:
            lines = [l.strip() for l in original_text.split('\n') if l.strip()]
            answer = lines[-1] if lines else original_text.strip()

        # Step 5: Post-processing cleanup
        answer = answer.strip()

        # Remove common prefixes
        answer = re.sub(r'^(Therefore,|Thus,|So,|Hence,|因此，|所以，)\s*', '', answer, flags=re.IGNORECASE)

        # Remove "the answer is" type phrases
        answer = re.sub(r'^(The answer is|Answer is|答案是)\s*[:：]?\s*', '', answer, flags=re.IGNORECASE)

        # For algebra: clean up equation formatting
        if subject == "algebra":
            # Ensure proper spacing around equals
            answer = re.sub(r'\s*=\s*', ' = ', answer)
            # Remove trailing explanation after the equation
            if '=' in answer:
                parts = answer.split('.')
                # Keep the part with the equation
                for part in parts:
                    if '=' in part:
                        answer = part.strip()
                        break

        # Step 6: Length limit
        if len(answer) > 5000:
            answer = answer[:5000].rsplit('. ', 1)[0]
            if not answer.endswith('.'):
                answer += '.'

        return answer.strip()

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Main inference with batched processing"""

        if not questions:
            return []

        # Group by subject for optimized batching
        subject_batches = {}
        for i, q in enumerate(questions):
            subject = q.get('subject', 'geography')
            if subject not in subject_batches:
                subject_batches[subject] = []
            subject_batches[subject].append((i, q))

        results = [None] * len(questions)

        # Process each subject batch
        for subject, batch in subject_batches.items():
            indices, qs = zip(*batch)
            prompts = [self._create_prompt(q['question'], subject) for q in qs]

            print(f"🔍 Processing {len(prompts)} {subject} questions (accuracy mode)...")

            # Get optimized sampling params
            params = self.params.get(subject, self.params['geography'])

            # Generate with vLLM
            outputs = self.llm.generate(prompts, params, use_tqdm=False)

            # Extract and post-process answers
            for idx, output, q in zip(indices, outputs, qs):
                raw_answer = output.outputs[0].text.strip()
                answer = self._extract_answer(raw_answer, subject)

                results[idx] = {
                    "questionID": q["questionID"],
                    "answer": answer
                }

        print(f"✅ Completed {len(results)} questions with high accuracy\n")
        return results


def loadPipeline():
    """Entry point for evaluation system"""
    return InferencePipeline()
