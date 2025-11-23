"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline with vLLM

OPTIMIZATIONS:
✅ vLLM for fast inference with Qwen 4B
✅ FP16 precision for T4 GPU (16GB)
✅ Memory-optimized batch processing for T4 constraints
✅ Batched processing by subject
✅ Chain-of-Thought (CoT) prompting for Algebra (8 examples)
✅ Self-Consistency for Algebra (n=5 diverse solutions, majority vote)
✅ Cache-Augmented Generation (CAG) for Chinese questions
✅ Comprehensive Chinese knowledge base (32KB, 1055 lines)
✅ Answer extraction (returns only final answers, not reasoning)
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
CHINESE_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "chinese_knowledge_base.txt")
ALGEBRA_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "algebra_knowledge_base.txt")


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

        # Load knowledge bases for CAG
        self.chinese_kb = self._load_chinese_knowledge_base()
        self.algebra_kb = self._load_algebra_knowledge_base()

        # Optimized configuration for T4 16GB GPU (speed + stability)
        print("🚀 Loading Qwen 4B model with vLLM (FP16)...")

        self.llm = LLM(
            model=RAW_MODEL_PATH,             # Load directly from HF cache
            dtype="half",                     # FP16 for compute
            gpu_memory_utilization=0.90,      # Higher utilization for better performance
            max_model_len=2048,               # Safe context length for all questions
            enforce_eager=False,              # Enable CUDA graphs for speed
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
            temperature=0.4,   # Slightly higher for more diversity
            top_p=0.95,        # High for complex reasoning
            max_tokens=800,    # Increased for Chain-of-Thought reasoning
            n=5,               # Generate 5 candidate solutions (more diverse)
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

    def _load_chinese_knowledge_base(self) -> str:
        """Load Chinese knowledge base for cache-augmented generation"""
        try:
            kb_path = Path(CHINESE_KB_PATH)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📚 Loaded Chinese knowledge base ({len(content)} chars)")
                return content
            else:
                print("⚠️  Chinese knowledge base not found, using default prompts")
                return ""
        except Exception as e:
            print(f"⚠️  Error loading Chinese KB: {e}")
            return ""

    def _load_algebra_knowledge_base(self) -> str:
        """Load Algebra knowledge base for formula/theorem reference"""
        try:
            kb_path = Path(ALGEBRA_KB_PATH)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📐 Loaded Algebra knowledge base ({len(content)} chars)")
                return content
            else:
                print("⚠️  Algebra knowledge base not found, using default prompts")
                return ""
        except Exception as e:
            print(f"⚠️  Error loading Algebra KB: {e}")
            return ""

    def _get_relevant_chinese_context(self, question: str) -> str:
        """Extract relevant context from knowledge base based on question keywords"""
        if not self.chinese_kb:
            return ""

        question_lower = question.lower()

        # Extract potential keywords (simplified approach)
        keyword_mapping = {
            '朝代': ['朝代', 'Dynasty', 'Dynasties'],
            '节日': ['节日', 'Festival', 'Festivals'],
            '哲学': ['哲学', '儒家', '道家', '佛教', 'Confucianism', 'Taoism', 'Buddhism'],
            '文学': ['文学', '诗', '词', '小说', 'Literature', 'Poetry', 'Novel'],
            '发明': ['发明', 'Invention', '造纸', '印刷', '指南针', '火药'],
            '艺术': ['艺术', '书法', '国画', 'Art', 'Calligraphy', 'Painting'],
            '饮食': ['饮食', '菜', '茶', 'Cuisine', 'Food', 'Tea'],
            '建筑': ['建筑', '故宫', '长城', 'Architecture', 'Forbidden City', 'Great Wall'],
            '戏曲': ['戏曲', '京剧', 'Opera'],
            '医学': ['医学', '中医', 'Medicine', 'TCM'],
            '武术': ['武术', 'Martial Arts', 'Kung Fu'],
            '汉字': ['汉字', '拼音', 'Characters', 'Pinyin'],
            '成语': ['成语', 'Idiom'],
            '地理': ['山', '河', '省', 'Mountain', 'River', 'Province'],
        }

        # Find matching sections
        relevant_sections = []
        for category, search_terms in keyword_mapping.items():
            if any(term in question or term in question_lower for term in search_terms):
                # Extract section from knowledge base
                for section_marker in [f"=== {category}", f"/ {category} "]:
                    if section_marker in self.chinese_kb:
                        start_idx = self.chinese_kb.find(section_marker)
                        if start_idx != -1:
                            # Find next section
                            next_section_idx = self.chinese_kb.find("\n===", start_idx + 10)
                            if next_section_idx == -1:
                                section_content = self.chinese_kb[start_idx:start_idx+2000]
                            else:
                                section_content = self.chinese_kb[start_idx:next_section_idx]
                            relevant_sections.append(section_content.strip())
                            break

        if relevant_sections:
            # Combine and limit to reasonable length
            context = "\n\n".join(relevant_sections[:2])  # Max 2 sections
            if len(context) > 1500:
                context = context[:1500] + "..."
            return context

        return ""

    def _get_relevant_algebra_context(self, question: str) -> str:
        """Extract relevant formulas/theorems from algebra KB based on question"""
        if not self.algebra_kb:
            return ""

        question_lower = question.lower()

        # Keyword mapping for algebra topics
        keyword_mapping = {
            'quadratic|二次': 'Quadratic',
            'factor|因式分解': 'Factoring',
            'exponent|指数': 'Exponents',
            'log|对数|logarithm': 'Logarithm',
            'inequal|不等式': 'Inequalities',
            'function|函数': 'Functions',
            'sequence|序列|数列': 'Sequences',
            'series|级数': 'Series',
            'permutation|排列': 'Permutations',
            'combination|组合': 'Combinations',
            'binomial|二项式': 'Binomial',
            'complex|复数': 'Complex',
            'matrix|矩阵': 'Matrices',
            'determinant|行列式': 'Determinants',
            'vector|向量': 'Vectors',
            'derivative|导数|微分': 'Derivatives',
            'integral|积分': 'Integrals',
            'sin|cos|tan|三角': 'Trigonometric',
            'circle|椭圆|ellipse|hyperbola|parabola|圆锥曲线': 'Conic',
            'prime|质数|gcd|lcm': 'Number Theory',
            'set|集合': 'Set Theory',
            'logic|逻辑': 'Logic',
        }

        # Find matching sections
        relevant_sections = []
        for pattern, topic in keyword_mapping.items():
            if re.search(pattern, question_lower):
                # Search for section in KB
                section_markers = [f"=== {topic}", f"/ {topic}"]
                for marker in section_markers:
                    if marker in self.algebra_kb:
                        start_idx = self.algebra_kb.find(marker)
                        if start_idx != -1:
                            # Find next section
                            next_section = self.algebra_kb.find("\n===", start_idx + 10)
                            if next_section == -1:
                                section = self.algebra_kb[start_idx:start_idx+1200]
                            else:
                                section = self.algebra_kb[start_idx:next_section]
                            relevant_sections.append(section.strip())
                            break

        if relevant_sections:
            # Combine sections (limit to 1000 chars for context window)
            context = "\n\n".join(relevant_sections[:2])
            if len(context) > 1000:
                context = context[:1000] + "..."
            return context

        return ""

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
            # Cache-Augmented Generation for Chinese questions
            context = self._get_relevant_chinese_context(question)

            if context:
                # Use knowledge base context
                prompt = f"""你是中国文化和语言专家。使用以下参考资料回答问题。

参考资料：
{context}

基于以上参考资料，直接回答以下问题：

问题: {question}
答案:"""
            else:
                # Fallback to simple prompt
                prompt = f"""你是中国文化和语言专家。直接回答以下中文问题。

问题: {question}
答案:"""

        elif subject == "algebra":
            # Cache-Augmented Generation + Chain-of-Thought for algebra
            context = self._get_relevant_algebra_context(question)

            if context:
                # Use formula/theorem context + CoT
                prompt = f"""You are a math expert. Use the formulas and theorems below to solve the problem.

REFERENCE FORMULAS:
{context}

Using the above formulas, solve step-by-step and provide the final answer.

Example 1:
Problem: Solve for x: 2x + 5 = 13
Solution: Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4.
Final Answer: x = 4

Example 2:
Problem: Simplify (x+3)(x-3)
Solution: Using difference of squares formula (a+b)(a-b) = a² - b². Here a=x, b=3.
Final Answer: x² - 9

Example 3:
Problem: If f(x) = 3x² - 2x + 1, find f(2)
Solution: Substitute x=2 into the function: f(2) = 3(2)² - 2(2) + 1 = 3(4) - 4 + 1 = 12 - 4 + 1 = 9.
Final Answer: 9

Example 4:
Problem: What is the derivative of x² + 3x?
Solution: Using power rule: d/dx(x²) = 2x, d/dx(3x) = 3. Sum: 2x + 3.
Final Answer: 2x + 3

Example 5:
Problem: Solve the system: x + y = 5, x - y = 1
Solution: Add equations: (x+y) + (x-y) = 5+1, so 2x = 6, x = 3. Substitute into first equation: 3 + y = 5, y = 2.
Final Answer: x = 3, y = 2

Example 6:
Problem: Find the area of a circle with radius 5
Solution: Use formula A = πr². A = π(5)² = 25π ≈ 78.54.
Final Answer: 25π or approximately 78.54

Example 7:
Problem: Factor x² + 5x + 6
Solution: Find two numbers that multiply to 6 and add to 5: 2 and 3. So (x+2)(x+3).
Final Answer: (x+2)(x+3)

Example 8:
Problem: What is 15% of 80?
Solution: 15% = 0.15. Multiply: 0.15 × 80 = 12.
Final Answer: 12

Now solve this problem step-by-step:

Problem: {question}
Solution:"""
            else:
                # Fallback without context
                prompt = f"""You are a math expert. Solve the problem step-by-step, then provide the final answer.

Example 1:
Problem: Solve for x: 2x + 5 = 13
Solution: Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4.
Final Answer: x = 4

Example 2:
Problem: Simplify (x+3)(x-3)
Solution: Using difference of squares formula (a+b)(a-b) = a² - b². Here a=x, b=3.
Final Answer: x² - 9

Example 3:
Problem: If f(x) = 3x² - 2x + 1, find f(2)
Solution: Substitute x=2 into the function: f(2) = 3(2)² - 2(2) + 1 = 3(4) - 4 + 1 = 12 - 4 + 1 = 9.
Final Answer: 9

Example 4:
Problem: What is the derivative of x² + 3x?
Solution: Using power rule: d/dx(x²) = 2x, d/dx(3x) = 3. Sum: 2x + 3.
Final Answer: 2x + 3

Example 5:
Problem: Solve the system: x + y = 5, x - y = 1
Solution: Add equations: (x+y) + (x-y) = 5+1, so 2x = 6, x = 3. Substitute into first equation: 3 + y = 5, y = 2.
Final Answer: x = 3, y = 2

Example 6:
Problem: Find the area of a circle with radius 5
Solution: Use formula A = πr². A = π(5)² = 25π ≈ 78.54.
Final Answer: 25π or approximately 78.54

Example 7:
Problem: Factor x² + 5x + 6
Solution: Find two numbers that multiply to 6 and add to 5: 2 and 3. So (x+2)(x+3).
Final Answer: (x+2)(x+3)

Example 8:
Problem: What is 15% of 80?
Solution: 15% = 0.15. Multiply: 0.15 × 80 = 12.
Final Answer: 12

Now solve this problem step-by-step:

Problem: {question}
Solution:"""

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
