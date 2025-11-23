"""
Tech Arena 2025 - Phase 2
Efficient LLM Inference Pipeline with optional AWQ quantization

Strategy:
- Use Llama-3.2-3B for fast, accurate inference (FP16 precision)
- Enhanced prompts with knowledge bases for better accuracy
- Batched processing by subject
- Optimized parameters for accuracy and speed balance
- Python calculator for algebra problems to improve speed and accuracy
"""

import os
import re
import ast
import operator
from typing import List, Dict
from vllm import LLM, SamplingParams
from pathlib import Path

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
CACHE_DIR = "/app/models"

# Model configuration dictionary
MODEL_CONFIGS = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "dtype": "float16",
        "quantization": "awq",
        "gpu_memory_utilization": 0.85,
    },
    # fallback / default config (FP16, no quantization)
    "default": {
        "dtype": "float16",
        "quantization": None,
        "gpu_memory_utilization": 0.90,
    },
}

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


def quantize_model_awq(model_path: str, cache_dir: str, model_name: str) -> str:
    """Quantize a model to 4‑bit AWQ on‑the‑fly.
    Returns the path to the quantized checkpoint. If the checkpoint already exists, it is reused.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    import torch, shutil, gc, os

    safe_name = model_name.split('/')[-1].lower().replace('-', '_')
    quant_path = os.path.join(cache_dir, f"{safe_name}_awq")

    if os.path.isdir(quant_path) and os.path.isfile(os.path.join(quant_path, "config.json")):
        print(f"✅ AWQ quantized model already cached at {quant_path}")
        return quant_path

    print(f"🔧 Quantizing {model_name} to 4‑bit AWQ …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model with CPU off‑loading to avoid OOM on T4
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        max_memory={0: "13GB", "cpu": "30GB"},
        offload_folder="offload_tmp",
        safetensors=True,
    )

    # Simple calibration set (20 diverse sentences)
    calibration_data = [
        "中国的首都是北京。",
        "Solve for x: 3x + 7 = 22",
        "The Industrial Revolution began in Great Britain.",
        "长城是世界著名的古代建筑奇迹。",
        "Calculate the derivative of f(x) = x^3 - 4x + 1",
        "Photosynthesis is the process by which plants convert light into energy.",
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "唐朝是中国历史上最繁荣的朝代之一。",
        "Find the roots of the quadratic equation: x^2 - 5x + 6 = 0",
        "The Amazon Rainforest is often referred to as the 'lungs of the Earth'.",
        "人工智能在医疗领域的应用前景如何？",
        "What is the area of a circle with radius 5?",
        "Albert Einstein developed the theory of relativity.",
        "Solve the system of equations: 2x + y = 10, x - y = 2",
        "The currency of Japan is the Yen.",
        "Calculate the integral of sin(x) from 0 to pi.",
        "Marie Curie was the first woman to win a Nobel Prize.",
        "If log_10(x) = 2, what is x?",
        "The Great Barrier Reef is located off the coast of Australia.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)

    os.makedirs(quant_path, exist_ok=True)
    model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
    tokenizer.save_pretrained(quant_path)
    print(f"✅ AWQ quantization complete. Model saved to {quant_path}")

    # Cleanup GPU/CPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if os.path.isdir("offload_tmp"):
        shutil.rmtree("offload_tmp", ignore_errors=True)
    return quant_path


class InferencePipeline:
    """Optimized inference pipeline with knowledge base integration"""

    def __init__(self):
        """Initialize pipeline with vLLM using FP16 (or AWQ) for optimal T4 performance"""

        # Resolve raw model path first
        raw_model_path = find_model_path(MODEL_NAME, CACHE_DIR)
        # Determine configuration (fallback to default)
        model_cfg = MODEL_CONFIGS.get(MODEL_NAME, MODEL_CONFIGS["default"])
        # If AWQ quantization is requested, perform on‑the‑fly quantization
        if model_cfg.get("quantization") == "awq":
            print("🔧 Starting AWQ on‑the‑fly quantization...")
            model_path = quantize_model_awq(raw_model_path, CACHE_DIR, MODEL_NAME)
        else:
            model_path = raw_model_path
        print(f"🚀 Loading {MODEL_NAME.split('/')[-1]} with vLLM (dtype={model_cfg['dtype']}, quantization={model_cfg.get('quantization')})...")

        # Optimized vLLM configuration for T4 GPU with FP16
        self.llm = LLM(
            model=model_path,
            dtype=model_cfg["dtype"],
            gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
            max_model_len=4096,
            enforce_eager=False,
            max_num_seqs=32,
            trust_remote_code=True,
            tensor_parallel_size=1,
            disable_log_stats=True,
            quantization=model_cfg.get("quantization"),
            enable_prefix_caching=True,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Load knowledge bases
        self.algebra_kb = self._load_knowledge_base("inferencePipeline/algebra_kb.txt")
        self.chinese_kb = self._load_knowledge_base("inferencePipeline/chinese_kb.txt")

        # Sampling parameters optimized for each subject
        self.params = {
            'algebra': SamplingParams(
                temperature=0.1,  # Low temperature for deterministic accuracy
                top_p=0.95,  # Good balance of creativity and focus
                top_k=50,   # Limit to most likely tokens
                max_tokens=800,  # Allow space for step-by-step reasoning
                stop=["ANSWER:", "答案：", "答案:", "\n\nQuestion:", "\n\nProblem:"],
            ),
            'chinese': SamplingParams(
                temperature=0.15,  # Slightly higher for natural language flow
                top_p=0.92,
                top_k=40,
                max_tokens=500,  # Appropriate for Chinese cultural questions
                stop=["答案：", "答案:", "END", "\n\n问题"],
            ),
            'geography': SamplingParams(
                temperature=0.12,  # Low for factual accuracy
                top_p=0.90,
                top_k=45,
                max_tokens=400,
                stop=["ANSWER:", "END", "\n\nQuestion:"],
            ),
            'history': SamplingParams(
                temperature=0.18,  # Slightly higher for narrative flow
                top_p=0.90,
                top_k=50,
                max_tokens=600,  # Allow for detailed historical context
                stop=["ANSWER:", "END", "\n\nQuestion:"],
            ),
            'default': SamplingParams(
                temperature=0.2,  # Balanced for general questions
                top_p=0.9,
                top_k=50,
                max_tokens=450,
                stop=["ANSWER:", "END", "\n\nQuestion:"],
            ),
        }

        print("✅ Pipeline initialized with knowledge bases and optimized parameters\n")

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression using AST parsing.
        This prevents execution of arbitrary code while allowing basic math operations.
        """
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode='eval')

            # Define allowed operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            def _eval_node(node):
                if isinstance(node, ast.Constant):  # Numbers
                    return node.value
                elif isinstance(node, ast.Num):  # For older Python versions
                    return node.n
                elif isinstance(node, ast.BinOp):  # Binary operations
                    left = _eval_node(node.left)
                    right = _eval_node(node.right)
                    op = ops.get(type(node.op))
                    if op is None:
                        raise ValueError(f"Unsupported operation: {type(node.op)}")
                    return op(left, right)
                elif isinstance(node, ast.UnaryOp):  # Unary operations
                    operand = _eval_node(node.operand)
                    op = ops.get(type(node.op))
                    if op is None:
                        raise ValueError(f"Unsupported unary operation: {type(node.op)}")
                    return op(operand)
                else:
                    raise ValueError(f"Unsupported node type: {type(node)}")

            result = _eval_node(tree.body)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")

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

    def _solve_algebra_fast(self, question: str) -> str:
        """
        Try to solve algebraic questions using Python calculator for speed and accuracy.
        """
        try:
            # Try to parse and extract a mathematical expression from the question
            # This is a simple implementation - can be enhanced with more complex parsing

            # Common patterns in algebra questions
            import re

            # Pattern 1: Simple equations like "2x + 3 = 7"
            patterns = [
                r'([0-9\+\-\*\/\(\)\. ]+)=([0-9\+\-\*\/\(\)\. ]+)',  # equation with equals
                r'calculate\s+([0-9\+\-\*\/\(\)\. ]+)',  # calculate command
                r'compute\s+([0-9\+\-\*\/\(\)\. ]+)',  # compute command
                r'what\sis\s([0-9\+\-\*\/\(\)\. ]+)\?',  # what is expression
            ]

            for pattern in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    expr = match.group(1).strip()
                    # Clean up the expression
                    expr = expr.replace(" ", "").replace("×", "*").replace("÷", "/")

                    # Only evaluate if it's a pure mathematical expression
                    if re.match(r'^[0-9\+\-\*\/\(\)\.]+$', expr):
                        result = self._safe_eval(expr)
                        return f"The answer is {result}"

            # If we couldn't extract a simple expression, return None to fall back to LLM
            return None
        except:
            # If safe eval fails, fall back to LLM
            return None

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
            # Use algebra knowledge base for better mathematical reasoning
            kb_context = self.algebra_kb[:3000] if self.algebra_kb else ""
            prompt = f"""You are an expert mathematician. Use the following algebra knowledge base to solve problems:

{kb_context}

Solve this step-by-step following mathematical best practices:
1. Identify knowns and unknowns in the problem
2. Choose the appropriate formula or method
3. Perform calculations carefully showing your work
4. Verify your solution by plugging back into original equation if applicable
5. State your final answer clearly with "ANSWER:"

Problem: {question}

Solution:"""

        elif subject == "chinese":
            # Enhanced Chinese prompt with cultural context and structured approach
            kb_context = self.chinese_kb[:3000] if self.chinese_kb else ""
            prompt = f"""你是中国文化和语言专家。具有深厚的中文语言、历史、哲学、艺术和文化传统知识。参考以下知识库回答问题：

{kb_context}

请按照以下步骤回答问题：
1. 理解问题的具体要求和背景
2. 根据你的中国文化和语言知识进行分析
3. 提供准确和详细的回答
4. 在回答结尾用中文"答案："标记最终答案

问题：{question}

分析和回答："""

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
            answer = text.split("ANSWER:")[-1].strip()
        elif "答案：" in text or "答案:" in text:
            # Handle Chinese answer markers
            match = re.search(r'(?:答案[：:])\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            else:
                answer = text.split("答案：")[-1].split("答案:")[-1].strip()

        # Additional Chinese extraction patterns for enhanced prompt
        if not answer.strip() and ("分析和回答：" in text or "分析：" in text):
            # Extract content after "分析和回答：" or "分析："
            parts = text.split("分析和回答：")
            if len(parts) > 1:
                answer = parts[-1]
            else:
                parts = text.split("分析：")
                if len(parts) > 1:
                    answer = parts[-1]

        if not answer.strip():
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
        """Main inference method with batched processing and fast algebra calculator"""

        if not questions:
            return []

        # Process algebra questions first with fast calculator
        results = [None] * len(questions)
        remaining_questions = []

        # First pass: handle algebra questions with calculator
        for i, q in enumerate(questions):
            if q.get('subject', 'default') == 'algebra':
                print(f"Attempting to solve algebra question with fast calculator...")
                fast_result = self._solve_algebra_fast(q['question'])

                if fast_result is not None:
                    # Use fast calculator result for simple expressions
                    results[i] = {
                        "questionID": q["questionID"],
                        "answer": fast_result
                    }
                    print(f"  ✅ Fast calculator success")
                else:
                    # Fall back to LLM for complex algebra
                    print(f"  ⚠️  Fast calculator not applicable, using LLM")
                    remaining_questions.append((i, q))
            else:
                # Non-algebra questions are added to remaining for batch processing
                remaining_questions.append((i, q))

        # Second pass: process remaining questions (non-algebra + complex algebra) with batching
        if remaining_questions:
            # Group remaining questions by subject for batch processing
            subject_batches = {}
            for idx, q in remaining_questions:
                subject = q.get('subject', 'default')
                if subject not in subject_batches:
                    subject_batches[subject] = []
                subject_batches[subject].append((idx, q))

            # Process each subject batch
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