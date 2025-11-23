"""
Tech Arena 2025 - Phase 2
High Accuracy LLM Inference Pipeline

Strategy:
- Use Llama-3.1-8B for maximum accuracy across all subjects
- Enhanced prompts with detailed examples and context
- Low temperature for deterministic responses
- Specialized handling for algebra and Chinese
"""

import os
import re
from typing import List, Dict
from vllm import LLM, SamplingParams
from pathlib import Path

# Configuration - Using Llama-3.1-8B for maximum accuracy
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
CACHE_DIR = "/app/models"

MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "dtype": "float16",
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "gpu_memory_utilization": 0.85,
    },
    "default": {
        "dtype": "float16",
        "quantization": None,
        "load_format": "auto",
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


class InferencePipeline:
    """High accuracy inference pipeline"""

    def __init__(self):
        """Initialize pipeline with vLLM for maximum accuracy"""

        raw_model_path = find_model_path(MODEL_NAME, CACHE_DIR)
        model_config = MODEL_CONFIGS.get(MODEL_NAME, MODEL_CONFIGS["default"])

        print(f" Loading {MODEL_NAME.split('/')[-1]} with vLLM (dtype={model_config['dtype']}, quantization={model_config.get('quantization')})...")

        self.llm = LLM(
            model=raw_model_path,
            dtype=model_config.get("dtype", "float16"),
            quantization=model_config.get("quantization", None),
            load_format=model_config.get("load_format", "auto"),
            gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.85),
            max_model_len=4096,
            enforce_eager=True,  # More stable on T4 with larger model
            max_num_seqs=16,  # Reduced batch size for 8B model
            trust_remote_code=True,
            tensor_parallel_size=1,
            disable_log_stats=True,
            seed=42  # For reproducible results
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Low temperature sampling parameters for maximum accuracy
        self.sampling_params = SamplingParams(
            temperature=0.05,  # Very low temperature for consistent, accurate answers
            top_p=0.95,
            top_k=-1,  # Use probability distribution instead of top-k
            max_tokens=800,  # Allow more tokens for complex answers
            stop=["</s>", "<|eot_id|>", "\n\nQuestion:", "\n\nProblem:", "\n\n问题"],
            presence_penalty=0.1,
            frequency_penalty=0.1
        )

        print("✅ High accuracy pipeline ready\n")

    def _create_prompt(self, question: str, subject: str) -> str:
        """Create highly effective prompts with examples and context for maximum accuracy"""

        if subject == "algebra":
            # Highly structured algebra prompt with examples
            prompt = f"""You are an expert mathematician. Solve the following algebra problem step by step with detailed reasoning.

INSTRUCTIONS:
1. Show your step-by-step work
2. Use proper algebraic methods
3. Box or clearly state the final answer
4. Be precise and accurate

EXAMPLES:
Example 1: If 2x + 3 = 7, then 2x = 4, so x = 2.
Example 2: If 3(x - 2) = 9, then x - 2 = 3, so x = 5.

QUESTION:
{question}

STEP-BY-STEP SOLUTION:
"""

        elif subject == "chinese":
            # Expert Chinese language and culture prompt
            prompt = f"""You are an expert in Chinese language, culture, and history. Provide accurate information about Chinese topics.

INSTRUCTIONS:
1. Use proper Chinese characters when appropriate
2. Provide culturally accurate information
3. Give clear, factual answers
4. Respect Chinese linguistic and cultural nuances

QUESTION:
{question}

ANSWER:
"""

        elif subject == "geography":
            # Factual geography prompt
            prompt = f"""You are an expert geographer. Provide accurate geographical information based on current data.

INSTRUCTIONS:
1. Provide factual information
2. Use proper country, city, and location names
3. Base answers on current geographical data
4. Be precise and accurate

QUESTION:
{question}

ANSWER:
"""

        elif subject == "history":
            # Factual history prompt
            prompt = f"""You are an expert historian. Provide accurate historical information based on documented facts.

INSTRUCTIONS:
1. Provide factually accurate information
2. Use proper dates, names, and historical context
3. Base answers on documented historical records
4. Be precise and avoid speculation

QUESTION:
{question}

ANSWER:
"""

        else:
            # Default prompt for any other subject
            prompt = f"""Answer the following question accurately and completely.

INSTRUCTIONS:
1. Provide accurate information
2. Be thorough but concise
3. Base answers on facts, not assumptions
4. Be precise

QUESTION:
{question}

ANSWER:
"""

        # Format with chat template
        messages = [
            {"role": "system", "content": "You are a highly accurate expert assistant. Provide precise, factual answers."},
            {"role": "user", "content": prompt}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_answer(self, text: str, subject: str) -> str:
        """Extract clean, accurate answer from model response"""

        # Remove common tokenization artifacts and special tags
        text = re.sub(r'<s>|</s>|<\|.*?\|>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?i)</?think>', '', text)
        text = text.strip()

        # Look for specific markers to extract the final answer
        if "ANSWER:" in text:
            # Extract everything after the last ANSWER: marker
            answer = text.split("ANSWER:")[-1].strip()
        elif "答案：" in text:
            # Chinese answer marker
            answer = text.split("答案：")[-1].strip()
        elif "答案:" in text:
            # Alternative Chinese answer marker
            answer = text.split("答案:")[-1].strip()
        elif "\n\n" in text and "STEP-BY-STEP SOLUTION:" in text:
            # For algebra, take the final part after step-by-step working
            parts = text.split("STEP-BY-STEP SOLUTION:")
            if len(parts) > 1:
                working = parts[1]
                # Look for the final answer in the working
                lines = working.split('\n')
                # Look for the last line that looks like an answer
                potential_answers = []
                for line in reversed(lines):
                    line = line.strip()
                    if line and (line.startswith("So ") or line.startswith("Therefore") or line.startswith("The answer") or line.startswith("Thus") or "=" in line or "x =" in line or "x=" in line):
                        potential_answers.append(line)
                        break
                if potential_answers:
                    answer = potential_answers[0]
                else:
                    answer = working
            else:
                answer = text
        else:
            # Fallback: extract the most relevant last part of the response
            # Split by common separators and take the last meaningful segment
            segments = re.split(r'\n\s*\n|QUESTION:|PROBLEM:', text)
            answer = segments[-1].strip()
        
        # Clean up the answer
        answer = answer.strip()
        # Remove any remaining prefixes that might be part of the model's output
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        elif answer.lower().startswith("response:"):
            answer = answer[9:].strip()
        elif answer.lower().startswith("solution:"):
            answer = answer[9:].strip()
        
        # Ensure proper formatting and length
        answer = answer.strip()
        if len(answer) > 5000:
            answer = answer[:5000]
        
        return answer

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Main inference method with accuracy-focused processing"""

        if not questions:
            return []

        print(f"Processing {len(questions)} questions with high accuracy focus...")

        # Create all prompts
        prompts = []
        for q in questions:
            prompt = self._create_prompt(q['question'], q.get('subject', 'default'))
            prompts.append(prompt)

        # Generate all responses at once for efficiency
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        # Extract answers
        results = []
        for i, (output, q) in enumerate(zip(outputs, questions)):
            raw_answer = output.outputs[0].text.strip()
            clean_answer = self._extract_answer(raw_answer, q.get('subject', 'default'))
            
            results.append({
                "questionID": q["questionID"],
                "answer": clean_answer
            })
            
            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"Processed {i + 1}/{len(questions)} questions...")

        print(f"✅ Completed {len(results)} questions with high accuracy focus\n")
        return results


def loadPipeline():
    """Entry point for evaluation system - returns high accuracy pipeline"""
    return InferencePipeline()