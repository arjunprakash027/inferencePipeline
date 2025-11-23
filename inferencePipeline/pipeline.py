"""
Tech Arena 2025 - Phase 2
Optimized Multi-Model Inference Pipeline with On-the-Fly Quantization

Strategy:
- Subject-specific model routing for optimal accuracy
- On-the-fly 4-bit quantization using bitsandbytes
- Qwen3-4B for Algebra & Chinese (strong math & Chinese capabilities)
- Llama-3.2-3B for Geography & History (faster for factual queries)
- Optimized prompts per subject
- Batch processing for efficiency
"""

import os
import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path


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


class SubjectRouter:
    """Routes questions to the appropriate model based on subject"""

    def __init__(self):
        """Initialize models with on-the-fly 4-bit quantization"""
        print("🚀 Initializing Multi-Model Pipeline with On-the-Fly Quantization...")

        # Configure 4-bit quantization with bitsandbytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for better compression
            bnb_4bit_quant_type="nf4"  # NormalFloat4 - optimal for inference
        )

        # Model 1: Qwen3-1.7B for Algebra & Chinese
        print("📦 Loading Qwen3-1.7B (4-bit) for Algebra & Chinese...")
        qwen_path = find_model_path("Qwen/Qwen3-1.7B", CACHE_DIR)
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            trust_remote_code=True
        )
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.qwen_model.eval()
        print(f"✅ Qwen3-1.7B loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB)")

        # Model 2: Llama-3.2-3B for Geography & History
        print("📦 Loading Llama-3.2-3B (4-bit) for Geography & History...")
        llama_path = find_model_path("meta-llama/Llama-3.2-3B-Instruct", CACHE_DIR)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llama_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.llama_model.eval()
        print(f"✅ Llama-3.2-3B loaded (Total VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB)")

        print("✅ Multi-Model Pipeline Ready!\n")

    def _create_algebra_prompt(self, question: str) -> str:
        """Simple prompt for algebra"""
        return f"""Solve this math problem step by step and provide the final answer.

{question}"""

    def _create_chinese_prompt(self, question: str) -> str:
        """Simple prompt for Chinese"""
        return f"""Answer this question about Chinese language or culture accurately.

{question}"""

    def _create_geography_prompt(self, question: str) -> str:
        """Simple prompt for geography"""
        return f"""Answer this geography question accurately.

{question}"""

    def _create_history_prompt(self, question: str) -> str:
        """Simple prompt for history"""
        return f"""Answer this history question accurately.

{question}"""

    def _format_qwen_prompt(self, content: str) -> str:
        """Format prompt for Qwen model"""
        messages = [
            {"role": "user", "content": content}
        ]
        return self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _format_llama_prompt(self, content: str) -> str:
        """Format prompt for Llama model"""
        messages = [
            {"role": "user", "content": content}
        ]
        return self.llama_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _generate_response(self, model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from model"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for accuracy
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def _clean_answer(self, text: str) -> str:
        """Clean and extract final answer"""
        # Remove common artifacts
        text = re.sub(r'<\|.*?\|>|</?think>|<s>|</s>', '', text, flags=re.IGNORECASE)
        text = text.strip()

        # Extract answer after markers
        markers = ["Answer:", "答案:", "答案：", "Final Answer:", "Solution:", "Therefore,", "Thus,"]
        for marker in markers:
            if marker in text:
                text = text.split(marker)[-1].strip()
                break

        # Limit to 5000 characters
        if len(text) > 5000:
            text = text[:5000]

        return text.strip()

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process questions with subject-specific routing"""
        if not questions:
            return []

        print(f"🔄 Processing {len(questions)} questions with subject-specific routing...")
        results = []

        for i, q in enumerate(questions):
            question_text = q['question']
            subject = q.get('subject', 'default').lower()
            question_id = q['questionID']

            try:
                # Route to appropriate model based on subject
                if subject == "algebra":
                    prompt_content = self._create_algebra_prompt(question_text)
                    formatted_prompt = self._format_qwen_prompt(prompt_content)
                    response = self._generate_response(
                        self.qwen_model,
                        self.qwen_tokenizer,
                        formatted_prompt,
                        max_new_tokens=600  # More tokens for step-by-step math
                    )

                elif subject == "chinese":
                    prompt_content = self._create_chinese_prompt(question_text)
                    formatted_prompt = self._format_qwen_prompt(prompt_content)
                    response = self._generate_response(
                        self.qwen_model,
                        self.qwen_tokenizer,
                        formatted_prompt,
                        max_new_tokens=512
                    )

                elif subject == "geography":
                    prompt_content = self._create_geography_prompt(question_text)
                    formatted_prompt = self._format_llama_prompt(prompt_content)
                    response = self._generate_response(
                        self.llama_model,
                        self.llama_tokenizer,
                        formatted_prompt,
                        max_new_tokens=400
                    )

                elif subject == "history":
                    prompt_content = self._create_history_prompt(question_text)
                    formatted_prompt = self._format_llama_prompt(prompt_content)
                    response = self._generate_response(
                        self.llama_model,
                        self.llama_tokenizer,
                        formatted_prompt,
                        max_new_tokens=400
                    )

                else:
                    # Default to Qwen for unknown subjects
                    prompt_content = f"Answer this question accurately:\n\n{question_text}\n\nAnswer:"
                    formatted_prompt = self._format_qwen_prompt(prompt_content)
                    response = self._generate_response(
                        self.qwen_model,
                        self.qwen_tokenizer,
                        formatted_prompt
                    )

                # Clean and format answer
                clean_answer = self._clean_answer(response)
                results.append({
                    "questionID": question_id,
                    "answer": clean_answer
                })

                if (i + 1) % 10 == 0:
                    print(f"✅ Processed {i + 1}/{len(questions)} questions...")

            except Exception as e:
                print(f"⚠️ Error processing question {question_id}: {e}")
                # Fallback: return a generic answer
                results.append({
                    "questionID": question_id,
                    "answer": "Unable to process this question."
                })

        print(f"✅ Completed all {len(results)} questions!\n")
        return results


def loadPipeline():
    """Entry point for evaluation system"""
    return SubjectRouter()
