"""
Tech Arena 2025 - Phase 2
Fast Inference with On-the-Fly 4-bit Quantization

Strategy:
- Single Qwen3-1.7B model for all subjects (good balance of speed & accuracy)
- On-the-fly 4-bit quantization using bitsandbytes
- Optimized generation parameters for speed
- Subject-specific prompts with examples
"""

import os
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


class FastPipeline:
    """Fast inference with on-the-fly quantization"""

    def __init__(self):
        """Initialize with 4-bit quantization"""
        print("🚀 Loading Qwen3-1.7B with on-the-fly 4-bit quantization...")

        # 4-bit quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load model
        model_path = find_model_path("Qwen/Qwen3-1.7B", CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        print(f"✅ Model loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB)\n")

    def _create_prompt(self, question: str, subject: str) -> str:
        """Create optimized prompts"""

        if subject == "algebra":
            prompt = f"""Solve step by step:

{question}

Answer:"""

        elif subject == "chinese":
            prompt = f"""Answer accurately:

{question}

Answer:"""

        elif subject == "geography":
            prompt = f"""{question}

Answer:"""

        elif subject == "history":
            prompt = f"""{question}

Answer:"""

        else:
            prompt = f"""{question}

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _batch_generate(self, prompts: List[str]) -> List[str]:
        """Fast batch generation"""
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate with optimized parameters
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # Shorter for speed
                temperature=0.2,  # Low for accuracy
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # No beam search for speed
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode all outputs
        responses = []
        for i, output in enumerate(outputs):
            # Get only generated part
            generated_ids = output[inputs['input_ids'][i].shape[0]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())

        return responses

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process questions in batches"""
        if not questions:
            return []

        print(f"⚡ Processing {len(questions)} questions...")

        # Create prompts
        prompts = [self._create_prompt(q['question'], q.get('subject', 'default')) for q in questions]

        # Batch process (4 at a time for T4 GPU)
        batch_size = 4
        all_responses = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = self._batch_generate(batch_prompts)
            all_responses.extend(batch_responses)

            if (i + batch_size) % 10 == 0:
                print(f"Processed {min(i+batch_size, len(prompts))}/{len(prompts)}...")

        # Format results
        results = []
        for q, answer in zip(questions, all_responses):
            # Clean answer - remove think tags and artifacts
            answer = answer.replace("<think>", "").replace("</think>", "")
            answer = answer.replace("<|im_end|>", "").replace("</s>", "")
            answer = answer.strip()

            if len(answer) > 5000:
                answer = answer[:5000]

            results.append({
                "questionID": q["questionID"],
                "answer": answer
            })

        print(f"✅ Completed {len(results)} questions!\n")
        return results


def loadPipeline():
    """Entry point"""
    return FastPipeline()
