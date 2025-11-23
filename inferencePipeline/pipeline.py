"""
Tech Arena 2025 - Phase 2
Optimized Inference Pipeline
"""

import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path

CACHE_DIR = "/app/models"


def find_model_path(model_name: str, cache_dir: str) -> str:
    """Find model in HuggingFace cache"""
    cache_path = Path(cache_dir)
    hf_cache_name = "models--" + model_name.replace("/", "--")
    model_cache = cache_path / hf_cache_name

    if not model_cache.exists():
        raise FileNotFoundError(f"Model not found: {model_cache}")

    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots: {snapshots_dir}")

    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {snapshots_dir}")

    return str(sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True)[0])


class InferencePipeline:
    """Fast and accurate inference pipeline"""

    def __init__(self):
        print("🚀 Loading Llama-3.2-3B with 4-bit quantization...")

        # 4-bit NF4 quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load model with on-the-fly quantization
        model_path = find_model_path("meta-llama/Llama-3.2-3B-Instruct", CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Fix for decoder-only models
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        print(f"✅ Ready! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB\n")

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate answers for a batch of prompts"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05
            )

        answers = []
        for i, output in enumerate(outputs):
            generated = output[inputs['input_ids'][i].shape[0]:]
            answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            answers.append(answer)

        return answers

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process all questions"""
        if not questions:
            return []

        print(f"⚡ Processing {len(questions)} questions...")

        # Create prompts - just the question directly
        prompts = []
        for q in questions:
            messages = [{"role": "user", "content": q['question']}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        # Batch process (8 at a time for speed)
        batch_size = 8
        all_answers = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            answers = self._generate_batch(batch)
            all_answers.extend(answers)

        # Format results
        results = []
        for q, answer in zip(questions, all_answers):
            # Clean answer
            answer = answer.replace("<think>", "").replace("</think>", "")
            answer = answer.replace("assistant\n\n", "").replace("assistant\n", "")
            answer = answer.strip()

            if len(answer) > 5000:
                answer = answer[:5000]

            results.append({
                "questionID": q["questionID"],
                "answer": answer
            })

        print(f"✅ Done!\n")
        return results


def loadPipeline():
    """Entry point"""
    return InferencePipeline()
