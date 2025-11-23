"""
Tech Arena 2025 - Phase 2
Speed-Optimized High-Accuracy Pipeline
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
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots")
    return str(sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True)[0])


class FastAccuratePipeline:
    """Optimized for speed + accuracy"""

    def __init__(self):
        print("🚀 Loading Llama-3.2-3B with 4-bit quantization...")

        # 4-bit quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load 3B model for speed
        model_path = find_model_path("meta-llama/Llama-3.2-3B-Instruct", CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        print(f"✅ Ready! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB\n")

    def _create_prompt(self, question: str, subject: str) -> str:
        """Concise prompts for speed"""

        if subject == "algebra":
            content = f"""Solve step by step:

Example: 3x + 2 = 11 → 3x = 9 → x = 3

{question}"""

        elif subject == "chinese":
            content = f"""Answer accurately about Chinese language/culture:

{question}"""

        else:
            content = question

        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Fast batch generation"""
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
                max_new_tokens=100,  # Very short for maximum speed
                temperature=0,  # Greedy for fastest generation
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        answers = []
        for i, output in enumerate(outputs):
            generated = output[inputs['input_ids'][i].shape[0]:]
            answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            answers.append(answer)

        return answers

    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Fast batch processing"""
        if not questions:
            return []

        print(f"⚡ Processing {len(questions)} questions...")

        # Create prompts
        prompts = []
        for q in questions:
            subject = q.get('subject', '').lower()
            prompt = self._create_prompt(q['question'], subject)
            prompts.append(prompt)

        # Large batches for maximum speed
        batch_size = 16
        all_answers = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            answers = self._generate_batch(batch)
            all_answers.extend(answers)

        # Format results
        results = []
        for q, answer in zip(questions, all_answers):
            # Clean
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
    return FastAccuratePipeline()
