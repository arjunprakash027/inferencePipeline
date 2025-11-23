"""
Tech Arena 2025 - Phase 2
High-Accuracy Pipeline with Llama-3.1-8B
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


class HighAccuracyPipeline:
    """Maximum accuracy with Llama-3.1-8B"""

    def __init__(self):
        print("🚀 Loading Llama-3.1-8B with 4-bit quantization...")

        # Aggressive 4-bit quantization for 8B model
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load Llama-3.1-8B with on-the-fly quantization
        model_path = find_model_path("meta-llama/Llama-3.1-8B-Instruct", CACHE_DIR)
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
        """Create subject-optimized prompts"""

        if subject == "algebra":
            content = f"""Solve this algebra problem step by step. Show your work clearly and state the final answer.

Example:
Q: If 3x + 2 = 11, what is x?
A: Step 1: Subtract 2 from both sides
   3x + 2 - 2 = 11 - 2
   3x = 9

   Step 2: Divide both sides by 3
   x = 9 ÷ 3
   x = 3

   Final answer: x = 3

Now solve:
{question}"""

        elif subject == "chinese":
            content = f"""You are an expert in Chinese language, literature, history, and culture. Provide accurate and detailed information.

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
        """Generate answers with quality settings"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.15,  # Low for accuracy
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
        """Process questions with high accuracy"""
        if not questions:
            return []

        print(f"⚡ Processing {len(questions)} questions...")

        # Create prompts
        prompts = []
        for q in questions:
            subject = q.get('subject', '').lower()
            prompt = self._create_prompt(q['question'], subject)
            prompts.append(prompt)

        # Batch process (smaller batches for 8B model)
        batch_size = 4
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
    return HighAccuracyPipeline()
