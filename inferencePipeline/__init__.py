"""
Simplified Optimized Pipeline - Focus on Core Speedups
No fancy imports that might break
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import gc

# Disable warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class OptimizedInferencePipeline:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.batch_size = 32
        
        # CRITICAL: Reduced token limits for speed
        self.token_limits = {
            'algebra': 120,
            'history': 100,
            'geography': 70,
            'general': 80
        }
        
    def load_model(self):
        """Load model with basic optimizations"""
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        cache_dir = '/app/models'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        
        # Try torch compile (safe fallback)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("[COMPILE] Model compiled successfully")
        except:
            print("[COMPILE] Compilation not available, using standard model")
    
    def create_expert_prompt(self, question: str, subject: str) -> str:
        """
        MINIMAL PROMPTS - This is the KEY speedup!
        Removed all few-shot examples.
        """
        
        systems = {
            'algebra': "You are an algebra expert. Show clear steps.",
            'geography': "You are a geography expert. Give precise facts.",
            'history': "You are a history expert. Provide dates and context.",
            'general': "Answer accurately and concisely."
        }
        
        system_msg = systems.get(subject, systems['general'])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def detect_subject(self, question: str) -> str:
        """Fast subject detection"""
        q = question.lower()
        
        if any(kw in q for kw in ['solve', 'equation', 'factor', 'matrix', 'simplify', 'calculate']):
            return 'algebra'
        
        if any(kw in q for kw in ['country', 'capital', 'mountain', 'river', 'ocean', 'continent']):
            return 'geography'
        
        if any(kw in q for kw in ['war', 'revolution', 'century', 'when did', 'who was']):
            return 'history'
        
        return 'general'
    
    def process_batch(self, questions: List[str], subject: str) -> List[str]:
        """Process batch"""
        
        max_tokens = self.token_limits[subject]
        prompts = [self.create_expert_prompt(q, subject) for q in questions]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        answers = []
        for i, output in enumerate(outputs):
            generated_tokens = output[inputs['input_ids'][i].shape[0]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if len(answer) > 5000:
                answer = answer[:5000]
            
            answers.append(answer)
        
        return answers
    
    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """Main inference"""
        
        if self.model is None:
            self.load_model()
        
        # Group by subject
        subject_groups = {
            'algebra': [],
            'geography': [],
            'history': [],
            'general': []
        }
        
        for q in questions:
            subject = self.detect_subject(q['question'])
            subject_groups[subject].append(q)
        
        all_results = []
        
        for subject, subject_questions in subject_groups.items():
            if not subject_questions:
                continue
            
            questions_text = [q['question'] for q in subject_questions]
            question_ids = [q['questionID'] for q in subject_questions]
            
            # Process in batches
            for i in range(0, len(questions_text), self.batch_size):
                batch_qs = questions_text[i:i+self.batch_size]
                batch_ids = question_ids[i:i+self.batch_size]
                
                try:
                    batch_answers = self.process_batch(batch_qs, subject)
                    
                    for qid, answer in zip(batch_ids, batch_answers):
                        all_results.append({"questionID": qid, "answer": answer})
                        
                except Exception as e:
                    print(f"[ERROR] {e}")
                    for qid in batch_ids:
                        all_results.append({"questionID": qid, "answer": "Error."})
                
                if i % 32 == 0 and i > 0:
                    gc.collect()
        
        # Return in original order
        result_dict = {r['questionID']: r for r in all_results}
        return [result_dict[q['questionID']] for q in questions]


def loadPipeline():
    return OptimizedInferencePipeline()