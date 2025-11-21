"""
Local Transformers-based Inference Pipeline
Uses HuggingFace transformers for CPU-only inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import gc
from pathlib import Path

# Disable warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load settings at module import time (before any inference)
from . import get_settings
SETTINGS = get_settings()


class LocalInferencePipeline:
    """
    Transformers-based inference pipeline for local CPU execution
    """
    def __init__(self):
        """Initialize and LOAD MODEL immediately (not timed)"""
        local_cfg = SETTINGS['local']
        inference_cfg = SETTINGS['inference']
        
        self.device = local_cfg['device']
        self.batch_size = local_cfg['batch_size']
        
        # Token limits from settings
        self.token_limits = inference_cfg['token_limits']
        
        # LOAD MODEL IMMEDIATELY IN __INIT__
        print(f"[INIT] Loading model (this happens before timing starts)...")
        self.model, self.tokenizer = self._load_model()
        print(f"[INIT] ✓ Model loaded and ready for inference!")
    
    def _load_model(self):
        """
        PRIVATE method called ONLY from __init__
        Returns (model, tokenizer) tuple
        """
        model_cfg = SETTINGS['model']
        local_cfg = SETTINGS['local']
        
        model_name = model_cfg['name']
        cache_dir = model_cfg['cache_dir']
        
        # Load tokenizer
        print(f"[LOAD] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[LOAD] ✓ Tokenizer loaded")
        
        # Load transformers model
        print(f"[LOAD] Using transformers for local fallback")
        
        # Map string dtype to torch dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        torch_dtype = dtype_map.get(local_cfg['torch_dtype'], torch.bfloat16)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=local_cfg['low_cpu_mem_usage'],
        )
        model.eval()
        
        if local_cfg.get('use_torch_compile', False):
            try:
                compile_mode = local_cfg.get('compile_mode', 'reduce-overhead')
                print(f"[LOAD] Compiling model with torch.compile (mode={compile_mode})...")
                model = torch.compile(model, mode=compile_mode)
            except Exception as e:
                print(f"[LOAD] torch.compile failed: {e}, using uncompiled model")
        
        return model, tokenizer
    
    def create_expert_prompt(self, question: str, subject: str) -> str:
        """Enhanced prompts with few-shot examples"""
        
        if subject == "algebra":
            few_shot = """Examples:

Q: What kind of function has a graph that is a straight line?
A: A linear function: y = mx + b.

Q: Could you explain why matrices help solve { x + y = 5 ; x - y = 1 }?
A: Matrices organize this as A·v = b where A = [[1,1], [1,-1]], v = [x,y]ᵀ, b = [5,1]ᵀ. Solve via A⁻¹b: x = 3, y = 2.

Q: Factor x² + 7x + 12
A: Two numbers that multiply to 12 and add to 7 are 3 and 4. So x² + 7x + 12 = (x + 3)(x + 4).

Now answer:"""
            system = "You are an expert algebra tutor. Show clear mathematical reasoning."
            
        elif subject == "geography":
            few_shot = """Examples:

Q: What's the highest point in South America?
A: Aconcagua (6,961 m) in Argentina.

Q: Which country is known as the 'Land of the Thunder Dragon'?
A: Bhutan.

Q: What are the Llanos?
A: Vast tropical grasslands in Venezuela and eastern Colombia.

Q: What is the largest hot desert?
A: The Sahara in North Africa, approximately 9 million km².

Now answer:"""
            system = "You are an expert geographer. Provide precise facts and locations."
            
        elif subject == "history":
            few_shot = """Examples:

Q: When did World War II end?
A: 1945. Germany surrendered May 7-8 (V-E Day), Japan August 15 (V-J Day).

Q: In what ways did the Industrial Revolution influence political ideologies?
A: It fueled classical liberalism (free markets, rights), sparked socialism and Marxism (worker response to inequality), and provoked conservative reactions. Led to labor movements and social democracy.

Q: Who was the first US President?
A: George Washington (1789-1797).

Now answer:"""
            system = "You are an expert historian. Provide dates, causes, effects, and context."
            
        else:
            few_shot = """Example:
Q: What is photosynthesis?
A: Process where plants use sunlight, water, and CO₂ to produce glucose and oxygen.

Now answer:"""
            system = "You are a knowledgeable educational assistant."
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{few_shot}

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def detect_subject(self, question: str) -> str:
        """Fast subject detection"""
        q = question.lower()
        
        if any(kw in q for kw in ['solve', 'equation', 'factor', 'matrix', 'simplify']):
            return 'algebra'
        if any(kw in q for kw in ['country', 'capital', 'mountain', 'river', 'ocean']):
            return 'geography'
        if any(kw in q for kw in ['war', 'revolution', 'when did', 'who was', 'industrial']):
            return 'history'
        
        scores = {
            'algebra': sum(1 for kw in ['function', 'graph', 'calculate', 'variable'] if kw in q),
            'geography': sum(1 for kw in ['located', 'land', 'region', 'world'] if kw in q),
            'history': sum(1 for kw in ['century', 'ancient', 'influence', 'period'] if kw in q)
        }
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'

    def process_batch_transformers(self, questions: List[str], subject: str) -> List[str]:
        """Process batch using transformers - PURE INFERENCE (silent)"""
        gen_cfg = SETTINGS['inference']['generation']
        max_tokens = self.token_limits[subject]
        prompts = [self.create_expert_prompt(q, subject) for q in questions]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=gen_cfg['do_sample'],
                num_beams=gen_cfg['num_beams'],
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
        """Main inference function - silent, no console output."""
        batch_size = 16
        subject_groups = {'algebra': [], 'geography': [], 'history': [], 'general': []}
        
        for q in questions:
            subject = self.detect_subject(q['question'])
            subject_groups[subject].append(q)
        
        all_results = []
        
        for subject, subject_questions in subject_groups.items():
            if not subject_questions:
                continue
            
            questions_text = [q['question'] for q in subject_questions]
            question_ids = [q['questionID'] for q in subject_questions]
            
            for i in range(0, len(questions_text), self.batch_size):
                batch_qs = questions_text[i:i+self.batch_size]
                batch_ids = question_ids[i:i+batch_size]
                try:
                    batch_answers = self.process_batch_transformers(batch_qs, subject)
                    for qid, answer in zip(batch_ids, batch_answers):
                        all_results.append({"questionID": qid, "answer": answer})
                except Exception:
                    for qid in batch_ids:
                        all_results.append({"questionID": qid, "answer": "Error."})
                
                if i % 192 == 0 and i > 0:
                    gc.collect()
        
        result_dict = {r['questionID']: r for r in all_results}
        return [result_dict[q['questionID']] for q in questions]


def loadLocalPipeline():
    """
    Factory function - model loads HERE (before timing)
    Returns ready-to-use pipeline
    """
    print("[FACTORY] Creating and initializing local pipeline...")
    pipeline = LocalInferencePipeline()  # Model loads in __init__
    print("[FACTORY] ✓ Local pipeline ready for timed inference!")
    return pipeline
