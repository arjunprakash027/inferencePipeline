"""
Optimized Quantized Pipeline - Tech Arena 2025
8-bit quantization for best accuracy-memory tradeoff
Uses Llama-3.2-3B with bitsandbytes quantization
"""
import torch
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'

torch.set_num_threads(16)
torch.set_num_interop_threads(4)

class OptimizedInferencePipeline:
    def __init__(self):
        print("[INIT] Initializing 8-bit quantized pipeline...")
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.batch_size = 24  # Can go higher with quantization
        self.max_length = 200
        print(f"[INIT] Device: {self.device}, Batch: {self.batch_size}")
        
        print("[LOAD] Started loading model...")
        self.load_model()
        
    def load_model(self):
        print("[LOAD] Loading 8-bit quantized model...")
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        cache_dir = '/app/models'
        #cache_dir = './app/model/Llama-3.2-1B-Instruct'
        
        # 8-bit quantization config - best accuracy-memory tradeoff
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("[LOAD] Loading with 8-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            quantization_config=quantization_config,
            device_map={"":"cpu"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        self.model.config.use_cache = True
        print("[LOAD] 8-bit quantized model loaded!")
        
    def create_prompt(self, question: str, subject_hint: str = "") -> str:
        question_lower = question.lower()
        
        is_algebra = any(word in question_lower for word in ['solve', 'calculate', 'equation', 'algebra', 
                                                               'factor', 'simplify', 'expression']) or \
                     any(char in question for char in ['+', '-', '*', '/', '=', '^'])
        
        is_geography = any(word in question_lower for word in ['capital', 'country', 'city', 'continent', 
                                                                 'geography', 'ocean', 'mountain', 'river', 
                                                                 'located', 'where', 'border'])
        
        is_history = any(word in question_lower for word in ['when', 'who', 'history', 'war', 'century', 
                                                               'year', 'historical', 'ancient', 'battle', 
                                                               'president', 'king'])
        
        if is_algebra:
            system_msg = "You are a mathematics expert. Solve problems step-by-step and provide clear final answers."
        elif is_geography:
            system_msg = "You are a geography expert. Provide accurate, specific information about locations, capitals, and geographic features."
        elif is_history:
            system_msg = "You are a history expert. Provide accurate dates, names, and historical facts. Be specific about time periods and key figures."
        else:
            system_msg = "You are an educational assistant. Provide clear, accurate, and well-reasoned answers."
        
        instruction = "Answer within 4000 tokens."        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg} {instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def process_batch(self, questions: List[str]) -> List[str]:
        prompts = [self.create_prompt(q) for q in questions]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.05,
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
        print(f"[PIPELINE] Processing {len(questions)} questions...")
        
        if self.model is None:
            self.load_model()
        
        all_answers = []
        question_texts = [q['question'] for q in questions]
        
        for i in range(0, len(question_texts), self.batch_size):
            batch_questions = question_texts[i:i+self.batch_size]
            batch_ids = questions[i:i+self.batch_size]
            
            try:
                batch_answers = self.process_batch(batch_questions)
                
                for qid, answer in zip(batch_ids, batch_answers):
                    all_answers.append({
                        "questionID": qid['questionID'],
                        "answer": answer
                    })
                
                if i > 0 and i % 100 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"[ERROR] Batch {i}: {e}")
                for qid in batch_ids:
                    all_answers.append({
                        "questionID": qid['questionID'],
                        "answer": "Unable to generate answer."
                    })
        
        print(f"[PIPELINE] Completed!")
        return all_answers

def loadPipeline():
    print("Tech Arena 2025 - 8-bit Quantized Pipeline")
    return OptimizedInferencePipeline()