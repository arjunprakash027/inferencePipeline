
"""
Optimized LLM Inference Pipeline for Tech Arena 2025
Uses quantization, batching, and parallel processing for maximum efficiency.
"""

import torch
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
import os
from concurrent.futures import ThreadPoolExecutor
import gc

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

class OptimizedInferencePipeline:
    def __init__(self):
        print("Initializing Optimized Inference Pipeline...")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.max_length = 200
        
    def load_model(self):
        """Load quantized model with optimal settings"""
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        cache_dir = '/app/models'
        #cache_dir = './app/model'
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        
        # 4-bit quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        print("Model loaded successfully!")
        
    def create_prompt(self, question: str, subject_hint: str = "") -> str:
        """Create optimized prompt based on question type"""
        # Detect subject from question
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['solve', 'calculate', 'equation', 'x=', 'algebra', '+', '-', '*', '/', 'factor', 'simplify']):
            system_msg = "You are a math expert. Solve the problem step by step and provide the final answer clearly."
        elif any(word in question_lower for word in ['capital', 'country', 'city', 'continent', 'geography', 'ocean', 'mountain', 'river', 'located']):
            system_msg = "You are a geography expert. Provide accurate, concise answers about locations, countries, and geography."
        elif any(word in question_lower for word in ['when', 'who', 'history', 'war', 'century', 'year', 'historical', 'ancient', 'empire']):
            system_msg = "You are a history expert. Provide accurate, concise answers about historical events, dates, and figures."
        else:
            system_msg = "You are a helpful educational assistant. Provide clear, accurate, and concise answers."
        
        # Llama 3.2 instruction format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def process_batch(self, questions: List[str]) -> List[str]:
        """Process a batch of questions efficiently"""
        # Create prompts
        prompts = [self.create_prompt(q) for q in questions]
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate with optimal settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                do_sample=False,  # Greedy for consistency
                temperature=1.0,
                top_p=1.0,
                num_beams=1,  # Fast, no beam search
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache
            )
        
        # Decode outputs
        answers = []
        for i, output in enumerate(outputs):
            # Remove input prompt from output
            generated_tokens = output[inputs['input_ids'][i].shape[0]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            answer = answer.strip()
            
            # Truncate to 5000 characters as per requirements
            if len(answer) > 5000:
                answer = answer[:5000]
            
            answers.append(answer)
        
        return answers
    
    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """Main inference function"""
        print(f"Processing {len(questions)} questions...")
        
        if self.model is None:
            self.load_model()
        
        all_answers = []
        question_texts = [q['question'] for q in questions]
        
        # Process in batches
        for i in range(0, len(question_texts), self.batch_size):
            batch_questions = question_texts[i:i+self.batch_size]
            batch_ids = questions[i:i+self.batch_size]
            
            try:
                batch_answers = self.process_batch(batch_questions)
                
                # Format answers
                for qid, answer in zip(batch_ids, batch_answers):
                    all_answers.append({
                        "questionID": qid['questionID'],
                        "answer": answer
                    })
                
                # Progress indicator
                if (i + self.batch_size) % 50 == 0:
                    print(f"  Processed {min(i + self.batch_size, len(questions))}/{len(questions)} questions")
                
                # Clear cache periodically
                if i % 100 == 0 and i > 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Return empty answers for failed batch
                for qid in batch_ids:
                    all_answers.append({
                        "questionID": qid['questionID'],
                        "answer": "Unable to generate answer."
                    })
        
        print(f"Completed! Generated {len(all_answers)} answers")
        return all_answers


def loadPipeline():
    """
    Factory function to create and return the inference pipeline.
    This is the entry point called by run.py
    """
    pipeline = OptimizedInferencePipeline()
    return pipeline
