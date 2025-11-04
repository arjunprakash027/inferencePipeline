"""
Highly Optimized LLM Inference Pipeline for Tech Arena 2025
Focus: Speed + Accuracy through prompt engineering and parallelization
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Disable unnecessary warnings and features
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class OptimizedInferencePipeline:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.batch_size = 32  # Larger batches for CPU efficiency
        self.max_new_tokens = 80  # Balanced for quality answers
        
    def load_model(self):
        """Load model with CPU optimizations"""
        # Use Llama 3.2 1B for best balance of speed/accuracy
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cache_dir = '/app/models'
        
        # Load tokenizer with left padding for decoder models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        # CRITICAL: Left padding for batch inference
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with CPU optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.float32,  # FP32 for CPU stability
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        
        # Enable torch.compile for CPU optimization (Python 3.12 + PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except:
            pass  # Fallback if compile fails
    
    def create_expert_prompt(self, question: str, subject: str) -> str:
        """
        Expert-level prompt engineering for maximum accuracy.
        Each subject gets specialized instructions.
        """
        
        if subject == "algebra":
            system_prompt = """You are an expert mathematics tutor specializing in algebra. 
Your answers must be:
- Precise and mathematically accurate
- Concise but complete
- Include final answers clearly
- Show key steps for equations
- Use standard mathematical notation

Answer the question directly and accurately."""
            
        elif subject == "geography":
            system_prompt = """You are an expert geographer with comprehensive knowledge of world geography.
Your answers must be:
- Factually accurate with specific details
- Include relevant numbers (elevation, area, population) when applicable
- Mention locations precisely (countries, regions, coordinates)
- Be concise but informative

Answer the question directly and accurately."""
            
        elif subject == "history":
            system_prompt = """You are an expert historian with deep knowledge of world history.
Your answers must be:
- Historically accurate with specific dates/periods when relevant
- Explain causes and effects clearly
- Mention key figures, events, and movements
- Provide context concisely
- Be objective and balanced

Answer the question directly and accurately."""
            
        else:
            system_prompt = """You are a knowledgeable educational assistant.
Provide clear, accurate, and concise answers to questions.
Be direct and factual."""
        
        # Llama 3.2 chat format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def detect_subject(self, question: str) -> str:
        """Fast subject detection using keyword matching"""
        q_lower = question.lower()
        
        # Algebra keywords
        algebra_keywords = [
            'solve', 'equation', 'factor', 'simplify', 'polynomial',
            'matrix', 'matrices', 'function', 'graph', 'linear',
            'quadratic', 'variable', 'calculate', 'algebra', 'derivative',
            'integral', 'coefficient', 'expression', 'formula', 'theorem',
            'x=', 'y=', 'f(x)', 'vertex', 'slope', 'intercept'
        ]
        
        # Geography keywords
        geography_keywords = [
            'country', 'capital', 'city', 'continent', 'ocean',
            'mountain', 'river', 'lake', 'desert', 'geography',
            'located', 'region', 'territory', 'border', 'climate',
            'population', 'area', 'map', 'latitude', 'longitude',
            'peninsula', 'island', 'plateau', 'valley', 'coast',
            'nation', 'state', 'province', 'land', 'world'
        ]
        
        # History keywords
        history_keywords = [
            'history', 'war', 'century', 'ancient', 'medieval',
            'revolution', 'empire', 'dynasty', 'civilization', 'battle',
            'treaty', 'independence', 'colonial', 'king', 'queen',
            'president', 'historical', 'era', 'period', 'age',
            'founded', 'established', 'abolished', 'invasion', 'conquest'
        ]
        
        algebra_score = sum(1 for kw in algebra_keywords if kw in q_lower)
        geography_score = sum(1 for kw in geography_keywords if kw in q_lower)
        history_score = sum(1 for kw in history_keywords if kw in q_lower)
        
        scores = {
            'algebra': algebra_score,
            'geography': geography_score,
            'history': history_score
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    def process_batch(self, questions: List[str], subject: str) -> List[str]:
        """Process a batch of questions with optimized settings"""
        
        # Create subject-specific prompts
        prompts = [self.create_expert_prompt(q, subject) for q in questions]
        
        # Tokenize with left padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Optimized generation settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy = fastest + deterministic
                num_beams=1,  # No beam search = faster
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # KV cache for speed
                temperature=None,  # Disable sampling
                top_p=None,
                repetition_penalty=1.0,
            )
        
        # Decode outputs efficiently
        answers = []
        for i, output in enumerate(outputs):
            # Only decode the generated tokens
            generated_tokens = output[inputs['input_ids'][i].shape[0]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            answer = answer.strip()
            
            # Enforce character limit
            if len(answer) > 5000:
                answer = answer[:5000]
            
            answers.append(answer)
        
        return answers
    
    def process_subject_group(self, questions_with_ids: List[tuple], subject: str) -> List[Dict]:
        """Process all questions of a specific subject"""
        results = []
        questions = [q for q, _ in questions_with_ids]
        question_ids = [qid for _, qid in questions_with_ids]
        
        # Process in batches
        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i:i+self.batch_size]
            batch_ids = question_ids[i:i+self.batch_size]
            
            try:
                batch_answers = self.process_batch(batch_questions, subject)
                
                for qid, answer in zip(batch_ids, batch_answers):
                    results.append({
                        "questionID": qid,
                        "answer": answer
                    })
                    
            except Exception as e:
                # Fallback for failed batches
                for qid in batch_ids:
                    results.append({
                        "questionID": qid,
                        "answer": "Unable to generate answer due to processing error."
                    })
            
            # Memory cleanup every few batches
            if i % 100 == 0 and i > 0:
                gc.collect()
        
        return results
    
    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """
        Main inference function with parallel subject processing.
        """
        
        if self.model is None:
            self.load_model()
        
        # Group questions by subject for parallel processing
        subject_groups = {
            'algebra': [],
            'geography': [],
            'history': [],
            'general': []
        }
        
        for q in questions:
            subject = self.detect_subject(q['question'])
            subject_groups[subject].append((q['question'], q['questionID']))
        
        all_results = []
        
        # Process each subject group in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for subject, questions_with_ids in subject_groups.items():
                if questions_with_ids:  # Only process non-empty groups
                    future = executor.submit(
                        self.process_subject_group,
                        questions_with_ids,
                        subject
                    )
                    futures[future] = subject
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    pass  # Already handled in process_subject_group
        
        # Ensure results are in original order
        result_dict = {r['questionID']: r for r in all_results}
        ordered_results = [result_dict[q['questionID']] for q in questions]
        
        return ordered_results


def loadPipeline():
    """
    Factory function to create and return the inference pipeline.
    Entry point called by run.py
    """
    pipeline = OptimizedInferencePipeline()
    return pipeline