"""
Highly Optimized Llama 3.2 3B Inference Pipeline for Tech Arena 2025
Optimizations: Dynamic tokens, few-shot learning, sequential batching
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import gc

# Disable unnecessary features
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class OptimizedInferencePipeline:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.batch_size = 48  # Larger for 3B with 128GB RAM
        
        # Subject-specific token limits (your insight!)
        self.token_limits = {
            'algebra': 180,      # Math needs detailed steps
            'history': 150,      # History needs context
            'geography': 100,    # Geography is usually concise
            'general': 120       # Default
        }
        
    def load_model(self):
        """Load Llama 3.2 3B with aggressive optimizations"""
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cache_dir = './app/model'
        #cache_dir = '/app/models'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        # Left padding is critical
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,  # bfloat16 better than float32 on modern CPUs
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,  # Enable KV cache
        )
        
        self.model.eval()
        
        # Compile for CPU - significant speedup
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        except:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except:
                pass
    
    def create_expert_prompt(self, question: str, subject: str) -> str:
        """
        Enhanced prompts with few-shot examples for better accuracy.
        Each subject gets optimized examples and instructions.
        """
        
        if subject == "algebra":
            few_shot = """Examples of high-quality algebra answers:

Q: What kind of function has a graph that is a straight line?
A: A linear function has the form y = mx + b, where m is the slope and b is the y-intercept.

Q: Could you explain why matrices help solve { x + y = 5 ; x - y = 1 }?
A: Matrices organize the system as A·v = b where A = [[1,1], [1,-1]], v = [x, y]ᵀ, and b = [5, 1]ᵀ. Using inverse matrices or row reduction: multiply by A⁻¹ to get v = A⁻¹b, yielding x = 3, y = 2.

Q: Factor x² + 7x + 12
A: Find two numbers that multiply to 12 and add to 7: those are 3 and 4. Therefore: x² + 7x + 12 = (x + 3)(x + 4).

Now answer this question with the same level of detail:"""
            
            system = "You are an expert algebra tutor. Show clear mathematical reasoning and include key steps."
            
        elif subject == "geography":
            few_shot = """Examples of high-quality geography answers:

Q: What's the highest point in South America?
A: Aconcagua (6,961 m or 22,837 ft) in Argentina, located in the Andes mountain range.

Q: Which country is known as the 'Land of the Thunder Dragon'?
A: Bhutan, a landlocked country in the Eastern Himalayas.

Q: What are the Llanos?
A: The Llanos are vast tropical grasslands and savannas primarily in Venezuela and eastern Colombia, covering the Orinoco River basin.

Q: What is the largest desert in the world?
A: Antarctica is the largest desert (cold desert) at ~14 million km². The largest hot desert is the Sahara in North Africa at ~9 million km².

Now answer this question with specific facts and details:"""
            
            system = "You are an expert geographer. Provide precise facts, numbers, and locations."
            
        elif subject == "history":
            few_shot = """Examples of high-quality history answers:

Q: When did World War II end?
A: World War II ended in 1945. Nazi Germany surrendered on May 7-8, 1945 (V-E Day), and Imperial Japan surrendered on August 15, 1945 (V-J Day), with the formal surrender signed on September 2, 1945.

Q: In what ways did the Industrial Revolution influence political ideologies in Europe?
A: The Industrial Revolution (1760-1840) profoundly shaped European political thought. It fueled classical liberalism advocating free markets and individual rights. It also sparked socialism and Marxism as workers faced exploitation and inequality, leading to labor movements. Conservatives resisted rapid social change, while social democracy emerged as a middle path. The revolution fundamentally reshaped class structures and political discourse.

Q: Who was the first President of the United States?
A: George Washington served as the first President from 1789 to 1797, establishing many presidential precedents.

Now answer this question with historical context and key details:"""
            
            system = "You are an expert historian. Provide accurate dates, causes, effects, and context."
            
        else:
            few_shot = """Example:

Q: What is photosynthesis?
A: Photosynthesis is the process by which plants, algae, and some bacteria convert light energy (usually from the sun) into chemical energy stored in glucose. Using chlorophyll, they combine CO₂ and H₂O to produce glucose (C₆H₁₂O₆) and oxygen (O₂).

Now answer this question clearly:"""
            
            system = "You are a knowledgeable educational assistant. Provide clear, accurate answers."
        
        # Llama 3.2 optimized format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{few_shot}

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def detect_subject(self, question: str) -> str:
        """Optimized subject detection with weighted keywords"""
        q = question.lower()
        
        # Fast first-pass checks for obvious cases
        if any(kw in q for kw in ['solve', 'equation', 'factor', 'matrix', 'matrices', 'simplify']):
            return 'algebra'
        if any(kw in q for kw in ['country', 'capital', 'mountain', 'river', 'ocean', 'continent']):
            return 'geography'
        if any(kw in q for kw in ['war', 'revolution', 'century', 'when did', 'who was', 'industrial revolution']):
            return 'history'
        
        # Detailed scoring for ambiguous cases
        algebra_score = sum(1 for kw in ['function', 'graph', 'linear', 'quadratic', 
                                         'calculate', 'x=', 'y=', 'polynomial', 'coefficient',
                                         'variable', 'algebra', 'derivative'] if kw in q)
        
        geography_score = sum(1 for kw in ['located', 'region', 'land', 'world',
                                           'territory', 'geography', 'climate', 'area',
                                           'island', 'desert', 'known as'] if kw in q)
        
        history_score = sum(1 for kw in ['historical', 'ancient', 'empire', 'period',
                                         'influence', 'founded', 'established', 'era',
                                         'dynasty', 'civilization'] if kw in q)
        
        scores = {'algebra': algebra_score, 'geography': geography_score, 'history': history_score}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    def process_batch(self, questions: List[str], subject: str) -> List[str]:
        """Process batch with subject-specific token limits"""
        
        # Get subject-specific token limit
        max_tokens = self.token_limits[subject]
        
        # Create prompts with few-shot examples
        prompts = [self.create_expert_prompt(q, subject) for q in questions]
        
        # Tokenize efficiently
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Increased for few-shot examples
        ).to(self.device)
        
        # Generate with optimized settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # Dynamic per subject!
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.0,
            )
        
        # Fast decoding
        answers = []
        for i, output in enumerate(outputs):
            generated_tokens = output[inputs['input_ids'][i].shape[0]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Enforce limit
            if len(answer) > 5000:
                answer = answer[:5000]
            
            answers.append(answer)
        
        return answers
    
    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """
        Main inference - SEQUENTIAL processing for better CPU efficiency with 3B model.
        Parallel processing adds overhead on CPU with larger models.
        """
        
        if self.model is None:
            self.load_model()
        
        # Group by subject for optimized batching
        subject_groups = {'algebra': [], 'geography': [], 'history': [], 'general': []}
        
        for q in questions:
            subject = self.detect_subject(q['question'])
            subject_groups[subject].append(q)
        
        all_results = []
        
        # Process each subject sequentially (faster than parallel for 3B on CPU)
        for subject, subject_questions in subject_groups.items():
            if not subject_questions:
                continue
            
            # Extract questions and IDs
            questions_text = [q['question'] for q in subject_questions]
            question_ids = [q['questionID'] for q in subject_questions]
            
            # Process in batches
            for i in range(0, len(questions_text), self.batch_size):
                batch_qs = questions_text[i:i+self.batch_size]
                batch_ids = question_ids[i:i+self.batch_size]
                
                try:
                    batch_answers = self.process_batch(batch_qs, subject)
                    
                    for qid, answer in zip(batch_ids, batch_answers):
                        all_results.append({
                            "questionID": qid,
                            "answer": answer
                        })
                        
                except Exception as e:
                    for qid in batch_ids:
                        all_results.append({
                            "questionID": qid,
                            "answer": "Error generating answer."
                        })
                
                # Memory cleanup
                if i % 144 == 0 and i > 0:  # Every 3 full batches
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
        
        # Return in original order
        result_dict = {r['questionID']: r for r in all_results}
        return [result_dict[q['questionID']] for q in questions]


def loadPipeline():
    """Factory function for run.py"""
    return OptimizedInferencePipeline()