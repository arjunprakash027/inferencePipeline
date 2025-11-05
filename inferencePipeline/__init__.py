"""
Ultra-Optimized Llama.cpp Inference Pipeline for Tech Arena 2025
Uses GGUF Q4_K_M quantization for maximum CPU performance
Caches converted model for reuse
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import gc
from pathlib import Path

# Try to import llama-cpp-python, fallback to transformers
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    from transformers import AutoModelForCausalLM

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
        self.batch_size = 64  # Larger batches with quantization
        self.use_llamacpp = LLAMA_CPP_AVAILABLE
        
        # Subject-specific token limits
        self.token_limits = {
            'algebra': 180,
            'history': 150,
            'geography': 100,
            'general': 120
        }
        
    def convert_to_gguf(self, hf_model_path: str, output_path: str):
        """Convert HF model to GGUF format using llama.cpp converter"""
        import subprocess
        
        print(f"[CONVERT] Converting model to GGUF Q4_K_M format...")
        print(f"[CONVERT] This is a one-time operation, subsequent runs will be fast!")
        
        try:
            # Use llama.cpp's convert script
            cmd = [
                "python", "-m", "llama_cpp.convert",
                hf_model_path,
                "--outfile", output_path,
                "--outtype", "q4_k_m"
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[CONVERT] Conversion complete! GGUF saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"[CONVERT] Conversion failed: {e}")
            print(f"[CONVERT] Falling back to transformers...")
            return False
    
    def load_model(self):
        """Load model - use llama.cpp if available, else transformers"""
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cache_dir = '/app/models'
        
        # Always need tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.use_llamacpp:
            # Try to use llama.cpp for 2-3x speedup
            gguf_path = Path("./gguf_cache/llama-3.2-3b-instruct-q4_k_m.gguf")
            gguf_path.parent.mkdir(exist_ok=True)
            
            # Check if GGUF already exists (cached from previous run)
            if not gguf_path.exists():
                print(f"[LOAD] GGUF not found, converting model...")
                print(f"[LOAD] This will take 3-5 minutes but dramatically speeds up inference!")
                
                #hf_path = Path(cache_dir) / model_name.split('/')[-1]
                hf_path = Path(cache_dir) / "models--meta-llama--Llama-3.2-3B-Instruct"
                if not hf_path.exists():
                    hf_path = cache_dir
                
                success = self.convert_to_gguf(str(hf_path), str(gguf_path))
                
                if not success:
                    print(f"[LOAD] Conversion failed, using transformers instead")
                    self.use_llamacpp = False
            else:
                print(f"[LOAD] Found cached GGUF model, loading...")
            
            if self.use_llamacpp and gguf_path.exists():
                # Load GGUF with llama.cpp - blazing fast!
                self.model = Llama(
                    model_path=str(gguf_path),
                    n_ctx=2048,
                    n_threads=16,  # Use all 16 AMD cores
                    n_batch=512,
                    verbose=False,
                )
                print(f"[LOAD] llama.cpp model loaded successfully!")
                return
        
        # Fallback to transformers (original method)
        print(f"[LOAD] Using transformers (consider installing llama-cpp-python for 2-3x speedup)")
        self.use_llamacpp = False
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except:
            pass
    
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
    
    def process_batch_llamacpp(self, questions: List[str], subject: str) -> List[str]:
        """Process batch using llama.cpp (much faster)"""
        max_tokens = self.token_limits[subject]
        answers = []
        
        # llama.cpp doesn't support batching the same way, process individually
        # But it's so fast that individual processing is still faster than batched transformers!
        for question in questions:
            prompt = self.create_expert_prompt(question, subject)
            
            try:
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,  # Greedy
                    top_p=1.0,
                    echo=False,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                )
                
                answer = output['choices'][0]['text'].strip()
                
                if len(answer) > 5000:
                    answer = answer[:5000]
                
                answers.append(answer)
                
            except Exception as e:
                answers.append("Error generating answer.")
        
        return answers
    
    def process_batch_transformers(self, questions: List[str], subject: str) -> List[str]:
        """Process batch using transformers (fallback)"""
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
        """Main inference function"""
        
        if self.model is None:
            self.load_model()
        
        # Group by subject
        subject_groups = {'algebra': [], 'geography': [], 'history': [], 'general': []}
        
        for q in questions:
            subject = self.detect_subject(q['question'])
            subject_groups[subject].append(q)
        
        all_results = []
        
        # Process each subject sequentially
        for subject, subject_questions in subject_groups.items():
            if not subject_questions:
                continue
            
            questions_text = [q['question'] for q in subject_questions]
            question_ids = [q['questionID'] for q in subject_questions]
            
            # Choose processing method
            if self.use_llamacpp:
                # llama.cpp - process in smaller chunks for memory efficiency
                batch_size = 8  # Smaller batches for llamacpp individual processing
                for i in range(0, len(questions_text), batch_size):
                    batch_qs = questions_text[i:i+batch_size]
                    batch_ids = question_ids[i:i+batch_size]
                    
                    try:
                        batch_answers = self.process_batch_llamacpp(batch_qs, subject)
                        for qid, answer in zip(batch_ids, batch_answers):
                            all_results.append({"questionID": qid, "answer": answer})
                    except Exception:
                        for qid in batch_ids:
                            all_results.append({"questionID": qid, "answer": "Error."})
            else:
                # Transformers - use larger batches
                for i in range(0, len(questions_text), self.batch_size):
                    batch_qs = questions_text[i:i+self.batch_size]
                    batch_ids = question_ids[i:i+self.batch_size]
                    
                    try:
                        batch_answers = self.process_batch_transformers(batch_qs, subject)
                        for qid, answer in zip(batch_ids, batch_answers):
                            all_results.append({"questionID": qid, "answer": answer})
                    except Exception:
                        for qid in batch_ids:
                            all_results.append({"questionID": qid, "answer": "Error."})
                    
                    if i % 192 == 0 and i > 0:
                        gc.collect()
        
        # Return in original order
        result_dict = {r['questionID']: r for r in all_results}
        return [result_dict[q['questionID']] for q in questions]


def loadPipeline():
    """Factory function"""
    return OptimizedInferencePipeline()