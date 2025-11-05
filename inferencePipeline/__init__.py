"""
Ultra-Optimized Llama.cpp Inference Pipeline for Tech Arena 2025
v2
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import gc
import sys
import subprocess
from pathlib import Path
import multiprocessing


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
        """Initialize and LOAD MODEL immediately (not timed)"""
        self.device = "cpu"
        self.batch_size = 64
        self.use_llamacpp = LLAMA_CPP_AVAILABLE
        
        # Optimal thread configuration for 16-core AMD CPU
        self.n_cores = multiprocessing.cpu_count()
        self.n_physical_cores = self.n_cores // 2 if self.n_cores > 8 else self.n_cores
        self.n_threads = max(4, int(self.n_physical_cores * 0.6))
        self.n_threads_batch = max(2, self.n_physical_cores - self.n_threads)
        
        print(f"[INIT] Detected {self.n_cores} logical cores, {self.n_physical_cores} physical")
        print(f"[INIT] Using n_threads={self.n_threads}, n_threads_batch={self.n_threads_batch}")
        
        # Subject-specific token limits
        self.token_limits = {
            'algebra': 180,
            'history': 150,
            'geography': 100,
            'general': 120
        }
        
        # LOAD MODEL IMMEDIATELY IN __INIT__
        print(f"[INIT] Loading model (this happens before timing starts)...")
        self.model, self.tokenizer = self._load_model()
        print(f"[INIT] ✓ Model loaded and ready for inference!")
    
    def find_snapshot_path(self, hf_cache_dir: str) -> Path:
        """Find the actual model snapshot path in HuggingFace cache"""
        cache_path = Path(hf_cache_dir)
        snapshots_dir = cache_path / "snapshots"
        
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                snapshot = sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                print(f"[SNAPSHOT] Found: {snapshot}")
                return snapshot
        
        print(f"[SNAPSHOT] Using direct path: {cache_path}")
        return cache_path
    
    def convert_to_gguf(self, hf_model_path: str, output_path: str) -> bool:
        """
        Simple one-step conversion to Q8_0 (no llama-quantize binary needed)
        Q8_0 is directly supported by convert_hf_to_gguf.py
        """
        print(f"[CONVERT] Converting to Q8_0 GGUF (one-step, 3-5 min)...")
        
        try:
            possible_scripts = [
                Path(__file__).parent / "llama.cpp" / "convert_hf_to_gguf.py",
                Path(__file__).parent / "llama.cpp" / "convert.py",
                Path("/app/llama.cpp/convert_hf_to_gguf.py"),
                Path("/app/llama.cpp/convert.py"),
            ]
            
            convert_script = None
            for script in possible_scripts:
                if script.exists():
                    convert_script = script
                    print(f"[CONVERT] Found script: {convert_script}")
                    break
            
            if convert_script is None:
                print(f"[CONVERT] ERROR: Conversion script not found!")
                return False
            
            model_path = self.find_snapshot_path(hf_model_path)
            
            # Direct Q8_0 conversion (no external tools needed)
            cmd = [
                sys.executable,
                str(convert_script),
                str(model_path),
                "--outfile", str(output_path),
                "--outtype", "q8_0"  # Directly supported!
            ]
            
            print(f"[CONVERT] Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[CONVERT] ✓ Conversion complete!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[CONVERT] ✗ Failed: {e}")
            print(f"[CONVERT] stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"[CONVERT] ✗ Failed: {e}")
            return False
    
    def _load_model(self):
        """
        PRIVATE method called ONLY from __init__
        Returns (model, tokenizer) tuple
        """
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cache_dir = '/app/models'
        
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
        
        if self.use_llamacpp:
            # Use llama.cpp for maximum speed
            gguf_path = Path("./gguf_cache/model_q8.gguf")
            gguf_path.parent.mkdir(exist_ok=True)
            
            # Check if GGUF exists, convert if not
            if not gguf_path.exists():
                print(f"[LOAD] GGUF not cached, converting (one-time 3-5 min)...")
                
                hf_path = Path(cache_dir) / "models--meta-llama--Llama-3.2-3B-Instruct"
                if not hf_path.exists():
                    hf_path = Path(cache_dir)
                
                success = self.convert_to_gguf(str(hf_path), str(gguf_path))
                
                if not success:
                    print(f"[LOAD] Conversion failed, falling back to transformers")
                    self.use_llamacpp = False
            else:
                print(f"[LOAD] ✓ Found cached GGUF!")
            
            if self.use_llamacpp and gguf_path.exists():
                # Load GGUF with optimal settings
                print(f"[LOAD] Loading GGUF with llama-cpp-python...")
                model = Llama(
                    model_path=str(gguf_path),
                    n_ctx=2048,
                    n_threads=self.n_threads,
                    n_threads_batch=self.n_threads_batch,
                    n_batch=512,
                    n_ubatch=256,
                    n_gpu_layers=0,
                    use_mmap=True,
                    use_mlock=False,
                    rope_freq_base=0.0,
                    rope_freq_scale=0.0,
                    flash_attn=True,
                    verbose=False,
                    logits_all=False,
                    embedding=False,
                    offload_kqv=False,
                )
                print(f"[LOAD] ✓ GGUF model loaded")
                
                # WARMUP - crucial for avoiding first-query slowness
                print(f"[LOAD] Warming up model (initializing caches)...")
                model("Test warmup", max_tokens=5, temperature=0.0, echo=False)
                print(f"[LOAD] ✓ Warmup complete")
                
                return model, tokenizer
        
        # Fallback to transformers
        print(f"[LOAD] Using transformers (slower, consider llama-cpp-python)")
        self.use_llamacpp = False
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        try:
            print(f"[LOAD] Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass
        
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
    
    def process_batch_llamacpp(self, questions: List[str], subject: str) -> List[str]:
        """Process batch using llama.cpp - PURE INFERENCE (fastest)"""
        max_tokens = self.token_limits[subject]
        answers = []
        
        for question in questions:
            prompt = self.create_expert_prompt(question, subject)
            
            try:
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    echo=False,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    repeat_penalty=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs=None,
                    stream=False,
                )
                
                answer = output['choices'][0]['text'].strip()
                
                if len(answer) > 5000:
                    answer = answer[:5000]
                
                answers.append(answer)
                
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                answers.append("Error generating answer.")
        
        return answers
    
    def process_batch_transformers(self, questions: List[str], subject: str) -> List[str]:
        """Process batch using transformers - PURE INFERENCE"""
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
        """
        Main inference function - ONLY GENERATION (timed section)
        Model is already loaded in __init__
        """
        batch_size = 16
        # Group by subject for better batching
        subject_groups = {'algebra': [], 'geography': [], 'history': [], 'general': []}
        
        for q in questions:
            subject = self.detect_subject(q['question'])
            subject_groups[subject].append(q)
        
        all_results = []
        
        # Process each subject group
        for subject, subject_questions in subject_groups.items():
            if not subject_questions:
                continue
            
            questions_text = [q['question'] for q in subject_questions]
            question_ids = [q['questionID'] for q in subject_questions]
            
            if self.use_llamacpp:
                # llama.cpp - optimized batches
                
                for i in range(0, len(questions_text), batch_size):
                    batch_qs = questions_text[i:i+batch_size]
                    batch_ids = question_ids[i:i+batch_size]
                    
                    try:
                        batch_answers = self.process_batch_llamacpp(batch_qs, subject)
                        for qid, answer in zip(batch_ids, batch_answers):
                            all_results.append({"questionID": qid, "answer": answer})
                    except Exception as e:
                        print(f"[ERROR] Batch failed: {e}")
                        for qid in batch_ids:
                            all_results.append({"questionID": qid, "answer": "Error."})
            else:
                # Transformers - larger batches
                for i in range(0, len(questions_text), self.batch_size):
                    batch_qs = questions_text[i:i+self.batch_size]
                    batch_ids = question_ids[i:i+batch_size]
                    
                    try:
                        batch_answers = self.process_batch_transformers(batch_qs, subject)
                        for qid, answer in zip(batch_ids, batch_answers):
                            all_results.append({"questionID": qid, "answer": answer})
                    except Exception as e:
                        print(f"[ERROR] Batch failed: {e}")
                        for qid in batch_ids:
                            all_results.append({"questionID": qid, "answer": "Error."})
                    
                    if i % 192 == 0 and i > 0:
                        gc.collect()
        
        # Return in original order
        result_dict = {r['questionID']: r for r in all_results}
        return [result_dict[q['questionID']] for q in questions]


def loadPipeline():
    """
    Factory function - model loads HERE (before timing)
    Returns ready-to-use pipeline
    """
    print("[FACTORY] Creating and initializing pipeline...")
    pipeline = OptimizedInferencePipeline()  # Model loads in __init__
    print("[FACTORY] ✓ Pipeline ready for timed inference!")
    return pipeline
