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
import platform


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
    def __init__(self, use_speculative_decoding=True):
        """Initialize and LOAD MODEL immediately (not timed)"""
        self.device = "cpu"
        self.batch_size = 64
        self.use_llamacpp = LLAMA_CPP_AVAILABLE
        
        # Speculative decoding configuration
        self.use_speculative_decoding = use_speculative_decoding and LLAMA_CPP_AVAILABLE
        self.draft_model = None
        self.spec_draft_count = 4  # Reduced from 8 for better CPU efficiency
        
        # Optimal thread configuration for 16-core AMD CPU
        self.n_cores = multiprocessing.cpu_count()
        self.n_physical_cores = self.n_cores // 2 if self.n_cores > 8 else self.n_cores
        self.n_threads = max(4, int(self.n_physical_cores * 0.6))
        self.n_threads_batch = max(2, self.n_physical_cores - self.n_threads)
        
        print(f"[INIT] Detected {self.n_cores} logical cores, {self.n_physical_cores} physical")
        print(f"[INIT] Using n_threads={self.n_threads}, n_threads_batch={self.n_threads_batch}")
        print(f"[INIT] Speculative decoding: {'ENABLED' if self.use_speculative_decoding else 'DISABLED'}")
        
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
    
    def convert_to_gguf(self, hf_model_path: str, output_path: str, target_q="q4_1") -> bool:
        """
        Convert HuggingFace model -> f16 GGUF -> quantize to q4_1 (or q4_0, q8_0, etc.)
        """
        try:
            model_path = self.find_snapshot_path(hf_model_path)
            tmp_f16 = Path(output_path).with_suffix(".f16.gguf")

            # Step 1: Convert to f16
            print(f"[CONVERT] Step 1: Converting HF -> f16 GGUF")
            possible_scripts = [
                Path(__file__).parent / "llama.cpp" / "convert_hf_to_gguf.py",
                Path(__file__).parent / "llama.cpp" / "convert.py",
            ]
            convert_script = next((p for p in possible_scripts if p.exists()), None)
            if convert_script is None:
                raise FileNotFoundError("convert_hf_to_gguf.py not found in llama.cpp")

            cmd_f16 = [
                sys.executable,
                str(convert_script),
                str(model_path),
                "--outfile", str(tmp_f16),
                "--outtype", "f16"
            ]
            subprocess.run(cmd_f16, check=True)
            print(f"[CONVERT] ✓ f16 GGUF created: {tmp_f16}")

            # Step 2: OS detection for quantizer binary
            current_os = platform.system().lower()
            if "darwin" in current_os:
                bin_dir = "bin_mac"
            elif "linux" in current_os:
                bin_dir = "bin_linux"
            else:
                raise RuntimeError(f"Unsupported OS detected: {current_os}")

            quant_bin = Path(__file__).parent / "llama.cpp" / bin_dir / "llama-quantize"
            if not quant_bin.exists():
                raise FileNotFoundError(f"Quantizer not found at {quant_bin}")

            if not os.access(quant_bin, os.X_OK):
                os.chmod(quant_bin, 0o755)

            cmd_quant = [
                str(quant_bin),
                str(tmp_f16),
                str(output_path),
                target_q
            ]
            print(f"[CONVERT] Step 2: Running quantizer ({current_os}) → {target_q}")
            subprocess.run(cmd_quant, check=True)
            print(f"[CONVERT] ✓ Quantization to {target_q} complete")

            return True

        except subprocess.CalledProcessError as e:
            print(f"[CONVERT] ✗ Subprocess failed: {e}")
            print(f"stderr:\n{getattr(e, 'stderr', '')}")
            return False
        except Exception as e:
            print(f"[CONVERT] ✗ Unexpected error: {e}")
            return False
    
    def _load_model(self):
        """
        PRIVATE method called ONLY from __init__
        Returns (model, tokenizer) tuple
        """
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        cache_dir = './app/model'
        
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
            draft_gguf_path = Path("./gguf_cache/model_q4_0_draft.gguf")
            gguf_path.parent.mkdir(exist_ok=True)
            
            # Check if target GGUF exists, convert if not
            if not gguf_path.exists():
                print(f"[LOAD] Target GGUF not cached, converting (one-time 3-5 min)...")
                
                hf_path = Path(cache_dir) / "models--meta-llama--Llama-3.2-1B-Instruct"
                if not hf_path.exists():
                    hf_path = Path(cache_dir)
                
                success = self.convert_to_gguf(str(hf_path), str(gguf_path), target_q="q8_0")
                
                if not success:
                    print(f"[LOAD] Conversion failed, falling back to transformers")
                    self.use_llamacpp = False
                    self.use_speculative_decoding = False
            else:
                print(f"[LOAD] ✓ Found cached target GGUF!")
            
            # Check if draft GGUF exists for speculative decoding
            if self.use_speculative_decoding and not draft_gguf_path.exists():
                print(f"[LOAD] Draft GGUF not cached, converting to q4_0 (1-2 min)...")
                
                hf_path = Path(cache_dir) / "models--meta-llama--Llama-3.2-1B-Instruct"
                if not hf_path.exists():
                    hf_path = Path(cache_dir)
                
                success = self.convert_to_gguf(str(hf_path), str(draft_gguf_path), target_q="q4_0")
                
                if not success:
                    print(f"[LOAD] Draft conversion failed, disabling speculative decoding")
                    self.use_speculative_decoding = False
            elif self.use_speculative_decoding:
                print(f"[LOAD] ✓ Found cached draft GGUF!")
            
            if self.use_llamacpp and gguf_path.exists():
                # Simple approach: just use q4_0 for speed (no speculation overhead)
                if self.use_speculative_decoding and draft_gguf_path.exists():
                    print(f"[LOAD] Loading q4_0 model (fast mode)...")
                    model = Llama(
                        model_path=str(draft_gguf_path),  # Use q4_0
                        n_ctx=2048,
                        n_threads=self.n_threads,
                        n_batch=512,
                        n_gpu_layers=0,
                        use_mmap=True,
                        use_mlock=False,
                        verbose=False,
                    )
                    print(f"[LOAD] ✓ q4_0 model loaded (2× faster than q8_0)")
                else:
                    # Load q8_0 if speculative decoding disabled
                    print(f"[LOAD] Loading q8_0 model...")
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
                    print(f"[LOAD] ✓ q8_0 model loaded")
                
                # Warmup
                print(f"[LOAD] Warming up model...")
                model("Test", max_tokens=5, temperature=0.0, echo=False)
                print(f"[LOAD] ✓ Warmup complete")
                
                if self.use_speculative_decoding:
                    print(f"[LOAD] ✓ FAST MODE: Using q4_0 (no speculation overhead)")
                
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

Q: Solve for x: 3x - 7 = 14
A: Add 7 to both sides: 3x = 21. Divide by 3: x = 7.

Q: What is the quadratic formula?
A: x = (-b ± √(b² - 4ac)) / 2a.

Q: Expand (x + 2)(x - 2)
A: This is a difference of squares: x² - 4.

Now answer:"""
            system = "You are an expert algebra tutor. Show clear mathematical reasoning."
            
        elif subject == "geography":
            few_shot = """Examples:

Q: What is the capital of France?
A: Paris.

Q: Which river is the longest in the world?
A: The Nile River (though the Amazon is contended).

Q: In which continent is the Sahara Desert located?
A: Africa.

Now answer:"""
            system = "You are an expert geographer. Provide precise facts and locations."
            
        elif subject == "history":
            few_shot = """Examples:

Q: Who was the first Emperor of Rome?
A: Augustus Caesar (Octavian).

Q: When did the Titanic sink?
A: April 15, 1912.

Q: What was the main cause of World War I?
A: The assassination of Archduke Franz Ferdinand, along with alliances, imperialism, and nationalism.

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
        """Process batch using llama.cpp - optimized for speed"""
        max_tokens = self.token_limits[subject]
        answers = []
        
        # Model is already optimized (q4_0 when spec decoding enabled)
        gen_kwargs = {
            'max_tokens': max_tokens,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 1,
            'echo': False,
            'stop': ["<|eot_id|>", "<|end_of_text|>"],
            'repeat_penalty': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'logprobs': None,
            'stream': False,
        }
        
        for question in questions:
            prompt = self.create_expert_prompt(question, subject)
            try:
                output = self.model(prompt, **gen_kwargs)
                answer = output['choices'][0]['text'].strip()
                if len(answer) > 5000:
                    answer = answer[:5000]
                answers.append(answer)
            except Exception:
                # Fail silently; avoids I/O bottleneck
                answers.append("Error.")
        return answers


    def process_batch_transformers(self, questions: List[str], subject: str) -> List[str]:
        """Process batch using transformers - PURE INFERENCE (silent)"""
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
            
            if self.use_llamacpp:
                for i in range(0, len(questions_text), batch_size):
                    batch_qs = questions_text[i:i+batch_size]
                    batch_ids = question_ids[i:i+batch_size]
                    try:
                        batch_answers = self.process_batch_llamacpp(batch_qs, subject)
                        for qid, answer in zip(batch_ids, batch_answers):
                            all_results.append({"questionID": qid, "answer": answer})
                    except Exception:
                        # Silent batch fail
                        for qid in batch_ids:
                            all_results.append({"questionID": qid, "answer": "Error."})
            else:
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



def loadPipeline():
    """
    Factory function - model loads HERE (before timing)
    Returns ready-to-use pipeline
    """
    print("[FACTORY] Creating and initializing pipeline...")
    pipeline = OptimizedInferencePipeline()  # Model loads in __init__
    print("[FACTORY] ✓ Pipeline ready for timed inference!")
    return pipeline
