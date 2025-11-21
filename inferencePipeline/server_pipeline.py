"""
Server-Based Inference Pipeline
Uses llama-server for high-throughput concurrent inference.
"""

import os
import time
import atexit
import requests
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

class LlamaServer:
    """Manages the llama-server process"""
    
    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 8081, n_gpu_layers: int = 0):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process = None
        
        # Find server binary
        self.server_bin = Path(__file__).parent / "llama.cpp" / "bin_mac" / "llama-server"
        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found at {self.server_bin}")
            
        # Start server
        self._start_server(model_path, n_gpu_layers)
        atexit.register(self.stop)

    def _start_server(self, model_path: str, n_gpu_layers: int):
        """Start the llama-server process"""
        cmd = [
            str(self.server_bin),
            "-m", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", "2048",           # Context size
            "-ngl", str(n_gpu_layers), # GPU layers
            "--parallel", "8",      # Max concurrent requests
            "-cb",                  # Continuous batching
            "--log-disable"         # Reduce log noise
        ]
        
        print(f"[SERVER] Starting llama-server on port {self.port}...")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for health check
        print("[SERVER] Waiting for server to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print("[SERVER] ✓ Server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
            
        self.stop()
        raise RuntimeError("Server failed to start in 30 seconds")

    def stop(self):
        """Stop the server process"""
        if self.process:
            print("[SERVER] Shutting down...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def completion(self, prompt: str, max_tokens: int = 100) -> str:
        """Send completion request"""
        data = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "stop": ["<|eot_id|>", "<|end_of_text|>"]
        }
        
        try:
            response = requests.post(f"{self.base_url}/completion", json=data)
            if response.status_code == 200:
                return response.json()['content'].strip()
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"


class ServerInferencePipeline:
    def __init__(self):
        """Initialize pipeline and start server"""
        self.server = None
        # Use q8_0 model for higher accuracy
        self.model_path = Path("./gguf_cache/model_q8.gguf")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please run the normal pipeline once to convert it.")

        # Start server (n_gpu_layers=0 for CPU)
        self.server = LlamaServer(str(self.model_path), n_gpu_layers=0)
        
        # Subject-specific token limits
        self.token_limits = {
            'algebra': 180,
            'history': 150,
            'geography': 100,
            'general': 120
        }

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
            system = "You are an expert algebra tutor."
        elif subject == "geography":
            few_shot = """Examples:
Q: What is the capital of France?
A: Paris.
Q: Which river is the longest in the world?
A: The Nile River (though the Amazon is contended).
Q: In which continent is the Sahara Desert located?
A: Africa.
Now answer:"""
            system = "You are an expert geographer."
        elif subject == "history":
            few_shot = """Examples:
Q: Who was the first Emperor of Rome?
A: Augustus Caesar (Octavian).
Q: When did the Titanic sink?
A: April 15, 1912.
Q: What was the main cause of World War I?
A: The assassination of Archduke Franz Ferdinand, along with alliances, imperialism, and nationalism.
Now answer:"""
            system = "You are an expert historian."
        else:
            few_shot = "Answer the question directly."
            system = "You are a helpful assistant."
        
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{few_shot}\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def detect_subject(self, question: str) -> str:
        """Fast subject detection"""
        q = question.lower()
        if any(kw in q for kw in ['solve', 'equation', 'factor', 'matrix']): return 'algebra'
        if any(kw in q for kw in ['country', 'capital', 'mountain', 'river']): return 'geography'
        if any(kw in q for kw in ['war', 'revolution', 'when did', 'who was']): return 'history'
        return 'general'

    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """Main inference function - Concurrent HTTP requests"""
        results = []
        
        # Prepare all prompts
        tasks = []
        for q in questions:
            subject = self.detect_subject(q['question'])
            prompt = self.create_expert_prompt(q['question'], subject)
            max_tokens = self.token_limits[subject]
            tasks.append((q['questionID'], prompt, max_tokens))
        
        # Execute concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_qid = {
                executor.submit(self.server.completion, prompt, max_tokens): qid 
                for qid, prompt, max_tokens in tasks
            }
            
            for future in as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    answer = future.result()
                except Exception as e:
                    answer = f"Error: {e}"
                results.append({"questionID": qid, "answer": answer})
        
        return results

def loadServerPipeline():
    """Factory function"""
    print("[FACTORY] Initializing Client-Server Pipeline...")
    return ServerInferencePipeline()
