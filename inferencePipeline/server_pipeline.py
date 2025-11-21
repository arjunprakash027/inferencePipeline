"""
Server-Based Inference Pipeline
Uses llama-server for high-throughput concurrent inference.
"""

import os
import time
import atexit
import asyncio
import aiohttp
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Dict

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
            "-ngl", str(n_gpu_layers), # GPU layers (0 for CPU)
            "--parallel", "8",      # Max concurrent requests
            "-cb",                  # Continuous batching
        ]
        
        print(f"[SERVER] Starting llama-server on port {self.port}...")
        
        # Show output for debugging
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        import requests
        
        # Wait for health check (synchronous to avoid event loop issues during init)
        print("[SERVER] Waiting for server to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print("[SERVER] ✓ Server is ready!")
                    return
            except requests.exceptions.ConnectionError as e:
                if i == 0:  # First attempt
                    print(f"[SERVER] Waiting... (ConnectionError)")
            time.sleep(1)
        
        # Server failed - show error output
        print("[SERVER] ✗ Health check failed. Server output:")
        if self.process.poll() is not None:  # Process ended
            stdout, stderr = self.process.communicate()
            if stdout:
                print(f"STDOUT:\n{stdout[:2000]}")
            if stderr:
                print(f"STDERR:\n{stderr[:2000]}")
        else:
            print("[SERVER] Process is still running but not responsive")
        
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

    async def completion(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int = 100) -> str:
        """Send completion request (async)"""
        data = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "stop": ["<|eot_id|>", "<|end_of_text|>"]
        }
        
        try:
            async with session.post(f"{self.base_url}/completion", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['content'].strip()
                else:
                    return f"Error: {response.status}"
        except Exception as e:
            return f"Error: {e}"


class ServerInferencePipeline:
    def __init__(self):
        """Initialize pipeline and start server"""
        self.server = None
        # Use the q4_0 draft model as the main model for speed
        self.model_path = Path("./gguf_cache/model_q4_0_draft.gguf")
        
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
Q: What kind of function has a graph that is a straight line?
A: A linear function: y = mx + b.
Q: Factor x² + 7x + 12
A: (x + 3)(x + 4).
Now answer:"""
            system = "You are an expert algebra tutor."
        elif subject == "geography":
            few_shot = """Examples:
Q: What's the highest point in South America?
A: Aconcagua.
Q: What is the largest hot desert?
A: The Sahara.
Now answer:"""
            system = "You are an expert geographer."
        elif subject == "history":
            few_shot = """Examples:
Q: When did World War II end?
A: 1945.
Q: Who was the first US President?
A: George Washington.
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
        """Main inference function - Concurrent async HTTP requests"""
        return asyncio.run(self._async_call(questions))
    
    async def _async_call(self, questions: List[Dict]) -> List[Dict]:
        """Async implementation of inference"""
        # Prepare all prompts
        tasks = []
        for q in questions:
            subject = self.detect_subject(q['question'])
            prompt = self.create_expert_prompt(q['question'], subject)
            max_tokens = self.token_limits[subject]
            tasks.append((q['questionID'], prompt, max_tokens))
        
        # Execute concurrently with aiohttp
        async with aiohttp.ClientSession() as session:
            async_tasks = [
                self._process_single(session, qid, prompt, max_tokens)
                for qid, prompt, max_tokens in tasks
            ]
            results = await asyncio.gather(*async_tasks)
        
        return list(results)
    
    async def _process_single(self, session: aiohttp.ClientSession, qid: str, prompt: str, max_tokens: int) -> Dict:
        """Process a single question asynchronously"""
        try:
            answer = await self.server.completion(session, prompt, max_tokens)
        except Exception as e:
            answer = f"Error: {e}"
        return {"questionID": qid, "answer": answer}

def loadServerPipeline():
    """Factory function"""
    print("[FACTORY] Initializing Client-Server Pipeline...")
    return ServerInferencePipeline()
