"""
Server-Based Inference Pipeline
Uses llama-server for high-throughput concurrent inference.
"""

import os
import time
import atexit
import asyncio
import aiohttp
import platform
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Dict

# Load settings at module import time (before any inference)
from . import get_settings
SETTINGS = get_settings()

class LlamaServer:
    """Manages the llama-server process"""
    
    def __init__(self, model_path: str, host: str = None, port: int = None, n_gpu_layers: int = None):
        # Use settings if not provided
        server_cfg = SETTINGS['server']
        self.host = host or server_cfg['host']
        self.port = port or server_cfg['port']
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else server_cfg['n_gpu_layers']
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None
        
        # Detect OS and find server binary
        current_os = platform.system().lower()
        if "darwin" in current_os:
            bin_dir = "bin_mac"
        elif "linux" in current_os:
            bin_dir = "bin_linux"
        else:
            raise RuntimeError(f"Unsupported OS: {current_os}")
        
        self.server_bin = Path(__file__).parent / "llama.cpp" / bin_dir / "llama-server"
        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found at {self.server_bin}")
            
        # Start server
        self._start_server(model_path)
        atexit.register(self.stop)

    def _start_server(self, model_path: str):
        """Start the llama-server process"""
        server_cfg = SETTINGS['server']
        
        cmd = [
            str(self.server_bin),
            "-m", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(server_cfg['context_size']),
            "-ngl", str(self.n_gpu_layers),
            "--parallel", str(server_cfg['parallel_requests']),
        ]
        
        if server_cfg.get('continuous_batching', True):
            cmd.append("-cb")
        
        print(f"[SERVER] Starting llama-server on port {self.port}...")
        print(f"[SERVER] Command: {' '.join(cmd)}")
        
        # Redirect output to file for debugging
        self.log_file = open("server_debug.log", "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=self.log_file,
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
        gen_cfg = SETTINGS['inference']['generation']
        
        data = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": gen_cfg['temperature'],
            "top_p": gen_cfg['top_p'],
            "top_k": gen_cfg['top_k'],
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


import re
import requests

# KB Paths
CHINESE_KB_PATH = os.path.join(os.path.dirname(__file__), "chinese_kb.txt")
ALGEBRA_KB_PATH = os.path.join(os.path.dirname(__file__), "algebra_kb.txt")

class ServerInferencePipeline:
    def __init__(self):
        """Initialize pipeline and start server"""
        model_cfg = SETTINGS['model']
        server_cfg = SETTINGS['server']
        inference_cfg = SETTINGS['inference']
        
        self.server = None
        # Use model from settings
        gguf_cache = Path(model_cfg['gguf_cache_dir'])
        self.model_path = gguf_cache / server_cfg['model_file']
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please run the normal pipeline once to convert it.")

        # Load knowledge bases for CAG
        self.chinese_kb = self._load_chinese_knowledge_base()
        self.algebra_kb = self._load_algebra_knowledge_base()

        # Start server using settings
        self.server = LlamaServer(str(self.model_path))
        
        # Token limits from settings
        self.token_limits = inference_cfg['token_limits']

        # Warmup the cache with Knowledge Bases
        self._warmup_cache()

    def _load_chinese_knowledge_base(self) -> str:
        """Load Chinese knowledge base for cache-augmented generation"""
        try:
            kb_path = Path(CHINESE_KB_PATH)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📚 Loaded Chinese knowledge base ({len(content)} chars)")
                return content
            else:
                print("⚠️  Chinese knowledge base not found")
                return ""
        except Exception as e:
            print(f"⚠️  Error loading Chinese KB: {e}")
            return ""

    def _load_algebra_knowledge_base(self) -> str:
        """Load Algebra knowledge base for formula/theorem reference"""
        try:
            kb_path = Path(ALGEBRA_KB_PATH)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📐 Loaded Algebra knowledge base ({len(content)} chars)")
                return content
            else:
                print("⚠️  Algebra knowledge base not found")
                return ""
        except Exception as e:
            print(f"⚠️  Error loading Algebra KB: {e}")
            return ""

    def _warmup_cache(self):
        """
        Warmup the KV cache by processing the Knowledge Bases once.
        This ensures subsequent requests using these KBs hit the cache.
        """
        print("🔥 Warming up cache with Knowledge Bases...")
        
        warmup_prompts = []
        
        # Chinese KB Warmup
        if self.chinese_kb:
            prompt = self.create_expert_prompt("预热", "chinese")
            warmup_prompts.append(prompt)
            
        # Algebra KB Warmup
        if self.algebra_kb:
            prompt = self.create_expert_prompt("Warmup", "algebra")
            warmup_prompts.append(prompt)
            
        if warmup_prompts:
            # Run synchronous requests to populate cache
            for prompt in warmup_prompts:
                try:
                    requests.post(
                        f"{self.server.base_url}/completion",
                        json={"prompt": prompt, "n_predict": 1, "temperature": 0.0}
                    )
                except Exception as e:
                    print(f"⚠️ Warmup failed: {e}")
            print("✅ Cache warmed up!")
        else:
            print("⚠️ No KBs to warm up.")

    def create_expert_prompt(self, question: str, subject: str) -> str:
        """Enhanced prompts with CAG and few-shot examples"""
        
        if subject == "chinese":
            system = f"""你是中国文化和语言专家。使用以下参考资料回答问题。

参考资料：
{self.chinese_kb}

基于以上参考资料，直接回答以下问题："""
            user_content = f"问题: {question}\n答案:"

        elif subject == "algebra":
            system = f"""You are a math expert. Use the formulas and theorems below to solve the problem.

REFERENCE FORMULAS:
{self.algebra_kb}

Using the above formulas, solve step-by-step and provide the final answer."""
            
            examples = """
Example 1:
Problem: Solve for x: 2x + 5 = 13
Solution: Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4.
Final Answer: x = 4

Example 2:
Problem: Simplify (x+3)(x-3)
Solution: Using difference of squares formula (a+b)(a-b) = a² - b². Here a=x, b=3.
Final Answer: x² - 9

Example 3:
Problem: If f(x) = 3x² - 2x + 1, find f(2)
Solution: Substitute x=2 into the function: f(2) = 3(2)² - 2(2) + 1 = 3(4) - 4 + 1 = 12 - 4 + 1 = 9.
Final Answer: 9

Example 4:
Problem: What is the derivative of x² + 3x?
Solution: Using power rule: d/dx(x²) = 2x, d/dx(3x) = 3. Sum: 2x + 3.
Final Answer: 2x + 3

Example 5:
Problem: Solve the system: x + y = 5, x - y = 1
Solution: Add equations: (x+y) + (x-y) = 5+1, so 2x = 6, x = 3. Substitute into first equation: 3 + y = 5, y = 2.
Final Answer: x = 3, y = 2

Example 6:
Problem: Find the area of a circle with radius 5
Solution: Use formula A = πr². A = π(5)² = 25π ≈ 78.54.
Final Answer: 25π or approximately 78.54

Example 7:
Problem: Factor x² + 5x + 6
Solution: Find two numbers that multiply to 6 and add to 5: 2 and 3. So (x+2)(x+3).
Final Answer: (x+2)(x+3)

Example 8:
Problem: What is 15% of 80?
Solution: 15% = 0.15. Multiply: 0.15 × 80 = 12.
Final Answer: 12"""
            
            user_content = f"{examples}\n\nNow solve this problem step-by-step:\n\nProblem: {question}\nSolution:"

        else:
            # History/Geography/Finance - Direct answer format
            system = "Answer this question directly with just the factual answer."
            examples = """
Question: Who discovered America?
Answer: Christopher Columbus in 1492.

Question: What is the highest point in South America?
Answer: Mount Aconcagua in Argentina (6,961 meters).

Question: Which country is known as the 'Land of the Thunder Dragon'?
Answer: Bhutan.

Question: Who won World War 2?
Answer: The Allied Powers (United States, Soviet Union, United Kingdom, and France)."""
            
            user_content = f"{examples}\n\nQuestion: {question}\nAnswer:"
        
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def detect_subject(self, question: str) -> str:
        """Fast subject detection"""
        q = question.lower()
        if any(kw in q for kw in ['solve', 'equation', 'factor', 'matrix']): return 'algebra'
        if any(kw in q for kw in ['country', 'capital', 'mountain', 'river']): return 'geography'
        if any(kw in q for kw in ['war', 'revolution', 'when did', 'who was']): return 'history'
        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', question): return 'chinese'
        return 'general'

    def _strip_thinking(self, text: str) -> str:
        """Remove <think> tags and their content"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?think>', '', text)
        return text.strip()

    def _extract_final_answer(self, text: str, subject: str) -> str:
        """Extract only the final answer from reasoning output"""
        if subject == 'algebra':
            # Strip reasoning prefixes
            reasoning_prefixes = [
                r'^(?:Okay|So|Let me|First|Now|Here|The user|To solve|We need).*?\n',
                r'^(?:I need to|I will|Let\'s|Step \d+).*?\n',
            ]
            for prefix in reasoning_prefixes:
                text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE)

            answer_patterns = [
                r'Answer:\s*(.+?)(?:\n\n|\n(?=Problem)|$)',
                r'(?:Final [Aa]nswer|ANSWER):\s*(.+?)(?:\n|$)',
                r'(?:Therefore|Thus|So),?\s*(.+?)(?:\n|$)',
                r'=\s*([^=\n]+)$',
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    answer = answer.rstrip('.').strip()
                    return answer.split('\n')[0]

        elif subject == 'chinese':
            answer_patterns = [
                r'(?:答案|最终答案)[:：]\s*(.+?)(?:\n\n|\n|$)',
                r'(?:Answer|answer):\s*(.+?)(?:\n\n|\n|$)',
                r'(?:结论|因此)[:：,，]\s*(.+?)(?:\n\n|\n|$)',
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    return re.sub(r'\n+', ' ', answer)

        return text

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
            max_tokens = self.token_limits.get(subject, 120)
            tasks.append((q['questionID'], prompt, max_tokens, subject))
        
        # Execute concurrently with aiohttp
        max_workers = SETTINGS['server']['max_workers']
        async with aiohttp.ClientSession() as session:
            async_tasks = [
                self._process_single(session, qid, prompt, max_tokens, subject)
                for qid, prompt, max_tokens, subject in tasks
            ]
            results = await asyncio.gather(*async_tasks)
        
        return list(results)
    
    async def _process_single(self, session: aiohttp.ClientSession, qid: str, prompt: str, max_tokens: int, subject: str) -> Dict:
        """Process a single question asynchronously"""
        try:
            raw_answer = await self.server.completion(session, prompt, max_tokens)
            # Clean and extract answer
            cleaned_answer = self._strip_thinking(raw_answer)
            final_answer = self._extract_final_answer(cleaned_answer, subject)
            
            # Length check
            if len(final_answer) > 5000:
                final_answer = final_answer[:5000].rsplit('. ', 1)[0] + '.'
                
        except Exception as e:
            final_answer = f"Error: {e}"
            
        return {"questionID": qid, "answer": final_answer}

def loadServerPipeline():
    """Factory function"""
    print("[FACTORY] Initializing Client-Server Pipeline...")
    return ServerInferencePipeline()
