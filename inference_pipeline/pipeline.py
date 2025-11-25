"""
Inference Pipeline
"""

from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .server import LlamaServer
from .prompts import create_expert_prompt
from .subjects import detect_subject

class InferencePipeline:
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

    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """Main inference function - Concurrent HTTP requests"""
        results = []

        # Prepare all prompts
        tasks = []
        for q in questions:
            subject = detect_subject(q['question'])
            prompt = create_expert_prompt(q['question'], subject)
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

def load_pipeline():
    """Factory function"""
    print("[FACTORY] Initializing Client-Server Pipeline...")
    return InferencePipeline()
