"""
Manages the llama-server process
"""

import time
import atexit
import requests
import subprocess
from pathlib import Path

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
