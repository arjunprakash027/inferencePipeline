import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv; load_dotenv()
os.getenv("HF_TOKEN")

# Configure
repo_id = "meta-llama/Llama-3.2-1B-Instruct"   # repo ID
cache_dir = "./app/model"                        # pick a writable absolute path
token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# Download the full snapshot into the cache (populate for offline use)
local_cache_path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    token=token,               # required for gated/private repos
    resume_download=True       # optional: resume if interrupted
)

print(f"Cached at: {local_cache_path}")

