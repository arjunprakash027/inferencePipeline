import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get token from environment
token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# Configure base cache directory
base_cache_dir = "./app/model"

# Define all models to download
models = [
    # {
    #     "repo_id": "google/embeddinggemma-300m",
    #     "name": "embeddinggemma-300m"
    # },
    # {
    #     "repo_id": "google/gemma-3-270m-it",
    #     "name": "gemma-3-270m-it"
    # },
    # {
    #     "repo_id": "google/gemma-3-1b-it",
    #     "name": "gemma-3-1b-it"
    # },
    # {
    #     "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
    #     "name": "Llama-3.2-1B-Instruct"
    # },
    # {
    #     "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
    #     "name": "Llama-3.2-3B-Instruct"
    # },
    # {
    #     "repo_id": "Qwen/Qwen3-Embedding-0.6B",
    #     "name": "Qwen3-Embedding-0.6B"
    # },
    # {
    #     "repo_id": "Qwen/Qwen3-0.6B",
    #     "name": "Qwen3-0.6B"
    # },
    # {
    #     "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
    #     "name": "Llama-3.1-8B-Instruct"
    # },
    {
        "repo_id": "Qwen/Qwen3-1.7B",
        "name": "Qwen3-1.7B"
    }
]

# Download each model
print(f"Starting download of {len(models)} models...")
print(f"Token available: {'Yes' if token else 'No'}")

for i, model in enumerate(models, 1):
    print(f"\n[{i}/{len(models)}] Downloading {model['name']}...")
    print(f"Repository: {model['repo_id']}")
    
    try:
        # Create model-specific cache directory
        cache_dir = os.path.join(base_cache_dir, model['name'])
        
        # Download the model
        local_cache_path = snapshot_download(
            repo_id=model['repo_id'],
            cache_dir=base_cache_dir,
            token=token,               
            resume_download=True       
        )
        
        print(f"Successfully cached at: {local_cache_path}")
    
    except Exception as e:
        print(f"Error downloading {model['name']}: {str(e)}")
        continue

print("Download complete!")