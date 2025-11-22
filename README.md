# Tech Arena 2025 - Phase 2 Inference Pipeline

**Efficient LLM inference with AWQ 4-bit quantization**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
from inferencePipeline import loadPipeline
pipeline = loadPipeline()
answers = pipeline(questions)
```

## Features

- ✅ AWQ 4-bit quantization (3x faster)
- ✅ vLLM continuous batching
- ✅ Python calculator for algebra
- ✅ T4-optimized

## Performance

- **Latency**: <60s for 500 questions
- **Accuracy**: >75%
- **Memory**: <5GB VRAM

## See FINAL_SOLUTION.md for details
