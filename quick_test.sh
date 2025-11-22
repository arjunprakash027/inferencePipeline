#!/bin/bash
# Quick Test Script for vLLM Pipeline - Tech Arena 2025
# Run this on your T4 GPU to verify everything works

set -e  # Exit on error

echo "========================================================================"
echo "🚀 TECH ARENA 2025 - QUICK TEST SCRIPT"
echo "========================================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Warning: Not in a virtual environment${NC}"
    echo "   Recommended: source venv/bin/activate"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: System Check
echo ""
echo "========================================================================"
echo "Step 1: System Diagnostics"
echo "========================================================================"
python check_system.py --skip-inference

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ System check failed${NC}"
    echo "   Please fix issues above before continuing"
    exit 1
fi

# Step 2: Ask about model download
echo ""
echo "========================================================================"
echo "Step 2: Model Download"
echo "========================================================================"

MODEL_DIR=${MODEL_CACHE_DIR:-"/app/models"}
echo "Model cache directory: $MODEL_DIR"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "✅ Model cache directory exists and is not empty"
    read -p "Skip model download? [Y/n]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "⏭️  Skipping model download"
    else
        python download_vllm_model.py
    fi
else
    echo "⚠️  Model cache is empty"
    read -p "Download Qwen2.5-3B-Instruct now? [Y/n]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        python download_vllm_model.py
    else
        echo "⏭️  Skipping model download (may fail later)"
    fi
fi

# Step 3: Run Tests
echo ""
echo "========================================================================"
echo "Step 3: Test Suite"
echo "========================================================================"
read -p "Run comprehensive tests? [Y/n]: " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    python test_vllm_pipeline.py

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
    else
        echo -e "${RED}❌ Some tests failed${NC}"
        echo "   Check output above for details"
        read -p "Continue to benchmark anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "⏭️  Skipping tests"
fi

# Step 4: Run Benchmark
echo ""
echo "========================================================================"
echo "Step 4: Performance Benchmark"
echo "========================================================================"
read -p "Run full benchmark (500 questions)? [Y/n]: " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "⚡ Running benchmark (this takes 1-2 minutes on T4)..."
    python benchmark_vllm.py

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Benchmark complete!${NC}"
        echo "   Results saved to benchmark_results.json"
    else
        echo -e "${RED}❌ Benchmark failed${NC}"
        exit 1
    fi
else
    echo "⏭️  Skipping benchmark"
fi

# Summary
echo ""
echo "========================================================================"
echo "🏁 QUICK TEST COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review test_results.json for test details"
echo "  2. Review benchmark_results.json for performance metrics"
echo "  3. Check SETUP_GUIDE.md for optimization tips"
echo "  4. Run 'python vllm_pipeline.py' for custom tests"
echo ""
echo -e "${GREEN}✅ Your T4 GPU is ready for competition!${NC}"
echo ""
