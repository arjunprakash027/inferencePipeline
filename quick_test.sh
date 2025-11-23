#!/bin/bash

# Quick Test Script for Pipeline Iterations
# Usage: ./quick_test.sh [iteration_name]

set -e  # Exit on error

ITERATION_NAME=${1:-"test_run"}

echo "=========================================="
echo "Pipeline Quick Test: $ITERATION_NAME"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "load.py" ]; then
    echo "Error: Please run this script from the jaadu directory"
    exit 1
fi

# Run the pipeline
echo "[1/3] Running inference pipeline..."
python3 load.py --name "$ITERATION_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Pipeline failed!"
    exit 1
fi

echo ""
echo "[2/3] Evaluating results..."
python3 evaluate_answers.py

echo ""
echo "[3/3] Showing latest experiment results..."
echo ""
tail -n 1 experiments.csv | column -t -s,

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""
echo "Files generated:"
echo "  - answers_output.json (answers)"
echo "  - experiments.csv (metrics)"
echo ""
echo "To view detailed answers:"
echo "  cat answers_output.json | jq '.[] | {subject, question, answer}' | less"
echo ""
echo "To view experiments history:"
echo "  column -t -s, experiments.csv | less"
