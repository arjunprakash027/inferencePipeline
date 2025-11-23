#!/usr/bin/env python3
"""
Benchmark script to test inference with and without speculative decoding
Tests accuracy and latency on sample questions
"""

import os
import sys
import time
import json
from pathlib import Path

# Test questions covering all subjects
TEST_QUESTIONS = [
    # Algebra
    {"questionID": "test_algebra_1", "subject": "algebra", "question": "Solve for x: 3x + 7 = 22"},
    {"questionID": "test_algebra_2", "subject": "algebra", "question": "What is the derivative of x² + 3x?"},
    {"questionID": "test_algebra_3", "subject": "algebra", "question": "Factor: x² + 5x + 6"},

    # Chinese
    {"questionID": "test_chinese_1", "subject": "chinese", "question": "中国的首都是哪里？"},
    {"questionID": "test_chinese_2", "subject": "chinese", "question": "谁是唐朝著名的诗人？"},
    {"questionID": "test_chinese_3", "subject": "chinese", "question": "春节是农历的哪一天？"},

    # Geography
    {"questionID": "test_geography_1", "subject": "geography", "question": "What is the capital of France?"},
    {"questionID": "test_geography_2", "subject": "geography", "question": "Which is the longest river in the world?"},

    # History
    {"questionID": "test_history_1", "subject": "history", "question": "When did World War 2 end?"},
    {"questionID": "test_history_2", "subject": "history", "question": "Who was the first president of the United States?"},
]

def run_benchmark(enable_speculative: bool):
    """Run benchmark with or without speculative decoding"""

    # Set environment variable
    os.environ["ENABLE_SPECULATIVE_DECODING"] = "true" if enable_speculative else "false"

    print("\n" + "="*80)
    print(f"🔬 BENCHMARK: Speculative Decoding {'ENABLED' if enable_speculative else 'DISABLED'}")
    print("="*80 + "\n")

    # Import pipeline (will use the env var we just set)
    sys.path.insert(0, str(Path(__file__).parent / "inferencePipeline"))

    # Clear any cached imports
    if 'pipeline' in sys.modules:
        del sys.modules['pipeline']

    from inferencePipeline.pipeline import loadPipeline

    print("📥 Loading pipeline...")
    load_start = time.perf_counter()
    pipeline = loadPipeline()
    load_time = time.perf_counter() - load_start
    print(f"✅ Pipeline loaded in {load_time:.2f} seconds\n")

    print(f"🧪 Running {len(TEST_QUESTIONS)} test questions...")
    inference_start = time.perf_counter()
    results = pipeline(TEST_QUESTIONS)
    inference_time = time.perf_counter() - inference_start

    avg_latency = inference_time / len(TEST_QUESTIONS)

    print(f"\n📊 RESULTS:")
    print(f"   Total inference time: {inference_time:.2f} seconds")
    print(f"   Average latency per question: {avg_latency:.3f} seconds")
    print(f"   Throughput: {len(TEST_QUESTIONS)/inference_time:.2f} questions/second")

    print(f"\n📝 Sample Answers:")
    for i, result in enumerate(results[:5]):  # Show first 5
        question = TEST_QUESTIONS[i]
        print(f"\n   Q{i+1} [{question['subject']}]: {question['question'][:60]}...")
        print(f"   A: {result['answer'][:100]}...")

    return {
        "speculative_enabled": enable_speculative,
        "load_time": load_time,
        "inference_time": inference_time,
        "avg_latency": avg_latency,
        "throughput": len(TEST_QUESTIONS)/inference_time,
        "results": results
    }

def main():
    """Run benchmarks for both configurations"""

    print("\n" + "🚀"*40)
    print("SPECULATIVE DECODING BENCHMARK TEST")
    print("🚀"*40)

    # Test WITHOUT speculative decoding
    print("\n\n### TEST 1: WITHOUT Speculative Decoding ###")
    results_without = run_benchmark(enable_speculative=False)

    # Clear GPU memory
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)  # Wait for cleanup

    # Test WITH speculative decoding
    print("\n\n### TEST 2: WITH Speculative Decoding ###")
    results_with = run_benchmark(enable_speculative=True)

    # Comparison
    print("\n\n" + "="*80)
    print("📊 COMPARATIVE ANALYSIS")
    print("="*80)

    print(f"\n⏱️  LATENCY COMPARISON:")
    print(f"   Without Speculative: {results_without['avg_latency']:.3f} sec/question")
    print(f"   With Speculative:    {results_with['avg_latency']:.3f} sec/question")

    speedup = results_without['avg_latency'] / results_with['avg_latency']
    print(f"\n   🚀 SPEEDUP: {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")

    improvement = ((results_without['avg_latency'] - results_with['avg_latency']) / results_without['avg_latency']) * 100
    print(f"   📈 Improvement: {improvement:.1f}%")

    print(f"\n🎯 THROUGHPUT COMPARISON:")
    print(f"   Without Speculative: {results_without['throughput']:.2f} q/s")
    print(f"   With Speculative:    {results_with['throughput']:.2f} q/s")

    print(f"\n⏲️  TOTAL INFERENCE TIME:")
    print(f"   Without Speculative: {results_without['inference_time']:.2f} seconds")
    print(f"   With Speculative:    {results_with['inference_time']:.2f} seconds")
    print(f"   Time saved: {results_without['inference_time'] - results_with['inference_time']:.2f} seconds")

    # Accuracy comparison (basic check)
    print(f"\n✅ ACCURACY CHECK:")
    matches = 0
    for i in range(len(results_without['results'])):
        if results_without['results'][i]['answer'] == results_with['results'][i]['answer']:
            matches += 1

    accuracy_match = (matches / len(results_without['results'])) * 100
    print(f"   Exact answer match: {matches}/{len(results_without['results'])} ({accuracy_match:.1f}%)")
    print(f"   Note: Different answers don't necessarily mean wrong - LLM sampling can vary")

    # Save results to JSON
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "without_speculative": {
                "load_time": results_without['load_time'],
                "inference_time": results_without['inference_time'],
                "avg_latency": results_without['avg_latency'],
                "throughput": results_without['throughput'],
            },
            "with_speculative": {
                "load_time": results_with['load_time'],
                "inference_time": results_with['inference_time'],
                "avg_latency": results_with['avg_latency'],
                "throughput": results_with['throughput'],
            },
            "speedup": speedup,
            "improvement_percent": improvement,
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
