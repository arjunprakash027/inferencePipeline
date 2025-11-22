"""
Performance Benchmark Script for vLLM Pipeline
Detailed performance analysis optimized for Tech Arena 2025 scoring

Scoring Formula: 60% Accuracy + 40% Latency
"""

import json
import time
import torch
import psutil
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from vllm_pipeline import WinningVLLMPipeline


# ============================================================================
# GPU MONITORING
# ============================================================================

def get_gpu_memory():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_gpu_info():
    """Get GPU information"""
    if torch.cuda.is_available():
        return {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "cuda_version": torch.version.cuda,
        }
    return {"name": "CPU", "total_memory_gb": 0, "cuda_version": "N/A"}


# ============================================================================
# BENCHMARK DATASETS
# ============================================================================

def generate_test_dataset(size: int = 500) -> List[Dict]:
    """
    Generate realistic test dataset with subject distribution
    Mimics actual competition distribution
    """

    # Subject-specific questions
    algebra_qs = [
        "What is 15 × 24?",
        "Solve for x: 3x - 7 = 14",
        "Factor x² + 5x + 6",
        "What is 20% of 450?",
        "Simplify: 2(x + 3) + 4",
        "What kind of function has a parabolic graph?",
        "Calculate: 12² + 5²",
        "Solve: 2x/3 = 8",
        "What is the slope of y = 2x + 5?",
        "Expand: (x + 2)(x - 3)",
    ]

    geography_qs = [
        "What is the capital of Japan?",
        "Which ocean is the largest?",
        "What is the highest mountain in the world?",
        "Which river is the longest in the world?",
        "What is the largest desert?",
        "Which country has the most people?",
        "What is the smallest continent?",
        "Which sea is the saltiest?",
        "What is the capital of Australia?",
        "Which country is known as the Land of the Rising Sun?",
    ]

    history_qs = [
        "Who was the first president of the United States?",
        "When did World War II end?",
        "What year did the French Revolution begin?",
        "Who was Napoleon Bonaparte?",
        "When did the Roman Empire fall?",
        "Who discovered America?",
        "When did the Cold War end?",
        "Who was Julius Caesar?",
        "When did the Renaissance begin?",
        "Who was Genghis Khan?",
    ]

    chinese_qs = [
        "What does 你好 mean in English?",
        "How do you say 'thank you' in Chinese?",
        "What is the difference between Simplified and Traditional Chinese?",
        "How many tones are in Mandarin Chinese?",
        "What does 谢谢 mean?",
        "What is Pinyin?",
        "How do you say 'goodbye' in Chinese?",
        "What does 中文 mean?",
        "How many Chinese characters exist?",
        "What is the Chinese word for 'water'?",
    ]

    # Competition-like distribution: 25% each subject
    questions = []
    subjects = {
        "algebra": algebra_qs,
        "geography": geography_qs,
        "history": history_qs,
        "chinese": chinese_qs,
    }

    question_id = 0
    per_subject = size // 4

    for subject, qs in subjects.items():
        for i in range(per_subject):
            questions.append({
                "questionID": f"bench_{question_id:04d}",
                "question": qs[i % len(qs)],
                "subject": subject,
            })
            question_id += 1

    # Fill remaining to reach exact size
    while len(questions) < size:
        questions.append({
            "questionID": f"bench_{question_id:04d}",
            "question": geography_qs[question_id % len(geography_qs)],
            "subject": "geography",
        })
        question_id += 1

    return questions[:size]


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_latency(pipeline, num_questions: int = 500) -> Dict:
    """
    Benchmark latency performance
    Simulates competition evaluation
    """
    print("\n" + "=" * 80)
    print(f"⏱️  LATENCY BENCHMARK ({num_questions} questions)")
    print("=" * 80)

    # Generate test data
    questions = generate_test_dataset(num_questions)

    # Subject distribution
    from collections import Counter
    subjects = [q["subject"] for q in questions]
    subject_dist = Counter(subjects)
    print(f"\n📊 Subject Distribution: {dict(subject_dist)}")

    # Warmup (not counted)
    print("\n🔥 Warming up (3 questions)...")
    warmup_qs = questions[:3]
    _ = pipeline(warmup_qs)
    print("✅ Warmup complete")

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Main benchmark
    print(f"\n⚡ Processing {num_questions} questions...")
    print("=" * 80)

    start_time = time.time()
    start_mem = get_gpu_memory()

    results = pipeline(questions)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    end_mem = get_gpu_memory()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    total_time = end_time - start_time

    # Calculate metrics
    throughput = num_questions / total_time
    avg_latency_ms = (total_time / num_questions) * 1000
    mem_usage = peak_mem - start_mem

    # Results
    print("=" * 80)
    print(f"✅ Benchmark Complete!")
    print("=" * 80)
    print(f"\n⏱️  TIMING METRICS:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Throughput: {throughput:.2f} questions/sec")
    print(f"   Avg latency: {avg_latency_ms:.1f}ms per question")

    print(f"\n💾 MEMORY METRICS:")
    print(f"   Start memory: {start_mem:.1f} MB")
    print(f"   End memory: {end_mem:.1f} MB")
    print(f"   Peak memory: {peak_mem:.1f} MB")
    print(f"   Memory used: {mem_usage:.1f} MB")

    # Competition targets
    print(f"\n🎯 COMPETITION TARGETS:")
    TARGET_TIME = 120  # 2 hours max
    COMPETITIVE_TIME = 90  # 1.5 min competitive
    TOP_TIER_TIME = 60  # 1 min top tier

    print(f"   Time limit: {TARGET_TIME}s")
    print(f"   Competitive: {COMPETITIVE_TIME}s")
    print(f"   Top tier: {TOP_TIER_TIME}s")
    print(f"   Your time: {total_time:.2f}s")

    if total_time < TOP_TIER_TIME:
        tier = "🥇 TOP TIER (1st-2nd place potential)"
    elif total_time < COMPETITIVE_TIME:
        tier = "🥈 COMPETITIVE (3rd-5th place potential)"
    elif total_time < TARGET_TIME:
        tier = "✅ ACCEPTABLE (Qualifies)"
    else:
        tier = "⚠️ NEEDS OPTIMIZATION"

    print(f"   Tier: {tier}")

    # Latency score (40% of total)
    # Normalized score: faster = higher score
    latency_score = max(0, min(100, (TARGET_TIME - total_time) / TARGET_TIME * 100))
    print(f"\n📊 LATENCY SCORE: {latency_score:.1f}/100 (40% of total score)")

    return {
        "total_time": total_time,
        "throughput": throughput,
        "avg_latency_ms": avg_latency_ms,
        "peak_memory_mb": peak_mem,
        "memory_used_mb": mem_usage,
        "latency_score": latency_score,
        "tier": tier,
    }


def benchmark_answer_length(pipeline, sample_size: int = 50) -> Dict:
    """
    Analyze answer length distribution
    Competition limit: 5000 chars
    """
    print("\n" + "=" * 80)
    print(f"📏 ANSWER LENGTH ANALYSIS ({sample_size} samples)")
    print("=" * 80)

    questions = generate_test_dataset(sample_size)
    results = pipeline(questions)

    lengths = [len(r["answer"]) for r in results]

    print(f"\n📊 LENGTH STATISTICS:")
    print(f"   Min: {min(lengths)} chars")
    print(f"   Max: {max(lengths)} chars")
    print(f"   Average: {sum(lengths)/len(lengths):.1f} chars")
    print(f"   Median: {sorted(lengths)[len(lengths)//2]} chars")

    # Check violations
    violations = sum(1 for l in lengths if l > 5000)
    print(f"\n⚠️  Violations (>5000 chars): {violations}/{sample_size}")

    if violations > 0:
        print("   🔴 WARNING: Some answers exceed limit!")
    else:
        print("   ✅ All answers within 5000 char limit")

    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "violations": violations,
    }


def benchmark_by_subject(pipeline, questions_per_subject: int = 25) -> Dict:
    """
    Benchmark performance by subject
    Identify strengths and weaknesses
    """
    print("\n" + "=" * 80)
    print(f"🎯 SUBJECT-SPECIFIC PERFORMANCE ({questions_per_subject}Q per subject)")
    print("=" * 80)

    subjects = ["algebra", "geography", "history", "chinese"]
    subject_results = {}

    for subject in subjects:
        # Generate subject-specific questions
        questions = generate_test_dataset(questions_per_subject * 4)
        subject_qs = [q for q in questions if q["subject"] == subject][:questions_per_subject]

        print(f"\n📚 Testing {subject.upper()}...")

        start = time.time()
        results = pipeline(subject_qs)
        elapsed = time.time() - start

        avg_latency = elapsed / len(subject_qs) * 1000
        throughput = len(subject_qs) / elapsed

        # Analyze answer characteristics
        avg_length = sum(len(r["answer"]) for r in results) / len(results)

        print(f"   ⏱️  Time: {elapsed:.2f}s")
        print(f"   📊 Throughput: {throughput:.2f} Q/s")
        print(f"   ⚡ Avg latency: {avg_latency:.1f}ms")
        print(f"   📏 Avg answer length: {avg_length:.0f} chars")

        subject_results[subject] = {
            "time": elapsed,
            "throughput": throughput,
            "avg_latency_ms": avg_latency,
            "avg_length": avg_length,
        }

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUBJECT COMPARISON")
    print("=" * 80)

    for subject, metrics in subject_results.items():
        print(f"\n{subject.upper()}:")
        print(f"  Latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"  Throughput: {metrics['throughput']:.2f} Q/s")

    return subject_results


def estimate_competition_score(latency_time: float, estimated_accuracy: float = 75.0) -> Dict:
    """
    Estimate final competition score
    Formula: 60% Accuracy + 40% Latency
    """
    print("\n" + "=" * 80)
    print("🏆 COMPETITION SCORE ESTIMATION")
    print("=" * 80)

    # Latency score (normalized: faster = higher)
    TARGET_TIME = 120
    latency_score = max(0, min(100, (TARGET_TIME - latency_time) / TARGET_TIME * 100))

    # Accuracy score (estimated - need real evaluation)
    accuracy_score = estimated_accuracy

    # Final score
    final_score = (accuracy_score * 0.6) + (latency_score * 0.4)

    print(f"\n📊 SCORE BREAKDOWN:")
    print(f"   Accuracy (60%): {accuracy_score:.1f}%")
    print(f"   Latency (40%): {latency_score:.1f}%")
    print(f"   FINAL SCORE: {final_score:.1f}%")

    # Competitive analysis
    print(f"\n🎯 COMPETITIVE POSITION:")
    if final_score >= 75:
        print(f"   🥇 EXCELLENT - Top 3 potential")
    elif final_score >= 65:
        print(f"   🥈 GOOD - Top 5 potential")
    elif final_score >= 55:
        print(f"   ✅ ACCEPTABLE - Qualifies for presentation")
    else:
        print(f"   ⚠️ NEEDS IMPROVEMENT")

    print(f"\n💡 IMPROVEMENT OPPORTUNITIES:")
    if accuracy_score < 75:
        print(f"   - Accuracy: +{75-accuracy_score:.1f}% → +{(75-accuracy_score)*0.6:.1f} pts")
    if latency_time > 90:
        potential_gain = ((latency_time - 90) / TARGET_TIME * 100) * 0.4
        print(f"   - Latency: -{latency_time-90:.1f}s → +{potential_gain:.1f} pts")

    return {
        "accuracy_score": accuracy_score,
        "latency_score": latency_score,
        "final_score": final_score,
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_full_benchmark():
    """Run complete benchmark suite"""
    print("\n" + "=" * 80)
    print("🔥 TECH ARENA 2025 - COMPLETE BENCHMARK SUITE")
    print("=" * 80)

    # System info
    gpu_info = get_gpu_info()
    print(f"\n🖥️  SYSTEM INFO:")
    print(f"   GPU: {gpu_info['name']}")
    print(f"   Total VRAM: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"   CUDA: {gpu_info['cuda_version']}")
    print(f"   PyTorch: {torch.__version__}")

    # Load pipeline
    print("\n📦 Loading pipeline...")
    start_load = time.time()
    pipeline = WinningVLLMPipeline(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        cache_dir=os.environ.get("MODEL_CACHE_DIR", "/app/models")
    )
    load_time = time.time() - start_load
    print(f"✅ Pipeline loaded in {load_time:.2f}s")

    # Run benchmarks
    results = {}

    # 1. Main latency benchmark (500Q)
    results["latency"] = benchmark_latency(pipeline, num_questions=500)

    # 2. Answer length analysis
    results["length"] = benchmark_answer_length(pipeline, sample_size=50)

    # 3. Subject-specific performance
    results["by_subject"] = benchmark_by_subject(pipeline, questions_per_subject=25)

    # 4. Score estimation
    results["score"] = estimate_competition_score(
        latency_time=results["latency"]["total_time"],
        estimated_accuracy=75.0  # Conservative estimate
    )

    # Save results
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": gpu_info,
            "load_time": load_time,
            "results": results,
        }, f, indent=2)

    print(f"\n💾 Full results saved to: {output_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("🏁 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\n⏱️  Latency: {results['latency']['total_time']:.2f}s for 500Q")
    print(f"📊 Throughput: {results['latency']['throughput']:.2f} Q/s")
    print(f"💾 Peak memory: {results['latency']['peak_memory_mb']:.1f} MB")
    print(f"🏆 Estimated score: {results['score']['final_score']:.1f}%")
    print(f"🎯 Tier: {results['latency']['tier']}")

    return results


if __name__ == "__main__":
    results = run_full_benchmark()
