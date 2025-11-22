"""
Comprehensive Test Suite for vLLM Pipeline
Tests functionality, accuracy, and performance on T4 GPU
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from vllm_pipeline import loadPipeline


# ============================================================================
# TEST DATA
# ============================================================================

TEST_QUESTIONS = [
    # Algebra
    {"questionID": "alg_1", "question": "What is 15 multiplied by 24?", "subject": "algebra"},
    {"questionID": "alg_2", "question": "Solve for x: 2x + 5 = 15", "subject": "algebra"},
    {"questionID": "alg_3", "question": "Factor x² + 7x + 12", "subject": "algebra"},
    {"questionID": "alg_4", "question": "What kind of function has a graph that is a straight line?", "subject": "algebra"},
    {"questionID": "alg_5", "question": "Calculate 25% of 840", "subject": "algebra"},

    # Geography
    {"questionID": "geo_1", "question": "What is the capital of Japan?", "subject": "geography"},
    {"questionID": "geo_2", "question": "Which ocean is the largest?", "subject": "geography"},
    {"questionID": "geo_3", "question": "What is the highest mountain in the world?", "subject": "geography"},
    {"questionID": "geo_4", "question": "Which country is known as the Land of the Rising Sun?", "subject": "geography"},
    {"questionID": "geo_5", "question": "What is the longest river in the world?", "subject": "geography"},

    # History
    {"questionID": "hist_1", "question": "Who was the first president of the United States?", "subject": "history"},
    {"questionID": "hist_2", "question": "When did World War II end?", "subject": "history"},
    {"questionID": "hist_3", "question": "What year did the French Revolution begin?", "subject": "history"},
    {"questionID": "hist_4", "question": "Who was Napoleon Bonaparte?", "subject": "history"},
    {"questionID": "hist_5", "question": "When did the Roman Empire fall?", "subject": "history"},

    # Chinese
    {"questionID": "chin_1", "question": "What does 你好 (nǐ hǎo) mean in English?", "subject": "chinese"},
    {"questionID": "chin_2", "question": "How many characters are in modern Chinese?", "subject": "chinese"},
    {"questionID": "chin_3", "question": "What is the difference between Simplified and Traditional Chinese?", "subject": "chinese"},
    {"questionID": "chin_4", "question": "What does 谢谢 mean?", "subject": "chinese"},
]


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_basic_functionality():
    """Test 1: Basic functionality - can the pipeline load and process questions?"""
    print("\n" + "=" * 80)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("=" * 80)

    try:
        # Load pipeline
        print("\n[1.1] Loading pipeline...")
        pipeline = loadPipeline()
        print("✅ Pipeline loaded successfully")

        # Test with single question
        print("\n[1.2] Testing single question...")
        single_q = [{"questionID": "test_single", "question": "What is 2 + 2?"}]
        result = pipeline(single_q)

        assert len(result) == 1, "Should return 1 result"
        assert "questionID" in result[0], "Result should have questionID"
        assert "answer" in result[0], "Result should have answer"
        print(f"✅ Single question test passed")
        print(f"   Answer: {result[0]['answer'][:100]}...")

        # Test with empty list
        print("\n[1.3] Testing empty input...")
        empty_result = pipeline([])
        assert len(empty_result) == 0, "Empty input should return empty result"
        print("✅ Empty input test passed")

        return True

    except Exception as e:
        print(f"❌ BASIC FUNCTIONALITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_subject_detection():
    """Test 2: Subject detection accuracy"""
    print("\n" + "=" * 80)
    print("TEST 2: SUBJECT DETECTION")
    print("=" * 80)

    try:
        pipeline = loadPipeline()

        test_cases = [
            ("What is 2x + 5?", "algebra"),
            ("Solve the equation x² = 16", "algebra"),
            ("What is the capital of France?", "geography"),
            ("Which ocean is largest?", "geography"),
            ("When did WWI start?", "history"),
            ("Who was Napoleon?", "history"),
            ("What does 你好 mean?", "chinese"),
        ]

        correct = 0
        for question, expected in test_cases:
            detected = pipeline._detect_subject(question)
            match = "✅" if detected == expected else "❌"
            print(f"{match} '{question[:40]}...' -> {detected} (expected: {expected})")
            if detected == expected:
                correct += 1

        accuracy = correct / len(test_cases) * 100
        print(f"\n📊 Subject Detection Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

        return accuracy >= 70  # 70% threshold

    except Exception as e:
        print(f"❌ SUBJECT DETECTION TEST FAILED: {e}")
        return False


def test_batch_processing():
    """Test 3: Batch processing with diverse questions"""
    print("\n" + "=" * 80)
    print("TEST 3: BATCH PROCESSING")
    print("=" * 80)

    try:
        pipeline = loadPipeline()

        print(f"\n[3.1] Processing {len(TEST_QUESTIONS)} questions in batch...")
        start = time.time()
        results = pipeline(TEST_QUESTIONS)
        elapsed = time.time() - start

        print(f"\n✅ Batch processing completed in {elapsed:.2f}s")
        print(f"📊 Throughput: {len(TEST_QUESTIONS)/elapsed:.2f} questions/sec")
        print(f"📊 Average latency: {elapsed/len(TEST_QUESTIONS)*1000:.1f}ms per question")

        # Verify results
        assert len(results) == len(TEST_QUESTIONS), "Should return all results"

        # Check all have required fields
        for r in results:
            assert "questionID" in r, "Missing questionID"
            assert "answer" in r, "Missing answer"
            assert len(r["answer"]) > 0, "Answer should not be empty"
            assert len(r["answer"]) <= 5000, f"Answer exceeds 5000 chars: {len(r['answer'])}"

        print(f"✅ All {len(results)} results valid")

        # Show sample results
        print("\n[3.2] Sample results:")
        for r in results[:3]:
            q = next(q for q in TEST_QUESTIONS if q["questionID"] == r["questionID"])
            print(f"\n   Q ({q['subject']}): {q['question']}")
            print(f"   A: {r['answer'][:150]}...")

        return True

    except Exception as e:
        print(f"❌ BATCH PROCESSING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_answer_quality():
    """Test 4: Answer quality with known questions"""
    print("\n" + "=" * 80)
    print("TEST 4: ANSWER QUALITY")
    print("=" * 80)

    try:
        pipeline = loadPipeline()

        quality_tests = [
            {
                "questionID": "qa_1",
                "question": "What is the capital of France?",
                "expected_keywords": ["Paris"],
            },
            {
                "questionID": "qa_2",
                "question": "Who was the first president of the United States?",
                "expected_keywords": ["Washington", "George"],
            },
            {
                "questionID": "qa_3",
                "question": "What is 12 × 12?",
                "expected_keywords": ["144"],
            },
            {
                "questionID": "qa_4",
                "question": "What is the largest ocean?",
                "expected_keywords": ["Pacific"],
            },
        ]

        results = pipeline(quality_tests)

        passed = 0
        for r in results:
            test = next(t for t in quality_tests if t["questionID"] == r["questionID"])
            answer_lower = r["answer"].lower()

            # Check if any expected keyword is in answer
            found = any(kw.lower() in answer_lower for kw in test["expected_keywords"])

            status = "✅" if found else "❌"
            print(f"\n{status} Q: {test['question']}")
            print(f"   Expected: {test['expected_keywords']}")
            print(f"   Got: {r['answer'][:150]}...")

            if found:
                passed += 1

        accuracy = passed / len(quality_tests) * 100
        print(f"\n📊 Answer Quality Score: {accuracy:.1f}% ({passed}/{len(quality_tests)})")

        return accuracy >= 60  # 60% threshold

    except Exception as e:
        print(f"❌ ANSWER QUALITY TEST FAILED: {e}")
        return False


def test_performance_target():
    """Test 5: Performance targets for competition"""
    print("\n" + "=" * 80)
    print("TEST 5: COMPETITION PERFORMANCE TARGETS")
    print("=" * 80)

    try:
        pipeline = loadPipeline()

        # Simulate 500 questions (use repeats for testing)
        print(f"\n[5.1] Simulating 500-question evaluation...")
        simulated_500 = []
        for i in range(500):
            q = TEST_QUESTIONS[i % len(TEST_QUESTIONS)].copy()
            q["questionID"] = f"sim_{i}"
            simulated_500.append(q)

        print(f"   Processing {len(simulated_500)} questions...")
        start = time.time()
        results = pipeline(simulated_500)
        elapsed = time.time() - start

        print(f"\n✅ Completed in {elapsed:.2f}s")
        print(f"📊 Throughput: {len(results)/elapsed:.2f} questions/sec")
        print(f"📊 Average latency: {elapsed/len(results)*1000:.1f}ms per question")

        # Competition targets
        TARGET_TIME = 120  # 2 minutes max
        COMPETITIVE_TIME = 90  # 1.5 minutes ideal

        print(f"\n🎯 COMPETITION BENCHMARKS:")
        print(f"   Time limit: {TARGET_TIME}s")
        print(f"   Competitive target: {COMPETITIVE_TIME}s")
        print(f"   Your time: {elapsed:.2f}s")

        if elapsed < COMPETITIVE_TIME:
            print(f"   🏆 EXCELLENT - Top 3 potential!")
        elif elapsed < TARGET_TIME:
            print(f"   ✅ GOOD - Within time limit")
        else:
            print(f"   ⚠️ SLOW - Needs optimization")

        return elapsed < TARGET_TIME

    except Exception as e:
        print(f"❌ PERFORMANCE TEST FAILED: {e}")
        return False


def test_error_handling():
    """Test 6: Error handling and robustness"""
    print("\n" + "=" * 80)
    print("TEST 6: ERROR HANDLING")
    print("=" * 80)

    try:
        pipeline = loadPipeline()

        # Test with malformed inputs
        edge_cases = [
            {"questionID": "edge_1", "question": ""},  # Empty question
            {"questionID": "edge_2", "question": "a" * 10000},  # Very long question
            {"questionID": "edge_3", "question": "§¶•∞§¶•∞§¶•"},  # Special characters
            {"questionID": "edge_4", "question": "What is " * 100},  # Repetitive
        ]

        print(f"\n[6.1] Testing edge cases...")
        results = pipeline(edge_cases)

        assert len(results) == len(edge_cases), "Should handle all edge cases"

        for r in results:
            assert "questionID" in r, "Should have questionID"
            assert "answer" in r, "Should have answer"
            print(f"✅ {r['questionID']}: Handled gracefully")

        print(f"\n✅ Error handling test passed")
        return True

    except Exception as e:
        print(f"❌ ERROR HANDLING TEST FAILED: {e}")
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 80)
    print("🧪 VLLM PIPELINE TEST SUITE")
    print("=" * 80)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Subject Detection", test_subject_detection),
        ("Batch Processing", test_batch_processing),
        ("Answer Quality", test_answer_quality),
        ("Performance Targets", test_performance_target),
        ("Error Handling", test_error_handling),
    ]

    results = {}
    passed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Final summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{len(tests)} tests passed ({passed/len(tests)*100:.1f}%)")

    if passed == len(tests):
        print("\n🏆 ALL TESTS PASSED - PIPELINE READY FOR COMPETITION!")
    elif passed >= len(tests) * 0.8:
        print("\n✅ MOST TESTS PASSED - Pipeline functional with minor issues")
    else:
        print("\n⚠️ SIGNIFICANT ISSUES - Review failed tests before submission")

    return results


if __name__ == "__main__":
    results = run_all_tests()

    # Save results to file
    output_file = Path(__file__).parent / "test_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": {k: "PASS" if v else "FAIL" for k, v in results.items()},
            "pass_rate": f"{sum(results.values())}/{len(results)}"
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
