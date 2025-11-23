"""
Test the enhanced inference pipeline with focus on Chinese and algebra improvements
"""
import time
import pandas as pd
from inferencePipeline import loadPipeline

def test_pipeline():
    print("Testing enhanced pipeline...")
    
    # Create some test questions to validate improvements
    test_questions = [
        {
            "questionID": "test_1",
            "subject": "algebra",
            "question": "Calculate 15 * 24"
        },
        {
            "questionID": "test_2",
            "subject": "algebra", 
            "question": "If x + 5 = 12, what is x?"
        },
        {
            "questionID": "test_3",
            "subject": "chinese",
            "question": "What is the capital of China?"
        },
        {
            "questionID": "test_4",
            "subject": "chinese",
            "question": "Who was Confucius?"
        },
        {
            "questionID": "test_5",
            "subject": "geography",
            "question": "What is the capital of France?"
        },
        {
            "questionID": "test_6",
            "subject": "history",
            "question": "When did World War II end?"
        }
    ]
    
    print(f"Loading pipeline...")
    pipeline = loadPipeline()
    
    print(f"Running inference on {len(test_questions)} test questions...")
    start_time = time.perf_counter()
    answers = pipeline(test_questions)
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    print(f"\n✅ Completed in {elapsed:.2f} seconds")
    print(f"Average: {elapsed/len(test_questions):.2f} sec/question\n")
    
    # Show results
    print("Test Results:")
    print("-" * 80)
    for i, (q, a) in enumerate(zip(test_questions, answers)):
        print(f"\n{i+1}. [{q['subject']}] {q['question']}")
        print(f"   Answer: {a['answer']}")
    
    # Check for specific improvements
    print("\n" + "="*80)
    print("Analysis:")
    
    algebra_answers = [a for q, a in zip(test_questions, answers) if q['subject'] == 'algebra']
    print(f"Algebra answers: {len(algebra_answers)}")
    for ans in algebra_answers:
        print(f"  - {ans['answer']}")
    
    chinese_answers = [a for q, a in zip(test_questions, answers) if q['subject'] == 'chinese']
    print(f"Chinese answers: {len(chinese_answers)}")
    for ans in chinese_answers:
        print(f"  - {ans['answer']}")
    
    print(f"\nOverall: Pipeline successfully processed {len(answers)} questions")

if __name__ == "__main__":
    test_pipeline()