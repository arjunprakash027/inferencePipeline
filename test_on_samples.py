"""
Test the inference pipeline on sample_questions.xlsx
"""
import time
import pandas as pd
from inferencePipeline import loadPipeline

print("Loading sample questions...")
df = pd.read_excel("sample_questions.xlsx")
questions = df.to_dict('records')

print(f"Loaded {len(questions)} questions")
print(f"Subjects: {df['subject'].value_counts().to_dict()}\n")

print("Loading inference pipeline...")
pipeline = loadPipeline()

print("\nRunning inference...")
start_time = time.perf_counter()
answers = pipeline(questions)
end_time = time.perf_counter()

elapsed = end_time - start_time
print(f"\n✅ Completed in {elapsed:.2f} seconds")
print(f"Average: {elapsed/len(questions):.2f} sec/question\n")

# Show sample results
print("Sample Answers (first 10):")
for i in range(min(10, len(answers))):
    q = questions[i]
    a = answers[i]
    print(f"\n{i+1}. [{q['subject']}] {q['question'][:80]}...")
    print(f"   Expected: {q['answer'][:100]}...")
    print(f"   Got: {a['answer'][:100]}...")

# Save results
output_df = pd.DataFrame(answers)
output_df.to_json("test_results.json", orient='records', indent=2)
print(f"\n💾 Saved {len(answers)} answers to test_results.json")
