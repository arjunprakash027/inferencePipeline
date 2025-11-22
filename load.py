import os
import time
import json
import argparse
import pandas as pd
from datetime import datetime
from inferencePipeline import loadPipeline

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def log_experiment(name: str, method: str, num_questions: int, total_time: float):
    """Append experiment results to experiments.csv"""
    csv_file = "experiments.csv"
    
    # Calculate metrics
    avg_latency = (total_time / num_questions) * 1000 if num_questions > 0 else 0
    throughput = num_questions / total_time if total_time > 0 else 0
    
    # Prepare data
    data = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "name": [name],
        "method": [method],
        "questions": [num_questions],
        "total_time_s": [round(total_time, 2)],
        "avg_latency_ms": [round(avg_latency, 1)],
        "throughput_qps": [round(throughput, 2)]
    }
    
    df = pd.DataFrame(data)
    
    # Append to CSV (create if doesn't exist)
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)
    
    print(f"[LOG] Experiment logged to {csv_file}")
    print(f"[LOG] Method: {method} | Time: {total_time:.2f}s | Speed: {throughput:.2f} q/s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference pipeline and log results.")
    parser.add_argument("--name", type=str, default="default_run", help="Name/Description of this experiment run")
    parser.add_argument("--method", type=str, default="local", choices=["local", "server"], help="Inference method: local (default) or server")
    args = parser.parse_args()

    # Load questions
    print(f"[MAIN] Starting pipeline run ({args.method})...")
    input_file = "questions.xlsx"
    
    # Fallback for sample file name if needed
    if not os.path.exists(input_file) and os.path.exists("sample_questions.xlsx"):
        input_file = "sample_questions.xlsx"

    df = pd.read_excel(input_file)
    questions = df.to_dict('records')
    print(f"[MAIN] Loaded {len(questions)} questions")

    # Initialize pipeline based on method
    pipeline = loadPipeline(method=args.method)

    # Run inference with simple timing
    start_time = time.perf_counter()
    answers = pipeline(questions)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Log results
    log_experiment(args.name, args.method, len(questions), total_time)

    # Save answers in JSON format
    output_file = "answers_output.json"
    
    # Merge questions with answers
    results = []
    for q in questions:
        # Find matching answer
        answer_dict = next((a for a in answers if a['questionID'] == q['questionID']), None)
        
        result = {
            "questionID": q['questionID'],
            "question": q.get('question', ''),
            "answer": answer_dict['answer'] if answer_dict else "No answer generated"
        }
        
        # Add any other fields from the original question
        for key, value in q.items():
            if key not in result:
                result[key] = value
                
        results.append(result)
    
    # Write to JSON with pretty formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[MAIN] Answers saved to {output_file}")
