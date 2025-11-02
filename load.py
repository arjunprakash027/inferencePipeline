import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inferencePipeline import loadPipeline

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


if __name__ == '__main__':
    # ====== CONFIG ======
    input_file = "sample_questions.xlsx"   
    output_file = "answers_output.csv"
    # =====================

    print("[MAIN] Starting pipeline run...")
    
    # Load data
    print(f"[MAIN] Reading questions from {input_file}...")
    df = pd.read_excel(input_file)
    
    print(f"[MAIN] Total rows read: {len(df)}")
    print("[MAIN] Preview:")
    print(df.head())

    # Prepare data for pipeline
    questions = []
    for _, row in df.iterrows():
        questions.append({
            "questionID": row["questionID"],
            "question": row["question"]
        })
    print(f"[MAIN] Prepared {len(questions)} questions for inference.")

    # Load inference pipeline
    pipeline = loadPipeline()
    print("[MAIN] Pipeline loaded successfully!")

    # Measure inference time
    start_time = time.time()
    print("[MAIN] Running inference...")
    answers = pipeline(questions)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"[MAIN] Inference complete in {total_time:.2f} seconds.")

    # Merge results back into DataFrame
    print("[MAIN] Merging results with original DataFrame...")
    answers_df = pd.DataFrame(answers)
    merged_df = df.merge(answers_df, on="questionID", how="left", suffixes=('', '_generated'))

    # Save results
    merged_df.to_csv(output_file, index=False)
    print(f"[MAIN] Results saved to {output_file}")
    
    # Optional: print few results
    print("\n[MAIN] Sample output:")
    print(merged_df[['question', 'answer_generated']].head())

    print("[MAIN] Done.")
