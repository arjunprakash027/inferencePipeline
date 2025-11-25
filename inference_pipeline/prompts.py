"""
Subject-specific prompts for the inference pipeline.
"""

def create_expert_prompt(question: str, subject: str) -> str:
    """Enhanced prompts with few-shot examples"""
    if subject == "algebra":
        few_shot = """Examples:
Q: Solve for x: 3x - 7 = 14
A: Add 7 to both sides: 3x = 21. Divide by 3: x = 7.
Q: What is the quadratic formula?
A: x = (-b ± √(b² - 4ac)) / 2a.
Q: Expand (x + 2)(x - 2)
A: This is a difference of squares: x² - 4.
Now answer:"""
        system = "You are an expert algebra tutor."
    elif subject == "geography":
        few_shot = """Examples:
Q: What is the capital of France?
A: Paris.
Q: Which river is the longest in the world?
A: The Nile River (though the Amazon is contended).
Q: In which continent is the Sahara Desert located?
A: Africa.
Now answer:"""
        system = "You are an expert geographer."
    elif subject == "history":
        few_shot = """Examples:
Q: Who was the first Emperor of Rome?
A: Augustus Caesar (Octavian).
Q: When did the Titanic sink?
A: April 15, 1912.
Q: What was the main cause of World War I?
A: The assassination of Archduke Franz Ferdinand, along with alliances, imperialism, and nationalism.
Now answer:"""
        system = "You are an expert historian."
    else:
        few_shot = "Answer the question directly."
        system = "You are a helpful assistant."

    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{few_shot}\\n\\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
