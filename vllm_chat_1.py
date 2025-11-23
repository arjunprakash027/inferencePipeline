import torch
from vllm import LLM, SamplingParams

# -----------------------------
# Full ultimate algebra teaching prompt
# -----------------------------
ULTIMATE_ALGEBRA_PROMPT = """
You are the world's most patient, perfect, step-by-step algebra teacher. 
You NEVER skip steps. You NEVER give short answers. You explain EVERYTHING like the student is seeing it for the first time. 
FOLLOW THESE RULES EXACTLY (no exceptions): 
ALWAYS combine like terms and show which ones 
ALWAYS distribute negatives correctly and show the distribution 
ALWAYS show PEMDAS order explicitly 
ALWAYS do the exact same operation to both sides of an equation 
ALWAYS distribute before anything else when parentheses are present 
When simplifying, write every single intermediate step 
When solving equations, use the "undo" method in reverse PEMDAS order 
For word problems: define variables → write equation → solve step-by-step → complete sentence answer 

KEY TEACHING POINTS YOU MUST ALWAYS MENTION:
• PEMDAS = Parentheses → Exponents → Multiply/Divide left-to-right → Add/Subtract left-to-right 
• When you see -(something), distribute the negative to EVERY term inside 
• Like terms must have the exact same variable and exponent 
• To move a term across the equal sign, do the opposite operation to both sides 
• Always check your final answer by substituting back into the original equation 

EXAMPLES YOU MUST IMITATE EXACTLY:
Expression: 2x + 4[2 - (5x - 3)] → First, distribute the negative inside: 2 - (5x - 3) = 2 - 5x + 3 = -5x + 5 → Now multiply by 4: 4[-5x + 5] = -20x + 20 → Put together: 2x + (-20x + 20) = 2x - 20x + 20 = -18x + 20
Expression: -(3 - 2(x + 5)) + 4(x - 1) → Distribute the negative: -(3 - 2x - 10) + 4x - 4 = -3 + 2x + 10 + 4x - 4 → Combine like terms: (2x + 4x) + (-3 + 10 - 4) = 6x + 3
Equation: 5x - 7 = 2(x + 6) → Distribute right side: 5x - 7 = 2x + 12 → Subtract 2x from both sides: 3x - 7 = 12 → Add 7 to both sides: 3x = 19 → Divide by 3: x = 19/3
Word problem example: "The perimeter of a rectangle is 54 cm. Length is 6 cm more than width." → Let width = w cm → Then length = w + 6 cm → Perimeter = 2w + 2(w + 6) = 2w + 2w + 12 = 4w + 12 = 54 → Subtract 12: 4w = 42 → Divide by 4: w = 10.5 → Length = 16.5 → Final answer: The width is 10.5 cm and the length is 16.5 cm.

YOU MUST NEVER SAY "just distribute" — you must SHOW every term. 
YOU MUST NEVER SAY "combine like terms" — you must SHOW which terms and how. 
YOU MUST NEVER GIVE ONLY THE FINAL ANSWER. 

Now wait for the student's question and teach perfectly.
"""

# -----------------------------
# Load FP16 model from app/models
# -----------------------------
MODEL = "app/models/your-llama-3b"  # original FP16 model

print(f"Loading FP16 model from {MODEL} with vLLM on GPU…")

llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    dtype="half",           # basic FP16 quantization
    trust_remote_code=True,
    enforce_eager=False,
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024,
    stop=["<|eot_id|>", "<|end_of_text|>"]
)

tokenizer = llm.get_tokenizer()
print("Model loaded! Ready to teach algebra.\n")

# -----------------------------
# Interactive loop
# -----------------------------
history = [{"role": "system", "content": "You are a helpful assistant."}]

math_keywords = [
    "solve", "equation", "algebra", "factor", "polynomial", "expression",
    "simplify", "quadratic", "perimeter", "area", "derivative", "integral"
]

while True:
    try:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "q", "bye"}:
            print("Goodbye!")
            break
        if not user:
            continue

        is_math_question = any(keyword in user.lower() for keyword in math_keywords)
        system_content = ULTIMATE_ALGEBRA_PROMPT if is_math_question else "You are a helpful assistant."

        history.append({"role": "system", "content": system_content})
        history.append({"role": "user", "content": user})

        prompt = tokenizer.apply_chat_template(
            history,
            chat_template=None,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=False
        )

        print("Assistant: ", end="", flush=True)
        outputs = llm.generate([prompt], sampling_params)
        reply = outputs[0].outputs[0].text
        print(reply)
        print()

        history.append({"role": "assistant", "content": reply})
        history = [msg for msg in history if msg["role"] != "system" or msg["content"] == "You are a helpful assistant."]

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
