# Tech Arena 2025 - Algebra-Specific Optimization Strategy

## Current Algebra Problem Analysis
The current pipeline is producing "stupid stuff" for algebra questions due to:
1. LLM hallucinations on mathematical reasoning
2. Poor step-by-step reasoning
3. Incorrect application of mathematical formulas
4. Lack of verification for numerical answers

## Model Switching: Qwen3-4B vs Llama-3.2-3B

### Current Llama-3.2-3B Performance Issues:
- Struggles with mathematical reasoning
- Tends to produce verbose explanations instead of direct calculations
- Often makes arithmetic errors
- Not optimized for mathematical problems

### Qwen3-4B Advantages:
- Better mathematical reasoning capabilities
- Stronger performance on quantitative tasks
- More recent architecture optimized for reasoning
- Better instruction following
- If latency is good, accuracy improvement could be significant

### Recommendation: Switch to Qwen3-4B
Given that the calculator approach failed and algebra problems need better reasoning, switching to Qwen3-4B is a strategic move.

## Algebra-Specific Optimizations

### 1. Mathematical Prompt Engineering
```
You are an expert mathematician solving algebra problems. Follow this exact format:

PROBLEM: [original question]
THINKING: [think through the problem step-by-step, identifying what is given and what needs to be found]
FORMULA: [state the specific formula or method to use]
CALCULATION: [show mathematical steps clearly]
ANSWER: [state the final result clearly]

Solve the problem: [original question]
```

### 2. Chain-of-Thought Examples
Include 2-3 examples in the prompt showing proper mathematical reasoning:
```
Example 1:
Problem: If x + 5 = 12, find x.
Thinking: Need to isolate x by subtracting 5 from both sides
Calculation: x + 5 = 12 → x = 12 - 5 = 7
Answer: x = 7

Now solve: [actual problem]
```

### 3. Verification Step
Add a verification step to check answers by substitution where possible.

### 4. Mathematical Context Injection
Add relevant mathematical formulas in the prompt based on detected problem type.

## Implementation Strategy

### Phase 1: Model Switch
1. Change model from Llama-3.2-3B to Qwen3-4B
2. Update model loading to use: "Qwen/Qwen3-4B-Instruct"
3. Test performance on algebra samples

### Phase 2: Prompt Optimization
1. Implement structured mathematical reasoning prompts
2. Add chain-of-thought examples
3. Include verification steps

### Phase 3: Fine-tuning (if time permits)
1. Create algebra-specific dataset
2. Fine-tune Qwen on mathematical problems
3. Optimize for step-by-step reasoning

## Expected Impact
Switching to Qwen3-4B with optimized mathematical prompting should:
- Significantly improve algebra accuracy
- Maintain good latency
- Provide more reliable mathematical reasoning
- Reduce hallucinations in calculations

## Model Loading Configuration
```python
MODEL_NAME = "Qwen/Qwen3-4B-Instruct"  # Updated model
```

This approach leverages Qwen's strength in mathematical reasoning while maintaining the architectural advantages of your current pipeline.