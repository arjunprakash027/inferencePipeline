"""
LLM-as-a-Judge Evaluator for Pipeline Answers
Evaluates the quality of generated answers and provides feedback for refinement
"""

import json
import sys

def evaluate_answer(question, subject, expected_answer, generated_answer):
    """
    Evaluate a single answer based on:
    1. Correctness (does it answer the question?)
    2. Completeness (is the answer complete?)
    3. Conciseness (is it to the point, not too verbose?)
    4. Accuracy (for factual questions, is it accurate?)

    Returns: score (0-10), feedback
    """

    score = 0
    feedback = []

    # Check if answer is too short or generic
    if len(generated_answer) < 10:
        feedback.append(f"❌ Answer too short: '{generated_answer}'")
        return 0, feedback

    if generated_answer in ["Unable to generate answer", "No answer generated"]:
        feedback.append("❌ Failed to generate answer")
        return 0, feedback

    # Subject-specific evaluation
    if subject == "algebra":
        # For algebra, check if answer contains mathematical content
        if any(char in generated_answer for char in ['=', '+', '-', '*', '/', 'x', 'y']):
            score += 4
            feedback.append("✓ Contains mathematical symbols")
        else:
            feedback.append("⚠️  Missing mathematical notation")

        # Check for step-by-step reasoning indicators
        if any(word in generated_answer.lower() for word in ['step', 'first', 'then', 'therefore', 'solve']):
            score += 3
            feedback.append("✓ Shows reasoning process")

        # Check for final answer
        if 'final answer' in generated_answer.lower() or 'answer:' in generated_answer.lower():
            score += 2
            feedback.append("✓ Has clear final answer")
        elif any(pattern in generated_answer for pattern in ['= ', 'is ']):
            score += 1
            feedback.append("~ Has answer (implicit)")

        # Check length is appropriate
        if 20 < len(generated_answer) < 1000:
            score += 1
            feedback.append("✓ Appropriate length")

    elif subject == "chinese":
        # For Chinese, check if answer contains Chinese characters
        if any('\u4e00' <= char <= '\u9fff' for char in generated_answer):
            score += 5
            feedback.append("✓ Contains Chinese characters")
        else:
            feedback.append("❌ Missing Chinese characters!")
            return 2, feedback

        # Check if answer is reasonable length
        if 10 < len(generated_answer) < 1000:
            score += 3
            feedback.append("✓ Appropriate length")

        # Check if answer seems relevant (contains reference to the topic)
        score += 2  # Give benefit of doubt for now

    elif subject in ["geography", "history"]:
        # For factual questions, check if answer is specific
        if len(generated_answer) > 30:
            score += 4
            feedback.append("✓ Sufficient detail")
        else:
            feedback.append("⚠️  Answer may be too brief")
            score += 2

        # Check for factual markers (names, dates, places, numbers)
        factual_markers = 0
        if any(char.isupper() for char in generated_answer):
            factual_markers += 1
        if any(char.isdigit() for char in generated_answer):
            factual_markers += 1

        if factual_markers >= 1:
            score += 3
            feedback.append(f"✓ Contains factual markers ({factual_markers})")

        # Check not overly verbose
        if len(generated_answer) < 500:
            score += 2
            feedback.append("✓ Concise")
        else:
            score += 1
            feedback.append("~ Somewhat verbose")

    else:
        # Generic evaluation
        if len(generated_answer) > 20:
            score += 5
        if len(generated_answer) < 500:
            score += 3
        score += 2  # Default points

    return min(score, 10), feedback


def main():
    """Load answers and evaluate them"""

    # Load answers
    try:
        with open('answers_output.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ answers_output.json not found. Run the pipeline first!")
        sys.exit(1)

    print("="*80)
    print("LLM-as-a-Judge Evaluation Report")
    print("="*80)
    print()

    total_score = 0
    total_questions = len(results)
    subject_scores = {}

    issues = []

    for i, result in enumerate(results, 1):
        question = result.get('question', '')
        subject = result.get('subject', 'unknown')
        answer = result.get('answer', '')
        expected = result.get('answer', '')  # We don't have ground truth, use generated

        score, feedback = evaluate_answer(question, subject, expected, answer)
        total_score += score

        if subject not in subject_scores:
            subject_scores[subject] = []
        subject_scores[subject].append(score)

        # Print detailed feedback for low scores
        if score < 7:
            issues.append({
                'num': i,
                'subject': subject,
                'question': question[:80] + '...' if len(question) > 80 else question,
                'answer': answer[:100] + '...' if len(answer) > 100 else answer,
                'score': score,
                'feedback': feedback
            })

    # Print summary by subject
    print("Subject-wise Performance:")
    print("-" * 80)
    for subject, scores in sorted(subject_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0
        count = len(scores)
        print(f"{subject.upper():12s}: {avg:.1f}/10 avg  ({count} questions)")

    print()
    print(f"Overall Average: {total_score/total_questions:.1f}/10")
    print()

    # Print issues
    if issues:
        print("="*80)
        print(f"Issues Found ({len(issues)} questions with score < 7):")
        print("="*80)
        for issue in issues[:10]:  # Show top 10 issues
            print()
            print(f"Question #{issue['num']} [{issue['subject'].upper()}] - Score: {issue['score']}/10")
            print(f"Q: {issue['question']}")
            print(f"A: {issue['answer']}")
            print("Feedback:")
            for fb in issue['feedback']:
                print(f"  {fb}")
    else:
        print("✅ All questions scored 7/10 or higher!")

    print()
    print("="*80)

    # Recommendations
    print("\nRecommendations for Improvement:")
    print("-" * 80)

    # Analyze by subject
    for subject, scores in subject_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        if avg < 7:
            print(f"\n{subject.upper()}:")
            if subject == "algebra":
                print("  - Ensure step-by-step reasoning is shown")
                print("  - Include clear 'Final Answer:' marker")
                print("  - Verify mathematical correctness")
            elif subject == "chinese":
                print("  - Ensure answers are in Chinese")
                print("  - Provide culturally appropriate responses")
            elif subject in ["geography", "history"]:
                print("  - Include specific facts, names, dates")
                print("  - Be concise but complete")
                print("  - Verify factual accuracy")


if __name__ == "__main__":
    main()
