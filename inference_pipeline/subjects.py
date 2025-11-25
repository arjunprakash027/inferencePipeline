"""
Subject detection for the inference pipeline.
"""

def detect_subject(question: str) -> str:
    """Fast subject detection"""
    q = question.lower()
    if any(kw in q for kw in ['solve', 'equation', 'factor', 'matrix']): return 'algebra'
    if any(kw in q for kw in ['country', 'capital', 'mountain', 'river']): return 'geography'
    if any(kw in q for kw in ['war', 'revolution', 'when did', 'who was']): return 'history'
    return 'general'
