"""
Feature extraction for AI vs Human text detection.
Phrase matching for explainability in Streamlit app only.
Actual model features come from TF-IDF in train.py.
"""
import re

AI_HEDGE_PATTERN     = r'\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence)\b'
AI_FILLER_PATTERN    = r'\b(in conclusion|in summary|it is worth noting|it should be noted|this study aims)\b'
AI_TECHNICAL_PATTERN = r'\b(significantly|approximately|substantially|notably|primarily)\b'
PASSIVE_PATTERN      = r'\b(is|are|was|were|been|being)\s+[a-z]{4,}ed\b'
CITATION_PATTERN     = r'\[\d+\]|\(\d{4}\)'

def get_phrase_matches(text: str) -> dict:
    """Find AI/Human signal phrases for explainability."""
    tl = text.lower()
    matches = {
        'hedge_words':       re.findall(AI_HEDGE_PATTERN, tl),
        'filler_phrases':    re.findall(AI_FILLER_PATTERN, tl),
        'technical_density': re.findall(AI_TECHNICAL_PATTERN, tl),
        'passive_voice':     re.findall(PASSIVE_PATTERN, tl),
        'citation_markers':  re.findall(CITATION_PATTERN, text),
    }
    return {k: list(set(v)) for k, v in matches.items() if v}