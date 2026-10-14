"""
Context Analyzer and Novelty Detection for ParamMap.
Scores current conversation context against known domains and detects shifts.
"""

from typing import Dict, List, Tuple, Optional
import math
import re
from dataclasses import dataclass
from .domains import DomainDefinition

@dataclass
class ContextAnalysisResult:
    domain_scores: Dict[str, float]  # domain_id -> score in [0, 1]
    dominant_domain: str
    dominant_score: float
    is_novel_domain: bool
    context_drift: float             # Drift magnitude compared to previous state

class ContextAnalyzer:
    """
    Analyzes input text in real-time, emitting domain affinity scores
    and tracking conversational topic drift.
    """

    def __init__(
        self,
        domains: List[DomainDefinition],
        drift_alpha: float = 0.7,
        novelty_threshold: float = 0.20
    ):
        self.domains = {d.domain_id: d for d in domains}
        self.drift_alpha = drift_alpha
        self.novelty_threshold = novelty_threshold
        self.previous_scores: Dict[str, float] = {d_id: 1.0 / len(self.domains) for d_id in self.domains}

    def update_domains(self, domains: List[DomainDefinition]) -> None:
        self.domains = {d.domain_id: d for d in domains}

    def analyze_lexical(self, text: str) -> Dict[str, float]:
        """
        Fast zero-overhead lexical domain scoring based on keywords and patterns.
        """
        lower = text.lower()
        raw_scores: Dict[str, float] = {}

        # Distinct syntax patterns
        code_markers = len(re.findall(r"\b(def|class|import|return|async|await|const|var|function|nullptr|include)\b|\bfrom\s+\w+\s+import\b", text))
        code_brackets = sum(text.count(sym) for sym in ["{", "}", "->", "::", "//", "/*", "*/", "=="])
        
        math_special = sum(text.count(sym) for sym in ["\\int", "\\sqrt", "∑", "π", "λ", "∈", "∀", "∃", "≠", "≈", "^"])
        math_functions = len(re.findall(r"\b(sin|cos|tan|log|exp|lim|integral|derivative|matrix|eigen)\b", text, re.IGNORECASE))
        # Match real arithmetic operators between tokens/numbers, avoiding hyphens in words like thread-safe
        math_ops = len(re.findall(r"(?:\d+\s*[\+\-\*\/]\s*\d+|[a-z]\s*[\+\*]\s*[a-z]|\s[=<>]\s)", text, re.IGNORECASE))

        for domain_id, domain in self.domains.items():
            score = 0.0
            for kw in domain.keywords:
                if kw.lower() in lower:
                    score += 2.5

            if domain_id.startswith("code"):
                score += code_markers * 3.5 + code_brackets * 1.0
            elif domain_id.startswith("math"):
                # If strong code markers are present, dampen simple arithmetic operators
                dampener = 0.3 if code_markers > 0 else 1.0
                score += math_special * 3.0 + math_functions * 2.5 + (math_ops * 1.0 * dampener)

            raw_scores[domain_id] = score

        # Softmax / Normalize
        total = sum(raw_scores.values()) + 1e-6
        if total <= 1e-4:
            # Uniform fallback
            uniform = 1.0 / len(self.domains)
            return {d_id: uniform for d_id in self.domains}

        return {d_id: raw_scores[d_id] / total for d_id in self.domains}

    def analyze(self, text: str) -> ContextAnalysisResult:
        """
        Performs full analysis on incoming text.
        """
        current_scores = self.analyze_lexical(text)

        # Topic drift: Euclidean distance between current and EMA previous
        drift = math.sqrt(
            sum((current_scores[d] - self.previous_scores.get(d, 0.0)) ** 2 for d in current_scores)
        )

        # Update EMA state
        for d in current_scores:
            self.previous_scores[d] = (
                self.drift_alpha * current_scores[d] +
                (1.0 - self.drift_alpha) * self.previous_scores.get(d, 0.0)
            )

        dominant_domain = max(current_scores.keys(), key=lambda k: current_scores[k])
        dominant_score = current_scores[dominant_domain]
        is_novel = dominant_score < self.novelty_threshold

        return ContextAnalysisResult(
            domain_scores=current_scores,
            dominant_domain=dominant_domain,
            dominant_score=dominant_score,
            is_novel_domain=is_novel,
            context_drift=drift,
        )
