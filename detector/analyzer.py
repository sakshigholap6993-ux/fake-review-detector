"""Combines the ML model and the rule-based signals into one verdict."""

from __future__ import annotations

from dataclasses import dataclass

from . import heuristics, model

FAKE_THRESHOLD = 0.5
MIN_WORDS = 3


@dataclass
class Analysis:
    text: str
    fake_probability: float
    ml_probability: float | None
    rule_score: float
    signals: list[heuristics.Signal]
    threshold: float = FAKE_THRESHOLD

    @property
    def is_fake(self) -> bool:
        return self.fake_probability >= self.threshold

    @property
    def verdict(self) -> str:
        return "FAKE" if self.is_fake else "REAL"

    @property
    def confidence(self) -> float:
        """How far the score sits past the cutoff toward its end of the scale, as 50%-100%.

        Measured against the active threshold, so a stricter or looser setting never
        yields a verdict backed by under 50% confidence.
        """
        p, t = self.fake_probability, self.threshold
        margin = (p - t) / (1 - t) if self.is_fake else (t - p) / t
        return 0.5 + 0.5 * margin

    @property
    def low_confidence(self) -> bool:
        return abs(self.fake_probability - self.threshold) < 0.1


def analyze(text: str, rating: int | None = None) -> Analysis:
    text = text.strip()
    if len(heuristics.WORD_RE.findall(text)) < MIN_WORDS:
        raise ValueError(f"Please provide at least {MIN_WORDS} words of review text.")

    signals = heuristics.run_signals(text, rating)
    rule_score = heuristics.combine(signals)
    ml_prob = model.fake_probability(text)
    # The two detectors catch different fakes: the model spots machine-written text,
    # the rules spot human-written hype/spam. Either one being confident is enough.
    final = rule_score if ml_prob is None else max(ml_prob, rule_score)
    return Analysis(text, final, ml_prob, rule_score, signals)
