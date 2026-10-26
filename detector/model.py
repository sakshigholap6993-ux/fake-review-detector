"""Machine-learning half of the detector: a TF-IDF + logistic regression classifier.

Trained by ``train_model.py`` on the public Fake Reviews Dataset (Salminen et al.,
2022: 20k genuine Amazon reviews vs 20k computer-generated ones).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "review_model.joblib"


@lru_cache(maxsize=1)
def load_model() -> dict | None:
    """Return the saved bundle ``{"pipeline", "metrics", ...}`` or None if not trained yet."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def fake_probability(text: str) -> float | None:
    bundle = load_model()
    if bundle is None:
        return None
    pipeline = bundle["pipeline"]
    fake_index = list(pipeline.classes_).index(1)
    return float(pipeline.predict_proba([text])[0][fake_index])
