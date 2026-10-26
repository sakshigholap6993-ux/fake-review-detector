"""Train the ML half of the detector and save it to models/review_model.joblib.

Dataset: "Fake Reviews Dataset" (Salminen et al., 2022, https://osf.io/tyue9/) -
40k Amazon reviews, half genuine (label OR) and half computer-generated (label CG).

Usage:
    python train_model.py                      # downloads the dataset if needed
    python train_model.py --data my_reviews.csv --text-col text --label-col label --fake-label 1
"""

from __future__ import annotations

import argparse
import urllib.request
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from detector.model import MODEL_PATH

DATASET_URL = "https://osf.io/download/3vds7/"
DEFAULT_DATA = Path("data/fake_reviews.csv")


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60_000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                  max_features=60_000, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=4.0, max_iter=2000)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--text-col", default="text_")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--fake-label", default="CG")
    args = parser.parse_args()

    if not args.data.exists():
        if args.data != DEFAULT_DATA:
            raise SystemExit(f"Dataset not found: {args.data}")
        print(f"Downloading dataset to {args.data} ...")
        args.data.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, args.data)

    df = pd.read_csv(args.data).dropna(subset=[args.text_col, args.label_col])
    texts = df[args.text_col].astype(str)
    labels = (df[args.label_col].astype(str) == str(args.fake_label)).astype(int)
    print(f"Loaded {len(df):,} reviews ({labels.sum():,} fake, {(1 - labels).sum():,} real)")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)

    pipeline = build_pipeline().fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "test_size": len(y_test),
    }
    print(classification_report(y_test, pred, target_names=["real", "fake"], digits=3))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    # Refit on everything for the shipped model; metrics above stay honest (held-out).
    pipeline = build_pipeline().fit(texts, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "metrics": metrics,
        "dataset": "Fake Reviews Dataset (Salminen et al., 2022)",
        "n_reviews": len(df),
        "trained_on": date.today().isoformat(),
    }, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
