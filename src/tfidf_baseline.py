"""
tfidf_baseline.py
Lexical baseline: TF-IDF + Logistic Regression with k-fold CV.
Provides the upper bound that structural features are compared against.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
GENRES      = ["literature", "news", "social"]


def run_tfidf_baseline(df: pd.DataFrame, n_splits: int = 5):
    """
    Evaluate TF-IDF lexical baseline with stratified k-fold CV.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_text = df["text"].values
    y      = df["genre"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    configs = {
        "TF-IDF (unigrams, 5k)": TfidfVectorizer(max_features=5000, ngram_range=(1, 1)),
        "TF-IDF (bigrams, 10k)": TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
        "TF-IDF (char, 5k)":     TfidfVectorizer(max_features=5000,  analyzer="char_wb", ngram_range=(3, 5)),
    }

    print("\n" + "=" * 60)
    print(f"TF-IDF BASELINE — {n_splits}-fold stratified CV")
    print("=" * 60)

    results = {}
    all_true, all_pred = [], []

    for config_name, vec in configs.items():
        model = Pipeline([
            ("vec", vec),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ])
        cv = cross_validate(
            model, X_text, y, cv=skf,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            n_jobs=-1,
        )
        acc = cv["test_accuracy"].mean()
        std = cv["test_accuracy"].std()
        f1  = cv["test_f1_macro"].mean()
        print(f"\n  {config_name}")
        print(f"  Accuracy  : {acc:.4f} ± {std:.4f}")
        print(f"  F1 (macro): {f1:.4f}")
        results[config_name] = {
            "accuracy_mean": round(acc, 4),
            "accuracy_std":  round(std, 4),
            "f1_macro_mean": round(f1,  4),
        }

    # Full eval for confusion matrix using best config
    best_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    best_model = Pipeline([
        ("vec", best_vec),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
    ])

    for train_idx, test_idx in skf.split(X_text, y):
        best_model.fit(X_text[train_idx], y[train_idx])
        preds = best_model.predict(X_text[test_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(preds)

    cm = confusion_matrix(all_true, all_pred, labels=GENRES)
    disp = ConfusionMatrixDisplay(cm, display_labels=GENRES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Confusion matrix — TF-IDF baseline (5-fold CV)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_tfidf.png"), dpi=150)
    plt.close()

    print(f"\n  Classification report (TF-IDF bigrams):")
    print(classification_report(all_true, all_pred, target_names=GENRES))

    with open(os.path.join(RESULTS_DIR, "tfidf_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"TF-IDF results saved → {RESULTS_DIR}/tfidf_results.json")
    return results
