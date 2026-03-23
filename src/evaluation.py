"""
evaluation.py
Shared evaluation utilities: model comparison table, result aggregation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
GENRES      = ["literature", "news", "social"]
GENRE_COLORS = {
    "literature": "#534AB7",
    "news":       "#0F6E56",
    "social":     "#993C1D",
}


def save_confusion_matrix(y_true, y_pred, tag: str = "model", cmap: str = "Blues"):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=GENRES)
    disp = ConfusionMatrixDisplay(cm, display_labels=GENRES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f"Confusion matrix — {tag}")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"confusion_matrix_{tag}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {out}")


def aggregate_results():
    """
    Load all result JSON files and print a unified model comparison table.
    Also saves results/model_comparison.csv.
    """
    rows = []

    # Structural classifier
    clf_path = os.path.join(RESULTS_DIR, "classifier_results.json")
    if os.path.exists(clf_path):
        with open(clf_path) as f:
            clf = json.load(f)
        for model_key, vals in clf.items():
            rows.append({
                "model": f"Structural ({model_key.upper()})",
                "accuracy": vals.get("accuracy_mean"),
                "accuracy_std": vals.get("accuracy_std"),
                "f1_macro": vals.get("f1_macro_mean"),
                "evaluation": "5-fold CV",
            })

    # TF-IDF
    tfidf_path = os.path.join(RESULTS_DIR, "tfidf_results.json")
    if os.path.exists(tfidf_path):
        with open(tfidf_path) as f:
            tfidf = json.load(f)
        for config, vals in tfidf.items():
            rows.append({
                "model": f"TF-IDF — {config}",
                "accuracy": vals.get("accuracy_mean"),
                "accuracy_std": vals.get("accuracy_std"),
                "f1_macro": vals.get("f1_macro_mean"),
                "evaluation": "5-fold CV",
            })

    # BERT
    bert_path = os.path.join(RESULTS_DIR, "bert_results.json")
    if os.path.exists(bert_path):
        with open(bert_path) as f:
            bert = json.load(f)
        model_name = bert.get("model", "BERT")
        acc = bert.get("accuracy")
        f1  = bert.get("classification_report", {}).get("macro avg", {}).get("f1-score")
        rows.append({
            "model": f"BERT ({model_name.split('/')[-1]})",
            "accuracy": acc,
            "accuracy_std": None,
            "f1_macro": f1,
            "evaluation": "80/20 split",
        })

    if not rows:
        logger.warning("No result files found in results/")
        return

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    out = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(out, index=False)
    logger.info(f"Model comparison saved → {out}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["accuracy"].fillna(0), color="#534AB7", alpha=0.8)

    # Error bars where available
    for i, (acc, std) in enumerate(zip(df["accuracy"], df["accuracy_std"])):
        if std is not None and not np.isnan(float(std)):
            ax.errorbar(i, acc, yerr=std, color="black", capsize=4, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison: accuracy across all systems")

    for bar, acc in zip(bars, df["accuracy"]):
        if acc is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, float(acc) + 0.01,
                    f"{float(acc):.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    logger.info("Model comparison plot saved")

    return df
