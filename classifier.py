"""
classifier.py
Structural feature classifier with:
- Stratified k-fold cross-validation (k=5)
- Feature importance (coefficient analysis)
- Ablation study across feature groups
- Confusion matrix and per-class metrics
"""

import os
import logging
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
GENRES      = ["literature", "news", "social"]

# Feature groups for ablation
SURFACE_FEATS = [
    "char_len", "word_len", "avg_word_len", "num_digits", "num_punct",
    "num_sentences", "avg_sentence_len", "digit_ratio", "punct_density",
]
LEXICAL_FEATS = [
    "stopword_ratio", "conjunction_count", "long_word_ratio",
    "unique_word_ratio", "question_word_ratio", "negation_ratio", "type_token_ratio",
]


def _make_pipeline(model_type: str = "lr"):
    if model_type == "lr":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    elif model_type == "svm":
        clf = LinearSVC(max_iter=2000, class_weight="balanced")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_splits: int = 5,
):
    """
    Train and evaluate the structural classifier using stratified k-fold CV.
    Saves results, feature importances, and confusion matrix.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print(f"STRUCTURAL CLASSIFIER — {n_splits}-fold stratified CV")
    print("=" * 60)

    results_summary = {}

    for model_name in ["lr", "rf", "svm"]:
        model = _make_pipeline(model_name)
        cv_results = cross_validate(
            model, X, y,
            cv=skf,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=True,
            n_jobs=-1,
        )

        acc_mean  = cv_results["test_accuracy"].mean()
        acc_std   = cv_results["test_accuracy"].std()
        f1_mean   = cv_results["test_f1_macro"].mean()
        f1_std    = cv_results["test_f1_macro"].std()

        print(f"\n  {model_name.upper()}:")
        print(f"  Accuracy  : {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  F1 (macro): {f1_mean:.4f} ± {f1_std:.4f}")

        results_summary[model_name] = {
            "accuracy_mean":  round(acc_mean, 4),
            "accuracy_std":   round(acc_std,  4),
            "f1_macro_mean":  round(f1_mean,  4),
            "f1_macro_std":   round(f1_std,   4),
        }

    # ---------------------------------------------------------------
    # Full Logistic Regression for interpretability
    # ---------------------------------------------------------------
    lr_pipe = _make_pipeline("lr")
    lr_pipe.fit(X, y)

    _save_feature_importance(lr_pipe, feature_names)
    _save_confusion_matrix(lr_pipe, X, y, skf, "structural")

    with open(os.path.join(RESULTS_DIR, "classifier_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"Classifier results saved → {RESULTS_DIR}/classifier_results.json")
    return results_summary


def _save_feature_importance(pipe, feature_names: list):
    """Save LR coefficient heatmap for top features per class."""
    clf   = pipe.named_steps["clf"]
    coef  = clf.coef_  # shape (n_classes, n_features)
    labels = clf.classes_

    top_n = 10
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]

    for ax, label, coefs in zip(axes, labels, coef):
        sorted_idx = np.argsort(coefs)
        top_idx    = np.concatenate([sorted_idx[:top_n], sorted_idx[-top_n:]])
        ax.barh(
            [feature_names[i] if i < len(feature_names) else str(i) for i in top_idx],
            coefs[top_idx],
            color=["#993C1D" if c < 0 else "#0F6E56" for c in coefs[top_idx]],
            alpha=0.8,
        )
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_title(f"Feature weights: {label}")
        ax.set_xlabel("LR coefficient")

    plt.suptitle("Top discriminative features per genre", y=1.02)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance plot saved")

    # Also save as CSV
    import pandas as pd
    rows = []
    for label, coefs in zip(clf.classes_, coef):
        for fname, coef_val in zip(feature_names, coefs):
            rows.append({"genre": label, "feature": fname, "coefficient": round(float(coef_val), 4)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)


def _save_confusion_matrix(pipe, X, y, skf, tag: str):
    """Aggregate confusion matrix over CV folds."""
    cm_total = np.zeros((3, 3), dtype=int)
    labels = sorted(set(y))
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        p = _make_pipeline("lr")
        p.fit(X[train_idx], y[train_idx])
        preds = p.predict(X[test_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(preds)

    cm = confusion_matrix(all_true, all_pred, labels=GENRES)
    disp = ConfusionMatrixDisplay(cm, display_labels=GENRES)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — {tag} classifier (5-fold CV)")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{tag}.png"), dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved ({tag})")

    print(f"\n  Full classification report ({tag}):")
    print(classification_report(all_true, all_pred, target_names=GENRES))


# ---------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------
def run_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_splits: int = 5,
):
    """
    Evaluate the contribution of each feature group by training on
    feature subsets and reporting accuracy.
    """
    import pandas as pd

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Build feature group index masks
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    surface_idx = [feat_idx[f] for f in SURFACE_FEATS if f in feat_idx]
    lexical_idx = [feat_idx[f] for f in LEXICAL_FEATS if f in feat_idx]
    syntactic_idx = [
        i for i, name in enumerate(feature_names)
        if name not in SURFACE_FEATS + LEXICAL_FEATS
    ]

    groups = {
        "Surface only":            surface_idx,
        "Lexical only":            lexical_idx,
        "Syntactic only":          syntactic_idx,
        "Surface + Lexical":       surface_idx + lexical_idx,
        "Surface + Syntactic":     surface_idx + syntactic_idx,
        "Lexical + Syntactic":     lexical_idx + syntactic_idx,
        "All features":            list(range(X.shape[1])),
    }

    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    print(f"  {'Group':<28} {'Accuracy':>10} {'±':>3} {'F1 macro':>10}")
    print("  " + "-" * 55)

    rows = []
    for group_name, indices in groups.items():
        if not indices:
            continue
        X_sub = X[:, indices]
        model = _make_pipeline("lr")
        cv = cross_validate(
            model, X_sub, y, cv=skf,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1,
        )
        acc = cv["test_accuracy"].mean()
        std = cv["test_accuracy"].std()
        f1  = cv["test_f1_macro"].mean()
        print(f"  {group_name:<28} {acc:>9.4f}  {std:.4f}  {f1:>9.4f}")
        rows.append({
            "feature_group": group_name,
            "n_features": len(indices),
            "accuracy": round(acc, 4),
            "accuracy_std": round(std, 4),
            "f1_macro": round(f1, 4),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "ablation_study.csv")
    df.to_csv(out, index=False)
    logger.info(f"Ablation study saved → {out}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#534AB7" if g == "All features" else "#B5D4F4" for g in df["feature_group"]]
    ax.barh(df["feature_group"], df["accuracy"], xerr=df["accuracy_std"],
            color=colors, alpha=0.85, capsize=4)
    ax.set_xlabel("Accuracy (5-fold CV)")
    ax.set_title("Ablation study: feature group contribution")
    ax.set_xlim(0, 1.0)
    for i, (acc, std) in enumerate(zip(df["accuracy"], df["accuracy_std"])):
        ax.text(acc + std + 0.005, i, f"{acc:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ablation_study.png"), dpi=150)
    plt.close()
    logger.info("Ablation study plot saved")

    return df
