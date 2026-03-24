"""
classifier.py  (updated — full ablation across all 3 feature tiers)
=====================================================================
Properly ablates across:
  Surface only | Lexical only | Syntactic only (dep features)
  Surface+Lexical | Surface+Syntactic | Lexical+Syntactic | ALL

Only runs syntactic ablation if .conllu features are present in feature matrix.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.feature_extractor import (
    SURFACE_FEATURE_NAMES,
    LEXICAL_FEATURE_NAMES,
    SYNTACTIC_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)
RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
GENRES      = ["literature", "news", "social"]
C = {"literature": "#5C4FC4", "news": "#0D7A5F", "social": "#C4411A",
     "bg": "#FAFAF8", "grid": "#E8E6DF", "text": "#1A1A18", "muted": "#7A7870"}


def _pipeline(model_type="lr"):
    if model_type == "lr":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                     class_weight="balanced", n_jobs=-1)
    elif model_type == "svm":
        clf = LinearSVC(max_iter=2000, class_weight="balanced")
    return Pipeline([("sc", StandardScaler(with_mean=(model_type != "svm"))), ("clf", clf)])


def train_classifier(X, y, feature_names, n_splits=5):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\n{'='*60}\nSTRUCTURAL CLASSIFIER — {n_splits}-fold stratified CV\n{'='*60}")

    results = {}
    for model_name in ["lr", "rf", "svm"]:
        model = _pipeline(model_name)
        cv    = cross_validate(model, X, y, cv=skf,
                               scoring=["accuracy","f1_macro","f1_weighted"],
                               return_train_score=True, n_jobs=-1)
        acc  = cv["test_accuracy"].mean()
        std  = cv["test_accuracy"].std()
        f1   = cv["test_f1_macro"].mean()
        print(f"\n  {model_name.upper()}: Accuracy={acc:.4f}±{std:.4f}  F1(macro)={f1:.4f}")
        results[model_name] = {"accuracy_mean": round(acc,4), "accuracy_std": round(std,4),
                                "f1_macro_mean": round(f1,4)}

    # Full LR for interpretability artifacts
    lr_full = _pipeline("lr")
    lr_full.fit(X, y)
    _save_feature_importance(lr_full, feature_names)
    _save_confusion_matrix(X, y, skf)

    with open(os.path.join(RESULTS_DIR, "classifier_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Classifier results saved")
    return results


def _save_feature_importance(pipe, feature_names):
    clf  = pipe.named_steps["clf"]
    coef = clf.coef_
    rows = []
    for label, coefs in zip(clf.classes_, coef):
        for fname, cval in zip(feature_names, coefs):
            rows.append({"genre": label, "feature": fname, "coefficient": round(float(cval),4)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

    fig, axes = plt.subplots(1, len(clf.classes_), figsize=(14,5), sharey=True)
    fig.patch.set_facecolor(C["bg"])
    for ax, label, coefs in zip(axes, clf.classes_, coef):
        ax.set_facecolor(C["bg"])
        top_idx = np.concatenate([np.argsort(coefs)[:5], np.argsort(coefs)[-8:]])
        names   = [feature_names[i] if i < len(feature_names) else str(i) for i in top_idx]
        vals    = coefs[top_idx]
        colors  = [C[label] if v >= 0 else "#D3D1C7" for v in vals]
        ax.barh(names, vals, color=colors, alpha=0.85, height=0.65)
        ax.axvline(0, color=C["text"], lw=0.8, alpha=0.5)
        ax.set_title(label.capitalize(), fontweight="bold", color=C[label], fontsize=11)
        ax.set_xlabel("LR coefficient"); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Feature importance — LR coefficients", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _save_confusion_matrix(X, y, skf):
    all_true, all_pred = [], []
    for train_idx, test_idx in skf.split(X, y):
        p = _pipeline("rf")
        p.fit(X[train_idx], y[train_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(p.predict(X[test_idx]))

    cm   = confusion_matrix(all_true, all_pred, labels=GENRES)
    disp = ConfusionMatrixDisplay(cm, display_labels=GENRES)
    fig, ax = plt.subplots(figsize=(6,5))
    fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix — RF structural (5-fold CV)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_structural.png"), dpi=150)
    plt.close()
    print(f"\n  Classification report (RF):")
    print(classification_report(all_true, all_pred, target_names=GENRES))


def run_ablation_study(X, y, feature_names, n_splits=5):
    """
    Full ablation across all feature tiers.
    Only runs groups that have features present in the matrix.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Build index groups
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    surface_idx   = [feat_idx[f] for f in SURFACE_FEATURE_NAMES if f in feat_idx]
    lexical_idx   = [feat_idx[f] for f in LEXICAL_FEATURE_NAMES if f in feat_idx]
    syntactic_idx = [feat_idx[f] for f in SYNTACTIC_FEATURE_NAMES if f in feat_idx]
    # Also catch any extra syntactic features (pos_ prefixed etc)
    syntactic_idx += [i for i, n in enumerate(feature_names)
                      if n not in SURFACE_FEATURE_NAMES + LEXICAL_FEATURE_NAMES
                      and i not in syntactic_idx]

    has_syntactic = len(syntactic_idx) > 0

    groups = {
        "Surface only":      surface_idx,
        "Lexical only":      lexical_idx,
        "Surface + Lexical": surface_idx + lexical_idx,
        "All features":      list(range(X.shape[1])),
    }
    if has_syntactic:
        groups["Syntactic only"]          = syntactic_idx
        groups["Surface + Syntactic"]     = surface_idx + syntactic_idx
        groups["Lexical + Syntactic"]     = lexical_idx + syntactic_idx

    print(f"\n{'='*60}\nABLATION STUDY — {'with' if has_syntactic else 'WITHOUT'} syntactic features\n{'='*60}")
    print(f"\n  {'Group':<28} {'n_feats':>8} {'Accuracy':>10} {'±':>6} {'F1 macro':>10}")
    print("  " + "-" * 66)

    rows = []
    for name, indices in groups.items():
        if not indices:
            continue
        X_sub = X[:, indices]
        cv    = cross_validate(_pipeline("rf"), X_sub, y, cv=skf,
                               scoring=["accuracy","f1_macro"], n_jobs=-1)
        acc = cv["test_accuracy"].mean()
        std = cv["test_accuracy"].std()
        f1  = cv["test_f1_macro"].mean()
        marker = " ★" if name == "Syntactic only" else ""
        print(f"  {name:<28} {len(indices):>8} {acc:>10.4f} {std:>6.4f} {f1:>10.4f}{marker}")
        rows.append({"feature_group": name, "n_features": len(indices),
                     "accuracy": round(acc,4), "accuracy_std": round(std,4),
                     "f1_macro": round(f1,4)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation_study.csv"), index=False)

    # Plot
    _plot_ablation(df, has_syntactic)
    logger.info("Ablation study saved")
    return df


def _plot_ablation(df, has_syntactic):
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(df)*0.6)))
    fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])

    def _color(name):
        if "All" in name:       return "#5C4FC4"
        if "Syntactic" in name and "only" in name: return "#0D7A5F"
        if "Syntactic" in name: return "#9FE1CB"
        if "Lexical" in name and "only" in name: return "#C4411A"
        return "#B5D4F4"

    colors = [_color(g) for g in df["feature_group"]]
    bars   = ax.barh(df["feature_group"], df["accuracy"],
                     xerr=df["accuracy_std"], color=colors, alpha=0.88,
                     height=0.55, capsize=4,
                     error_kw={"lw":1.4,"capthick":1.4,"ecolor":C["muted"]})
    for bar,acc,std in zip(bars, df["accuracy"], df["accuracy_std"]):
        ax.text(acc+std+0.004, bar.get_y()+bar.get_height()/2,
                f"{acc:.3f}", va="center", fontsize=8.5, color=C["text"])

    best = df["accuracy"].max()
    ax.axvline(best, color=C["muted"], linestyle="--", lw=1.2, alpha=0.7,
               label=f"Best={best:.3f}")
    ax.set_xlabel("Accuracy (5-fold CV)"); ax.set_xlim(0.6, 1.02)
    title = "Ablation study — all 3 feature tiers" if has_syntactic else \
            "Ablation study (★ syntactic tier requires parsed .conllu)"
    ax.set_title(title, fontsize=10, fontweight="bold", color=C["text"])
    ax.invert_yaxis(); ax.legend()
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(colors=C["muted"])

    if has_syntactic:
        syn_row = df[df["feature_group"]=="Syntactic only"]
        if not syn_row.empty:
            ax.annotate("★ Core claim:\ndep features only",
                        xy=(float(syn_row["accuracy"].iloc[0])+0.002, list(df["feature_group"]).index("Syntactic only")),
                        xytext=(float(syn_row["accuracy"].iloc[0])+0.04,
                                list(df["feature_group"]).index("Syntactic only")-0.8),
                        fontsize=7.5, color=C["news"],
                        arrowprops=dict(arrowstyle="->", color=C["news"], lw=1))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ablation_study.png"), dpi=150,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close()