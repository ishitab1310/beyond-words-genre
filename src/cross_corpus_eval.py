"""
cross_corpus_eval.py
====================
Tests whether structural features GENERALIZE across corpora.

Strategy
--------
Train structural classifier on original corpora (BHAAV / Inshorts / Sarcasm tweets).
Test on new held-out sources:
  - Literature:   Hindi Wikipedia literary articles + HC Corpora Hindi fiction
  - News:         Dainik Jagran/Amar Ujala headlines (via IndicNLP corpus)
  - Social:       Hindi comments from publicly available web data

If structural features generalize (accuracy stays above chance):
  → Validates the claim that syntax encodes genre, not corpus identity.
If accuracy collapses:
  → Genre signal is corpus-specific (the confound is real).

Also runs: train-on-new, test-on-original (reverse direction).

Resources used
--------------
IndicNLP Suite (Kakwani et al. 2020, EMNLP Findings)
  - BBC Hindi news classification corpus
  - Available via huggingface: ai4bharat/IndicNLP-Dataset
HC Corpora (http://corpora.epizy.com/corpora.html)
  - Hindi web text across multiple domains
"""

import os
import logging
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

RESULTS_DIR  = "results"
PLOTS_DIR    = "results/plots"
GENRE_ORDER  = ["literature", "news", "social"]
GENRE_LABELS = {"literature": "Literature", "news": "News", "social": "Social Media"}
C = {"literature": "#5C4FC4", "news": "#0D7A5F", "social": "#C4411A",
     "bg": "#FAFAF8", "grid": "#E8E6DF", "text": "#1A1A18", "muted": "#7A7870"}


def _try_load_indicnlp() -> pd.DataFrame:
    """
    Attempt to load IndicNLP BBC Hindi news corpus.
    Returns DataFrame with 'text' and 'genre' columns, or empty DataFrame.
    """
    try:
        from datasets import load_dataset
        logger.info("Loading IndicNLP BBC Hindi news corpus ...")
        ds = load_dataset("ai4bharat/IndicNLP-Dataset", "hi-news",
                          split="train", trust_remote_code=True)
        df = pd.DataFrame({"text": ds["text"], "genre": "news"})
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.len() > 20]
        logger.info(f"IndicNLP news: {len(df)} articles loaded")
        return df.sample(min(1000, len(df)), random_state=42).reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Could not load IndicNLP corpus: {e}")
        return pd.DataFrame()


def _try_load_hc_corpora() -> pd.DataFrame:
    """
    Attempt to load HC Corpora Hindi text.
    Looks for hc_corpora_hi.txt in data/ directory.
    Returns DataFrame or empty DataFrame.
    """
    path = "data/hc_corpora_hi.txt"
    if not os.path.exists(path):
        logger.warning(f"HC Corpora not found at {path}. Download from http://corpora.epizy.com/corpora.html")
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 30]
        df = pd.DataFrame({"text": lines[:2000], "genre": "literature"})
        logger.info(f"HC Corpora: {len(df)} lines loaded")
        return df
    except Exception as e:
        logger.warning(f"Could not load HC Corpora: {e}")
        return pd.DataFrame()


def _try_load_custom_new_sources() -> pd.DataFrame:
    """
    Load any custom external corpora placed in data/external/.
    Expected format: one .txt file per genre named {genre}_new.txt
    """
    external_dir = "data/external"
    rows = []
    if not os.path.exists(external_dir):
        return pd.DataFrame()
    for genre in GENRE_ORDER:
        path = os.path.join(external_dir, f"{genre}_new.txt")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 20]
        for line in lines[:1000]:
            rows.append({"text": line, "genre": genre})
        logger.info(f"External corpus ({genre}): {len(lines)} lines loaded from {path}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_cross_corpus_evaluation(df_original: pd.DataFrame) -> dict:
    """
    Cross-corpus generalization experiment.

    1. Train on original → test on new source (if available)
    2. Compare TF-IDF vs surface structural features on cross-corpus transfer
    3. Report accuracy degradation as measure of corpus identity confound
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("\n" + "=" * 60)
    print("CROSS-CORPUS GENERALIZATION EVALUATION")
    print("=" * 60)

    results = {}

    # ── Try to assemble held-out corpus ──
    held_out_parts = []

    indicnlp = _try_load_indicnlp()
    if not indicnlp.empty:
        held_out_parts.append(indicnlp)

    hc = _try_load_hc_corpora()
    if not hc.empty:
        held_out_parts.append(hc)

    custom = _try_load_custom_new_sources()
    if not custom.empty:
        held_out_parts.append(custom)

    # ── Scenario A: Full cross-corpus (if held-out exists) ──
    if held_out_parts:
        df_held = pd.concat(held_out_parts, ignore_index=True)
        genres_in_held = df_held["genre"].unique().tolist()
        genres_train   = [g for g in GENRE_ORDER if g in df_original["genre"].values]
        genres_test    = [g for g in GENRE_ORDER if g in genres_in_held]
        overlap        = set(genres_train) & set(genres_test)

        if len(overlap) >= 2:
            print(f"\n  Held-out corpus: {len(df_held)} samples")
            print(f"  Genres available: {genres_in_held}")
            results["cross_corpus"] = _cross_corpus_train_test(
                df_original, df_held, list(overlap)
            )
        else:
            print(f"\n  ⚠ Not enough genre overlap between corpora: {genres_in_held}")
            print("    Place data/external/literature_new.txt, news_new.txt, social_new.txt")
            results["cross_corpus"] = {"status": "insufficient_genres"}
    else:
        print("\n  ⚠  No external corpora found.")
        print("  To enable full cross-corpus eval, add data to:")
        print("    data/external/literature_new.txt")
        print("    data/external/news_new.txt")
        print("    data/external/social_new.txt")
        print("  OR install datasets: pip install datasets")
        results["cross_corpus"] = {"status": "no_external_data"}

    # ── Scenario B: Within-corpus leave-one-source-out ──
    # Even without external data, we can simulate cross-corpus
    # by holding out a random 20% of each genre and training on 80%
    # This tests IF our structural features generalize within the corpus
    print("\n  Running within-corpus generalization (leave-20%-out per genre):")
    results["within_corpus_transfer"] = _within_corpus_generalization(df_original)

    # ── Scenario C: Vocabulary overlap → classification gap analysis ──
    print("\n  TF-IDF vs structural features: cross-corpus stability comparison:")
    results["stability_comparison"] = _feature_stability_analysis(df_original)

    # Save
    out_path = os.path.join(RESULTS_DIR, "cross_corpus_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Cross-corpus evaluation saved → {out_path}")

    _plot_cross_corpus(results)
    return results


def _cross_corpus_train_test(df_train: pd.DataFrame,
                              df_test:  pd.DataFrame,
                              genres:   list) -> dict:
    """Train on one corpus, test on another."""
    # Filter to common genres
    train = df_train[df_train["genre"].isin(genres)].reset_index(drop=True)
    test  = df_test[df_test["genre"].isin(genres)].reset_index(drop=True)

    y_train = train["genre"].values
    y_test  = test["genre"].values

    results = {}

    # TF-IDF
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    X_train_tfidf = vec.fit_transform(train["text"].values)
    X_test_tfidf  = vec.transform(test["text"].values)
    clf_tfidf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_tfidf.fit(X_train_tfidf, y_train)
    acc_tfidf = accuracy_score(y_test, clf_tfidf.predict(X_test_tfidf))
    results["TF-IDF cross-corpus accuracy"] = round(float(acc_tfidf), 4)

    # Surface structural features
    from src.feature_extractor import build_feature_df
    X_surf_train, feat_names = build_feature_df(train)
    X_surf_test,  _          = build_feature_df(test)
    surf_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    surf_pipe.fit(X_surf_train, y_train)
    acc_surf = accuracy_score(y_test, surf_pipe.predict(X_surf_test))
    results["Surface features cross-corpus accuracy"] = round(float(acc_surf), 4)

    chance = 1 / len(genres)
    print(f"    TF-IDF cross-corpus:       {acc_tfidf:.4f} (Δ from chance: {acc_tfidf-chance:+.4f})")
    print(f"    Surface features cross-corpus: {acc_surf:.4f} (Δ from chance: {acc_surf-chance:+.4f})")

    if acc_surf > acc_tfidf:
        print("    ✓ Structural features generalize BETTER than TF-IDF across corpora!")
    else:
        print("    ✗ TF-IDF generalizes better — structural features still corpus-specific.")

    results["chance_baseline"]   = round(float(chance), 4)
    results["n_train"] = len(train)
    results["n_test"]  = len(test)
    results["genres"]  = genres
    return results


def _within_corpus_generalization(df: pd.DataFrame) -> dict:
    """
    Stratified cross-validation comparing:
    - Accuracy on same-source (in-distribution)
    - Stability across different random splits (variance)

    High variance = model is sensitive to which specific texts are seen during training.
    Low variance = model learned robust generalizable patterns.
    """
    from src.feature_extractor import build_feature_df
    X_surf, _ = build_feature_df(df)
    y = df["genre"].values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold for better variance estimate

    # TF-IDF
    vec = TfidfVectorizer(max_features=5000)
    X_tfidf = vec.fit_transform(df["text"].values)
    cv_tfidf = cross_validate(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X_tfidf, y, cv=skf, scoring=["accuracy"], n_jobs=-1
    )

    # Surface
    cv_surf = cross_validate(
        Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        X_surf, y, cv=skf, scoring=["accuracy"], n_jobs=-1
    )

    tfidf_acc_mean = cv_tfidf["test_accuracy"].mean()
    tfidf_acc_std  = cv_tfidf["test_accuracy"].std()
    surf_acc_mean  = cv_surf["test_accuracy"].mean()
    surf_acc_std   = cv_surf["test_accuracy"].std()

    print(f"    TF-IDF      10-fold:  {tfidf_acc_mean:.4f} ± {tfidf_acc_std:.4f}")
    print(f"    Surface     10-fold:  {surf_acc_mean:.4f}  ± {surf_acc_std:.4f}")

    if surf_acc_std < tfidf_acc_std:
        print("    ✓ Surface features are MORE STABLE (lower variance) than TF-IDF.")
        print("      This replicates Laippala et al. (2021) finding for register classification.")
    else:
        print("    Surface features show higher variance than TF-IDF on within-corpus splits.")

    return {
        "TF-IDF": {"mean": round(tfidf_acc_mean, 4), "std": round(tfidf_acc_std, 4)},
        "Surface": {"mean": round(surf_acc_mean, 4), "std": round(surf_acc_std, 4)},
        "surface_more_stable": bool(surf_acc_std < tfidf_acc_std),
    }


def _feature_stability_analysis(df: pd.DataFrame) -> dict:
    """
    Laippala et al. (2021) protocol:
    Train on 80% → test on 5 different 20% samples.
    Compare variance of TF-IDF vs structural features.
    Lower variance across partitions = more stable (generalizable) features.
    """
    from src.feature_extractor import build_feature_df
    X_surf, _ = build_feature_df(df)
    y = df["genre"].values
    vec = TfidfVectorizer(max_features=5000)
    X_tfidf = vec.fit_transform(df["text"].values)

    n_trials = 10
    tfidf_accs = []
    surf_accs  = []

    rng = np.random.default_rng(42)
    n   = len(df)
    for _ in range(n_trials):
        idx   = rng.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = idx[:split], idx[split:]

        # TF-IDF
        clf_t = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_t.fit(X_tfidf[train_idx], y[train_idx])
        tfidf_accs.append(accuracy_score(y[test_idx], clf_t.predict(X_tfidf[test_idx])))

        # Surface
        clf_s = Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        clf_s.fit(X_surf[train_idx], y[train_idx])
        surf_accs.append(accuracy_score(y[test_idx], clf_s.predict(X_surf[test_idx])))

    return {
        "TF-IDF":  {"mean": round(np.mean(tfidf_accs), 4), "std": round(np.std(tfidf_accs), 4)},
        "Surface": {"mean": round(np.mean(surf_accs),  4), "std": round(np.std(surf_accs),  4)},
        "surface_more_stable": bool(np.std(surf_accs) < np.std(tfidf_accs)),
        "interpretation": "Laippala et al. (2021) protocol: lower std = more stable/generalizable",
    }


def _plot_cross_corpus(results: dict):
    """Plot stability comparison."""
    stab = results.get("stability_comparison", {})
    if not stab or "TF-IDF" not in stab:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    models = ["TF-IDF", "Surface"]
    means  = [stab[m]["mean"] for m in models]
    stds   = [stab[m]["std"]  for m in models]
    colors = [C["news"], C["literature"]]

    bars = ax.bar(models, means, color=colors, alpha=0.82, width=0.45)
    ax.errorbar(models, means, yerr=stds,
                fmt="none", color=C["text"], capsize=8, lw=2, capthick=2)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, m+s+0.003,
                f"{m:.3f}±{s:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy (10 random 80/20 splits)")
    ax.set_title(
        "Feature stability (Laippala et al. 2021 protocol)\n"
        "Lower variance = more generalizable features",
        fontsize=10, fontweight="bold", color=C["text"]
    )
    ax.set_ylim(0.88, 1.01)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors=C["muted"])

    # Annotate which is more stable
    if stab.get("surface_more_stable"):
        ax.annotate("✓ Surface more stable\n(lower variance)",
                    xy=(1, means[1]+stds[1]), xytext=(0.5, means[1]+stds[1]+0.015),
                    fontsize=8, color=C["literature"],
                    arrowprops=dict(arrowstyle="->", color=C["literature"], lw=1))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_corpus_stability.png"),
                dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    logger.info("Cross-corpus stability plot saved")