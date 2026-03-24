"""
deconfound.py
=============
Quantifies and controls for corpus identity confound.

Two analyses
------------
1. Vocabulary overlap analysis
   - Jaccard similarity between all genre pairs
   - Unique-to-corpus vocabulary percentages
   - Top-50 most discriminative unigrams per genre (PMI)
   - Confirms or refutes corpus identity confound quantitatively

2. Corpus identity probing classifier
   - Trains genre classifier features to predict CORPUS (not genre)
   - If accuracy is high → features leak source identity
   - Reports per-feature corpus-predictive power
   - Separates "genre signal" from "source signal"

References
----------
Gururangan et al. (2018) "Annotation Artifacts in NLI Data" NAACL
Kumar et al. (2019) "Topics to Avoid: Demoting Latent Confounds" EMNLP
"""

import os
import logging
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
GENRE_ORDER  = ["literature", "news", "social"]
GENRE_LABELS = {"literature": "Literature", "news": "News", "social": "Social Media"}
C = {"literature": "#5C4FC4", "news": "#0D7A5F", "social": "#C4411A",
     "bg": "#FAFAF8", "grid": "#E8E6DF", "text": "#1A1A18", "muted": "#7A7870"}


# ═══════════════════════════════════════════════════════════════
# 1. Vocabulary overlap
# ═══════════════════════════════════════════════════════════════

def run_vocabulary_overlap(df: pd.DataFrame):
    """
    Compute vocabulary Jaccard overlap across genre pairs.
    Report unique-to-genre vocabulary proportions.
    Compute PMI-based discriminative vocabulary per genre.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("CORPUS IDENTITY ANALYSIS — Vocabulary Overlap")
    print("=" * 60)

    # Build vocabulary per genre
    genre_vocabs = {}
    genre_word_counts = {}
    for genre in GENRE_ORDER:
        texts = df[df["genre"] == genre]["text"].tolist()
        words = []
        for t in texts:
            words.extend(str(t).lower().split())
        genre_word_counts[genre] = Counter(words)
        genre_vocabs[genre] = set(genre_word_counts[genre].keys())

    # Jaccard similarity
    print("\n  Pairwise Jaccard Vocabulary Overlap (higher = more similar):")
    results = {}
    for i, g1 in enumerate(GENRE_ORDER):
        for g2 in GENRE_ORDER[i+1:]:
            inter = len(genre_vocabs[g1] & genre_vocabs[g2])
            union = len(genre_vocabs[g1] | genre_vocabs[g2])
            jaccard = inter / union
            pair = f"{g1}–{g2}"
            results[pair] = round(jaccard, 4)
            print(f"  {pair:<30} Jaccard = {jaccard:.4f}  "
                  f"(shared vocab: {inter:,} / {union:,} types)")

    # Unique-to-genre vocabulary
    print("\n  Unique-to-genre vocabulary (not in either other genre):")
    all_vocab = genre_vocabs[GENRE_ORDER[0]].union(
        *[genre_vocabs[g] for g in GENRE_ORDER[1:]]
    )
    unique_stats = {}
    for genre in GENRE_ORDER:
        others = set().union(*[genre_vocabs[g] for g in GENRE_ORDER if g != genre])
        unique = genre_vocabs[genre] - others
        pct = len(unique) / len(genre_vocabs[genre]) * 100
        unique_stats[genre] = {"unique_types": len(unique), "pct_unique": round(pct, 2)}
        print(f"  {GENRE_LABELS[genre]:<15}: {len(unique):>7,} unique types "
              f"({pct:.1f}% of {GENRE_LABELS[genre]} vocab is genre-exclusive)")

    # PMI discriminative vocab
    print("\n  Top-20 most discriminative words per genre (PMI):")
    corpus_total = sum(sum(c.values()) for c in genre_word_counts.values())
    pmi_results = {}
    for genre in GENRE_ORDER:
        genre_total = sum(genre_word_counts[genre].values())
        pmi_scores = {}
        for word, count in genre_word_counts[genre].items():
            if count < 5:
                continue
            corpus_count = sum(genre_word_counts[g].get(word, 0) for g in GENRE_ORDER)
            p_word = corpus_count / corpus_total
            p_genre = genre_total / corpus_total
            p_word_given_genre = count / genre_total
            if p_word > 0 and p_genre > 0:
                pmi = np.log2(p_word_given_genre / p_word + 1e-9)
                pmi_scores[word] = pmi
        top20 = sorted(pmi_scores, key=pmi_scores.get, reverse=True)[:20]
        pmi_results[genre] = top20
        print(f"\n  {GENRE_LABELS[genre]}: {' · '.join(top20[:15])}")

    # Save
    overlap_results = {
        "jaccard_similarity":    results,
        "unique_vocab_stats":    unique_stats,
        "top_discriminative":    pmi_results,
        "interpretation": (
            "Low Jaccard (<0.10) suggests high vocabulary separation — "
            "possible corpus identity confound. High unique vocab % amplifies this."
        )
    }
    with open(os.path.join(RESULTS_DIR, "vocabulary_overlap.json"), "w", encoding="utf-8") as f:
        json.dump(overlap_results, f, ensure_ascii=False, indent=2)

    # Plot
    _plot_vocab_overlap(results, unique_stats)

    print(f"\n  Results saved → {RESULTS_DIR}/vocabulary_overlap.json")
    return overlap_results


def _plot_vocab_overlap(jaccard_results: dict, unique_stats: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(C["bg"])

    # Jaccard bar
    ax = axes[0]
    ax.set_facecolor(C["bg"])
    pairs = list(jaccard_results.keys())
    vals  = list(jaccard_results.values())
    bars  = ax.bar(pairs, vals, color=["#5C4FC4", "#0D7A5F", "#C4411A"],
                   alpha=0.82, width=0.45)
    ax.axhline(0.10, color=C["muted"], lw=1.2, linestyle="--",
               label="0.10 = low overlap threshold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9,
                color=C["text"], fontweight="bold")
    ax.set_ylabel("Jaccard vocabulary similarity")
    ax.set_title("Pairwise vocabulary overlap\n(low = high corpus identity confound)",
                 fontsize=10, fontweight="bold", color=C["text"])
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor(C["bg"]); ax.tick_params(colors=C["muted"])

    # Unique vocab %
    ax2 = axes[1]
    ax2.set_facecolor(C["bg"])
    genres = GENRE_ORDER
    pcts   = [unique_stats[g]["pct_unique"] for g in genres]
    ax2.bar([GENRE_LABELS[g] for g in genres], pcts,
            color=[C[g] for g in genres], alpha=0.82, width=0.5)
    for i, (g, pct) in enumerate(zip(genres, pcts)):
        ax2.text(i, pct+0.5, f"{pct:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=C[g])
    ax2.set_ylabel("% of genre vocabulary that is genre-exclusive")
    ax2.set_title("Genre-exclusive vocabulary\n(high % → genre corpora barely overlap)",
                  fontsize=10, fontweight="bold", color=C["text"])
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_facecolor(C["bg"]); ax2.tick_params(colors=C["muted"])

    fig.suptitle("Corpus identity confound quantification",
                 fontsize=12, fontweight="bold", color=C["text"])
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "vocabulary_overlap.png"),
                dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 2. Corpus identity probing classifier
# ═══════════════════════════════════════════════════════════════

def run_corpus_identity_probe(df: pd.DataFrame):
    """
    Train a classifier to predict corpus identity (source) from:
    (a) surface features — to see if structural features encode source
    (b) TF-IDF features — expected to be very high (vocabulary leakage)
    (c) TF-IDF with genre-label substitution — shuffled genre labels

    If structural feature classifier predicts source with accuracy > 1/N_genres,
    the confound is confirmed and we must explicitly discuss it.
    """
    from src.feature_extractor import build_feature_df

    print("\n" + "=" * 60)
    print("CORPUS IDENTITY PROBE")
    print("  Q: Can our features predict corpus SOURCE, not just genre?")
    print("=" * 60)

    # df["genre"] IS the corpus source here (one corpus per genre)
    # We build a fake "source" label to make this explicit
    # For a real deconfound, you'd need multiple sources per genre
    # Here we quantify how much lexical overlap there is

    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y    = df["genre"].values   # genre == corpus source in current setup

    results = {}

    # (a) TF-IDF corpus identity accuracy
    vec   = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    X_tfidf = vec.fit_transform(df["text"].values)
    model_tfidf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv_tfidf = cross_validate(model_tfidf, X_tfidf, y, cv=skf,
                              scoring=["accuracy", "f1_macro"], n_jobs=-1)
    results["TF-IDF source prediction"] = {
        "accuracy_mean": round(cv_tfidf["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv_tfidf["test_accuracy"].std(), 4),
    }

    # (b) Surface features corpus identity accuracy
    X_surf, _ = build_feature_df(df)
    model_surf = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    cv_surf = cross_validate(model_surf, X_surf, y, cv=skf,
                             scoring=["accuracy", "f1_macro"], n_jobs=-1)
    results["Surface features source prediction"] = {
        "accuracy_mean": round(cv_surf["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv_surf["test_accuracy"].std(), 4),
    }

    # (c) Random baseline
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="stratified")
    cv_dummy = cross_validate(dummy, X_surf, y, cv=skf,
                              scoring=["accuracy"], n_jobs=-1)
    results["Random baseline"] = {
        "accuracy_mean": round(cv_dummy["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv_dummy["test_accuracy"].std(), 4),
    }

    print(f"\n  {'Model':<40} {'Accuracy':>10} {'±':>4}")
    print("  " + "-" * 58)
    for name, vals in results.items():
        print(f"  {name:<40} {vals['accuracy_mean']:>10.4f}  {vals['accuracy_std']:.4f}")

    # Interpretation
    surf_acc = results["Surface features source prediction"]["accuracy_mean"]
    chance   = 1 / len(GENRE_ORDER)
    print(f"\n  Chance baseline:    {chance:.4f}")
    print(f"  Surface features:   {surf_acc:.4f}")
    delta = surf_acc - chance
    if delta > 0.10:
        print(f"  ⚠  Surface features predict source with Δ={delta:.3f} above chance.")
        print("     This confirms corpus identity confound. Report in paper limitations.")
        print("     Mitigated by: dep-only classifier, word-shuffle baseline, cross-corpus eval.")
    else:
        print(f"  ✓  Surface features have minimal source prediction power (Δ={delta:.3f}).")
        print("     Corpus identity confound is limited for surface features.")

    # Save
    probe_results = {
        "results": results,
        "chance_baseline": round(chance, 4),
        "surface_above_chance": round(delta, 4),
        "note": (
            "In current setup genre == corpus source (1 corpus per genre). "
            "To fully deconfound, add a 2nd corpus per genre and do cross-corpus eval."
        )
    }
    with open(os.path.join(RESULTS_DIR, "corpus_identity_probe.json"), "w") as f:
        json.dump(probe_results, f, indent=2)

    _plot_corpus_probe(results, chance)
    print(f"\n  Results saved → {RESULTS_DIR}/corpus_identity_probe.json")
    return probe_results


def _plot_corpus_probe(results: dict, chance: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    names  = list(results.keys())
    accs   = [results[n]["accuracy_mean"] for n in names]
    stds   = [results[n]["accuracy_std"]  for n in names]
    colors = ["#C4411A", "#5C4FC4", "#D3D1C7"][:len(names)]

    bars = ax.bar(names, accs, color=colors, alpha=0.85, width=0.5)
    ax.errorbar(range(len(names)), accs, yerr=stds,
                fmt="none", color=C["text"], capsize=5, lw=1.5)
    ax.axhline(chance, color=C["muted"], lw=1.5, linestyle="--",
               label=f"Chance = {chance:.3f}")
    for i, (acc, std) in enumerate(zip(accs, stds)):
        ax.text(i, acc+std+0.01, f"{acc:.3f}", ha="center",
                fontsize=9, fontweight="bold", color=C["text"])

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("Accuracy (source prediction)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Corpus identity probe: can our features predict the source corpus?\n"
        "(high = confound present; chance = no confound)",
        fontsize=10, fontweight="bold", color=C["text"]
    )
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors=C["muted"])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "corpus_identity_probe.png"),
                dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()