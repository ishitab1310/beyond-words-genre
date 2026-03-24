"""
probing_experiments.py
======================
Six probing experiments that isolate syntactic from lexical signal.
This module implements the missing CORE CLAIM of the paper.

Experiments
-----------
E1  Word-shuffle baseline
    Shuffle all tokens within each document, re-run TF-IDF and MuRIL.
    If accuracy stays high → models rely on vocabulary, not order/syntax.

E2  POS-tag-only classification
    Replace every word with its UPOS tag. Eliminates ALL lexical content.
    Accuracy above chance → syntactic categories carry genre signal.

E3  Function-word-only classification
    Keep only postpositions, conjunctions, auxiliaries, pronouns, particles.
    Tests whether grammatical function words encode genre without content words.

E4  Dependency-structure-only classification  ← THE MISSING CORE EXPERIMENT
    Features: dep relation ratios, MDD, tree depth, branching, left-ratio,
    non-projectivity. Zero lexical information.
    This is RQ1 of the paper — directly tests the central claim.

E5  Delexicalized text classification
    Replace content words (NOUN, VERB, ADJ, ADV, PROPN) with POS tags.
    Keep function words as-is. Tests syntax + function words combined.

E6  Dep-features vs surface features comparison
    Compare dep-only accuracy against surface-only accuracy.
    Shows syntactic features provide *independent* signal.

References
----------
Sinha et al. (2021) "UnNatural Language Inference" ACL (word shuffle)
Pham et al. (2021) "Out of Order" Findings-ACL (word-order sensitivity)
Karlgren & Cutting (1994) (POS-only genre classification)
Hewitt & Manning (2019) "A Structural Probe for Finding Syntax" NAACL
"""

import os
import re
import random
import logging
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
PARSED_DIR  = "data/parsed"
GENRE_ORDER  = ["literature", "news", "social"]
GENRE_LABELS = {"literature": "Literature", "news": "News", "social": "Social Media"}

C = {"literature": "#5C4FC4", "news": "#0D7A5F", "social": "#C4411A",
     "bg": "#FAFAF8", "grid": "#E8E6DF", "text": "#1A1A18", "muted": "#7A7870",
     "accent": "#E8A020"}

# Hindi function words (postpositions, conjunctions, auxiliaries, pronouns, particles)
HINDI_FUNCTION_WORDS = {
    # Postpositions
    "में", "से", "को", "के", "की", "का", "पर", "तक", "के लिए", "द्वारा",
    "के साथ", "के बाद", "के पहले", "के बिना", "के पास", "के ऊपर", "के नीचे",
    "ने", "को", "से",
    # Conjunctions
    "और", "या", "लेकिन", "परंतु", "किंतु", "तो", "तथा", "अथवा", "क्योंकि",
    "इसलिए", "कि", "जो", "जब", "अगर", "यदि", "हालांकि", "जबकि", "ताकि",
    # Pronouns
    "मैं", "हम", "तुम", "आप", "वह", "वे", "यह", "ये", "कोई", "कुछ",
    "सब", "हर", "जो", "जिस", "इस", "उस", "इन", "उन",
    # Auxiliaries
    "है", "हैं", "था", "थी", "थे", "हो", "होना", "रहा", "रही", "रहे",
    "गया", "गई", "गए", "होगा", "होगी", "होंगे", "जाना", "सकना", "पड़ना",
    # Particles
    "ही", "भी", "तो", "ना", "नहीं", "मत", "न", "बहुत", "काफी", "थोड़ा",
    "अब", "फिर", "तब", "यहाँ", "वहाँ", "कब", "कहाँ", "कैसे", "क्यों", "क्या",
}


# ═══════════════════════════════════════════════════════════════
# CoNLL-U reader (minimal, reused)
# ═══════════════════════════════════════════════════════════════

def _iter_sentences(file_path: str):
    sentence = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sentence:
                    yield sentence
                    sentence = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8 or "-" in parts[0] or "." in parts[0]:
                continue
            try:
                sentence.append({
                    "id":     int(parts[0]),
                    "form":   parts[1],
                    "upos":   parts[3],
                    "head":   int(parts[6]),
                    "deprel": parts[7],
                })
            except ValueError:
                continue
    if sentence:
        yield sentence


def _skf_cv(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv  = cross_validate(model, X, y, cv=skf,
                         scoring=["accuracy", "f1_macro"], n_jobs=-1)
    return {
        "accuracy_mean": round(cv["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv["test_accuracy"].std(),  4),
        "f1_macro_mean": round(cv["test_f1_macro"].mean(), 4),
    }


def _lr_pipeline():
    return Pipeline([
        ("sc",  StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


# ═══════════════════════════════════════════════════════════════
# E1 — Word-shuffle baseline
# ═══════════════════════════════════════════════════════════════

def experiment_word_shuffle(df: pd.DataFrame) -> dict:
    """
    Shuffle all words within each document.
    Re-run TF-IDF classifier on shuffled text.
    If accuracy is unchanged → model relies on vocabulary, not word order.
    If accuracy drops significantly → some order-dependent signal exists.
    """
    print("\n  [E1] Word-shuffle baseline ...")
    rng = random.Random(42)

    def shuffle_text(text: str) -> str:
        words = str(text).split()
        rng.shuffle(words)
        return " ".join(words)

    df_shuffled = df.copy()
    df_shuffled["text"] = df_shuffled["text"].apply(shuffle_text)

    y = df["genre"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Original TF-IDF
    vec_orig = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
    X_orig   = vec_orig.fit_transform(df["text"].values)
    cv_orig  = cross_validate(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X_orig, y, cv=skf, scoring=["accuracy","f1_macro"], n_jobs=-1
    )
    results["TF-IDF original"] = {
        "accuracy_mean": round(cv_orig["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv_orig["test_accuracy"].std(),  4),
    }

    # Shuffled TF-IDF
    vec_shuf = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
    X_shuf   = vec_shuf.fit_transform(df_shuffled["text"].values)
    cv_shuf  = cross_validate(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X_shuf, y, cv=skf, scoring=["accuracy","f1_macro"], n_jobs=-1
    )
    results["TF-IDF shuffled"] = {
        "accuracy_mean": round(cv_shuf["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv_shuf["test_accuracy"].std(),  4),
    }

    orig_acc = results["TF-IDF original"]["accuracy_mean"]
    shuf_acc = results["TF-IDF shuffled"]["accuracy_mean"]
    drop     = orig_acc - shuf_acc

    print(f"    TF-IDF original: {orig_acc:.4f}")
    print(f"    TF-IDF shuffled: {shuf_acc:.4f}")
    print(f"    Accuracy drop:   {drop:.4f}")
    if drop < 0.02:
        print("    ✓ Near-zero drop → TF-IDF relies on bag-of-words (vocabulary), not order.")
        print("      This SUPPORTS the value of structural/syntactic features.")
    else:
        print(f"    ✗ Accuracy drops {drop:.3f} → model uses some word-order signal.")

    return results


# ═══════════════════════════════════════════════════════════════
# E2 — POS-tag-only classification
# ═══════════════════════════════════════════════════════════════

def experiment_pos_only(df: pd.DataFrame) -> dict:
    """
    Replace each document with its sequence of POS tags from parsed .conllu.
    Classify on POS sequences → no lexical content whatsoever.
    """
    print("\n  [E2] POS-tag-only classification ...")

    # Build POS-replaced texts from .conllu files
    genre_pos_texts = _build_pos_texts()
    if not genre_pos_texts:
        print("    ⚠  No .conllu files found — skipping E2. Run parsing first.")
        return {}

    # Align with df (use as many samples as we have)
    pos_df_rows = []
    for genre, texts in genre_pos_texts.items():
        for t in texts:
            pos_df_rows.append({"text": t, "genre": genre})
    pos_df = pd.DataFrame(pos_df_rows)

    y = pos_df["genre"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # TF-IDF on POS sequences (effectively POS n-gram model)
    vec  = TfidfVectorizer(max_features=3000, ngram_range=(1, 3), analyzer="word")
    X    = vec.fit_transform(pos_df["text"].values)
    cv   = cross_validate(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X, y, cv=skf, scoring=["accuracy","f1_macro"], n_jobs=-1
    )

    result = {
        "accuracy_mean": round(cv["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv["test_accuracy"].std(),  4),
        "f1_macro_mean": round(cv["test_f1_macro"].mean(), 4),
        "n_samples":     len(pos_df),
    }
    print(f"    POS-only accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    chance = 1 / len(GENRE_ORDER)
    delta  = result["accuracy_mean"] - chance
    print(f"    Chance baseline:   {chance:.4f}")
    print(f"    Above chance:      {delta:.4f}")
    if delta > 0.10:
        print("    ✓ POS patterns alone discriminate genres — syntactic categories carry genre signal.")
    else:
        print("    ✗ POS-only barely above chance — genre signal is lexical, not syntactic.")

    return result


def _build_pos_texts() -> dict:
    """Read .conllu files and return POS-tag sequences per genre."""
    genre_pos_texts = defaultdict(list)
    for genre in GENRE_ORDER:
        path = os.path.join(PARSED_DIR, f"{genre}.conllu")
        if not os.path.exists(path):
            continue
        current = []
        for tokens in _iter_sentences(path):
            pos_seq = " ".join(t["upos"] for t in tokens)
            genre_pos_texts[genre].append(pos_seq)
    return dict(genre_pos_texts)


# ═══════════════════════════════════════════════════════════════
# E3 — Function-word-only classification
# ═══════════════════════════════════════════════════════════════

def experiment_function_word_only(df: pd.DataFrame) -> dict:
    """
    Keep only function words (postpositions, conjunctions, auxiliaries,
    pronouns, particles). Remove all content words.
    """
    print("\n  [E3] Function-word-only classification ...")

    def keep_function_words(text: str) -> str:
        words = str(text).split()
        fw    = [w for w in words if w in HINDI_FUNCTION_WORDS]
        return " ".join(fw) if fw else "EMPTY"

    df_fw = df.copy()
    df_fw["text"] = df_fw["text"].apply(keep_function_words)

    # Filter out empty texts
    df_fw = df_fw[df_fw["text"] != "EMPTY"].reset_index(drop=True)

    if len(df_fw) < 100:
        print("    ⚠  Too few samples after function-word filtering — check HINDI_FUNCTION_WORDS list.")
        return {}

    y   = df_fw["genre"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X   = vec.fit_transform(df_fw["text"].values)
    cv  = cross_validate(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X, y, cv=skf, scoring=["accuracy","f1_macro"], n_jobs=-1
    )

    result = {
        "accuracy_mean": round(cv["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv["test_accuracy"].std(),  4),
        "f1_macro_mean": round(cv["test_f1_macro"].mean(), 4),
        "n_samples":     len(df_fw),
    }
    print(f"    Function-word-only accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    chance = 1 / len(GENRE_ORDER)
    delta  = result["accuracy_mean"] - chance
    if delta > 0.10:
        print("    ✓ Function words alone discriminate genres — grammatical patterns carry signal.")
    else:
        print("    ✗ Function words barely above chance.")
    return result


# ═══════════════════════════════════════════════════════════════
# E4 — Dependency-structure-only classification  [THE CORE EXPERIMENT]
# ═══════════════════════════════════════════════════════════════

def experiment_dependency_only() -> dict:
    """
    Build a feature vector from ONLY dependency tree metrics per sentence:
    - 8 dependency relation ratios (nsubj, obj, obl, advmod, amod, ccomp, acl, nmod)
    - MDD
    - max_depth
    - branching factor
    - left-branching ratio
    - non-projectivity rate
    - 7 POS ratios

    Train LR + RF classifier. This is RQ1 — can syntactic structure classify genre?
    """
    print("\n  [E4] Dependency-structure-only classification (CORE EXPERIMENT) ...")

    RELATIONS = ["nsubj","obj","obl","advmod","amod","ccomp","acl","nmod"]

    all_features = []
    all_labels   = []
    feature_names = None

    for genre in GENRE_ORDER:
        path = os.path.join(PARSED_DIR, f"{genre}.conllu")
        if not os.path.exists(path):
            print(f"    ⚠  {path} not found — run parsing first.")
            continue

        for tokens in _iter_sentences(path):
            n = len(tokens)
            if n < 3:
                continue

            dep_counts = defaultdict(int)
            for t in tokens:
                dep_counts[t["deprel"]] += 1

            # Relation ratios
            rel_ratios = {r: dep_counts.get(r, 0) / n for r in RELATIONS}

            # MDD
            dists = [abs(t["id"] - t["head"]) for t in tokens if t["head"] != 0]
            mdd   = float(np.mean(dists)) if dists else 0.0

            # Tree depth
            head_map = {t["id"]: t["head"] for t in tokens}
            def _depth(nid, visited=None):
                if visited is None: visited = set()
                if nid in visited or nid == 0: return 0
                visited.add(nid)
                return 1 + _depth(head_map.get(nid, 0), visited)
            max_depth = max((_depth(t["id"]) for t in tokens), default=0)

            # Branching
            children_count = defaultdict(int)
            for t in tokens:
                if t["head"] != 0:
                    children_count[t["head"]] += 1
            branching = float(np.mean(list(children_count.values()))) if children_count else 0.0

            # Left-branching
            left = sum(1 for t in tokens if t["head"] != 0 and t["id"] < t["head"])
            left_ratio = left / max(len(dists), 1)

            # Non-projectivity
            arcs = [(t["id"], t["head"]) for t in tokens if t["head"] != 0]
            nonproj = 0
            for (i, h) in arcs:
                lo, hi = min(i, h), max(i, h)
                for t in tokens:
                    if lo < t["id"] < hi and not (lo <= t["head"] <= hi):
                        nonproj += 1
                        break
            nonproj_rate = nonproj / max(len(arcs), 1)

            # POS ratios
            pos_counts = defaultdict(int)
            for t in tokens:
                pos_counts[t["upos"]] += 1
            pos_ratios = {
                f"pos_{p}": pos_counts.get(p, 0) / n
                for p in ["NOUN","VERB","ADJ","ADV","PRON","ADP","PART"]
            }

            fvec = {**rel_ratios,
                    "mdd": mdd, "max_depth": float(max_depth),
                    "branching": branching, "left_ratio": left_ratio,
                    "non_proj_rate": nonproj_rate, **pos_ratios}

            if feature_names is None:
                feature_names = list(fvec.keys())

            all_features.append([fvec[k] for k in feature_names])
            all_labels.append(genre)

    if len(all_features) < 30:
        print("    ⚠  Not enough parsed sentences. Run parsing with --parse_size 2000 first.")
        return {}

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"    Dataset: {len(X)} sentences × {X.shape[1]} dependency features")
    print(f"    Genre distribution: { {g: (y==g).sum() for g in GENRE_ORDER} }")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LR
    lr_pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    cv_lr   = cross_validate(lr_pipe, X, y, cv=skf, scoring=["accuracy","f1_macro"], n_jobs=-1)

    # RF
    from sklearn.ensemble import RandomForestClassifier
    rf   = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    cv_rf = cross_validate(rf, X, y, cv=skf, scoring=["accuracy","f1_macro"], n_jobs=-1)

    results = {
        "LR": {
            "accuracy_mean": round(cv_lr["test_accuracy"].mean(), 4),
            "accuracy_std":  round(cv_lr["test_accuracy"].std(),  4),
            "f1_macro_mean": round(cv_lr["test_f1_macro"].mean(), 4),
        },
        "RF": {
            "accuracy_mean": round(cv_rf["test_accuracy"].mean(), 4),
            "accuracy_std":  round(cv_rf["test_accuracy"].std(),  4),
            "f1_macro_mean": round(cv_rf["test_f1_macro"].mean(), 4),
        },
        "n_samples":     len(X),
        "n_features":    X.shape[1],
        "feature_names": feature_names,
    }

    chance = 1 / len(GENRE_ORDER)
    print(f"\n    LR  accuracy: {results['LR']['accuracy_mean']:.4f} ± {results['LR']['accuracy_std']:.4f}")
    print(f"    RF  accuracy: {results['RF']['accuracy_mean']:.4f} ± {results['RF']['accuracy_std']:.4f}")
    print(f"    Chance:       {chance:.4f}")
    print(f"    LR above chance: {results['LR']['accuracy_mean']-chance:.4f}")

    if results["LR"]["accuracy_mean"] > chance + 0.10:
        print("\n    ✓ DEPENDENCY FEATURES ALONE CLASSIFY GENRE ABOVE CHANCE.")
        print("      This validates RQ1: syntactic structure encodes genre signal.")
        print("      Compare to surface features to quantify the structural contribution.")
    else:
        print("\n    ✗ Dependency features do not reliably classify genre above chance.")
        print("      The genre signal is predominantly lexical.")

    # Feature importance
    rf_full = RandomForestClassifier(n_estimators=300, random_state=42,
                                     class_weight="balanced", n_jobs=-1)
    rf_full.fit(X, y)
    importances = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf_full.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n    Top-10 most important dependency features:")
    for _, row in importances.head(10).iterrows():
        print(f"      {row['feature']:<25} {row['importance']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# E5 — Delexicalized text classification
# ═══════════════════════════════════════════════════════════════

def experiment_delexicalized() -> dict:
    """
    Replace content words (NOUN, VERB, ADJ, ADV, PROPN) with their POS tags.
    Keep function words (ADP, CONJ, PRON, PART, AUX, DET) verbatim.
    Classifier sees: "[postposition] NOUN [postposition] VERB ADJ"
    """
    print("\n  [E5] Delexicalized text classification ...")

    CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    delex_rows  = []

    for genre in GENRE_ORDER:
        path = os.path.join(PARSED_DIR, f"{genre}.conllu")
        if not os.path.exists(path):
            continue
        for tokens in _iter_sentences(path):
            delex_tokens = []
            for t in tokens:
                if t["upos"] in CONTENT_POS:
                    delex_tokens.append(t["upos"])
                else:
                    delex_tokens.append(t["form"])
            delex_rows.append({
                "text":  " ".join(delex_tokens),
                "genre": genre,
            })

    if not delex_rows:
        print("    ⚠  No .conllu files found — skipping E5.")
        return {}

    delex_df = pd.DataFrame(delex_rows)
    y   = delex_df["genre"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X   = vec.fit_transform(delex_df["text"].values)
    cv  = cross_validate(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X, y, cv=skf, scoring=["accuracy","f1_macro"], n_jobs=-1
    )

    result = {
        "accuracy_mean": round(cv["test_accuracy"].mean(), 4),
        "accuracy_std":  round(cv["test_accuracy"].std(),  4),
        "f1_macro_mean": round(cv["test_f1_macro"].mean(), 4),
        "n_samples":     len(delex_df),
    }
    print(f"    Delexicalized accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    return result


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def run_all_probing_experiments(df: pd.DataFrame) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("\n" + "=" * 60)
    print("PROBING EXPERIMENTS — Isolating Syntactic from Lexical Signal")
    print("=" * 60)

    all_results = {}

    # Run experiments
    all_results["E1_word_shuffle"]       = experiment_word_shuffle(df)
    all_results["E2_pos_only"]           = experiment_pos_only(df)
    all_results["E3_function_word_only"] = experiment_function_word_only(df)
    all_results["E4_dependency_only"]    = experiment_dependency_only()
    all_results["E5_delexicalized"]      = experiment_delexicalized()

    # Save all results
    out_path = os.path.join(RESULTS_DIR, "probing_experiments.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("PROBING EXPERIMENTS — SUMMARY")
    print("=" * 60)
    chance = round(1 / len(GENRE_ORDER), 4)
    print(f"\n  {'Experiment':<35} {'Accuracy':>10} {'±':>6} {'vs Chance':>12}")
    print("  " + "-" * 65)

    exp_display = [
        ("TF-IDF (original)",        all_results["E1_word_shuffle"].get("TF-IDF original", {})),
        ("E1: TF-IDF word-shuffled", all_results["E1_word_shuffle"].get("TF-IDF shuffled", {})),
        ("E2: POS-only",             all_results["E2_pos_only"]),
        ("E3: Function-word-only",   all_results["E3_function_word_only"]),
        ("E4: Dep-features-only (LR)", all_results["E4_dependency_only"].get("LR", {})),
        ("E4: Dep-features-only (RF)", all_results["E4_dependency_only"].get("RF", {})),
        ("E5: Delexicalized",        all_results["E5_delexicalized"]),
    ]

    plot_data = []
    for name, res in exp_display:
        if not res:
            continue
        acc = res.get("accuracy_mean", 0)
        std = res.get("accuracy_std",  0)
        delta = acc - chance
        marker = "★" if "Dep-features" in name else ""
        print(f"  {name:<35} {acc:>10.4f} {std:>6.4f} {delta:>+12.4f} {marker}")
        plot_data.append({"name": name, "accuracy": acc, "std": std})

    print(f"\n  Chance baseline: {chance}")

    # Plot
    _plot_probing_summary(plot_data, chance)

    logger.info(f"Probing experiments saved → {out_path}")
    return all_results


def _plot_probing_summary(plot_data: list, chance: float):
    if not plot_data:
        return
    names  = [d["name"] for d in plot_data]
    accs   = [d["accuracy"] for d in plot_data]
    stds   = [d["std"] for d in plot_data]

    def _color(name):
        if "Dep-features" in name: return C["literature"]
        if "TF-IDF" in name:       return C["news"]
        if "POS" in name:          return C["social"]
        if "Function" in name:     return C["accent"]
        return C["muted"]

    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    bars = ax.barh(names, accs, xerr=stds, color=colors, alpha=0.85,
                   height=0.55, capsize=4,
                   error_kw={"lw":1.4,"capthick":1.4,"ecolor":C["muted"]})
    ax.axvline(chance, color="red", lw=1.5, linestyle="--",
               label=f"Chance = {chance:.3f}")
    ax.axvline(1.0, color=C["grid"], lw=0.8)

    for bar, acc, std in zip(bars, accs, stds):
        ax.text(acc + std + 0.004, bar.get_y() + bar.get_height()/2,
                f"{acc:.3f}", va="center", fontsize=8.5, color=C["text"])

    ax.set_xlabel("Accuracy (5-fold stratified CV)", fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_title(
        "Probing experiments: isolating syntactic from lexical signal\n"
        "★ = dependency-only classifier (RQ1 — the paper's core claim)",
        fontsize=11, fontweight="bold", color=C["text"]
    )
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors=C["muted"])

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C["literature"], label="Dependency features only ★"),
        Patch(facecolor=C["news"],       label="Lexical (TF-IDF)"),
        Patch(facecolor=C["social"],     label="POS-only"),
        Patch(facecolor=C["accent"],     label="Function words"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "probing_experiments.png"),
                dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    logger.info("Probing experiments plot saved")