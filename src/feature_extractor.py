"""
feature_extractor.py  (updated for EMNLP version)
==================================================
Three feature tiers:

SURFACE  (16 features)   — character/word/sentence statistics
LEXICAL  (7 features)    — stopword, conjunction, TTR ratios
SYNTACTIC (23 features)  — dep relation ratios, MDD, tree geometry, POS ratios
                           loaded from parsed .conllu files and cached

When syntactic features are available they are merged automatically.
The ablation study in classifier.py uses group masks to isolate tiers.
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STOPWORDS = set([
    "है", "का", "के", "की", "और", "को", "में", "से", "यह", "वह",
    "पर", "एक", "हैं", "था", "थी", "थे", "कि", "जो", "भी", "तो",
    "हो", "इस", "उस", "ने", "ही", "हम", "आप", "वे", "इन", "उन",
])
CONJUNCTIONS = set([
    "और", "लेकिन", "क्योंकि", "पर", "तथा", "या", "परंतु", "किंतु",
    "अथवा", "इसलिए", "जिससे", "ताकि", "मगर",
])
QUESTION_WORDS = set(["क्या", "कौन", "कहाँ", "कब", "कैसे", "क्यों", "कितना", "कितनी"])
NEGATION_WORDS = set(["नहीं", "न", "मत", "ना", "नहिं"])

# Feature group membership (for ablation)
SURFACE_FEATURE_NAMES = [
    "char_len", "word_len", "avg_word_len", "num_digits", "num_punct",
    "num_sentences", "avg_sentence_len", "digit_ratio", "punct_density",
]
LEXICAL_FEATURE_NAMES = [
    "stopword_ratio", "conjunction_count", "long_word_ratio",
    "unique_word_ratio", "question_word_ratio", "negation_ratio", "type_token_ratio",
]
SYNTACTIC_FEATURE_NAMES = [
    "nsubj", "obj", "obl", "advmod", "amod", "ccomp", "acl", "nmod",
    "mdd", "max_depth", "branching", "left_ratio", "non_proj_rate",
    "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_PRON",
    "pos_ADP", "pos_PART", "pos_CCONJ", "pos_SCONJ", "pos_PUNCT",
]


def extract_surface_features(text: str) -> dict:
    text   = str(text)
    words  = text.split()
    wl     = len(words)
    cl     = len(text)

    if wl == 0:
        return {k: 0.0 for k in SURFACE_FEATURE_NAMES + LEXICAL_FEATURE_NAMES}

    avg_word_len = float(np.mean([len(w) for w in words]))
    num_digits   = sum(c.isdigit() for c in text)
    num_punct    = sum(not c.isalnum() and not c.isspace() for c in text)
    sentences    = [s.strip() for s in text.replace(".", "।").split("।") if s.strip()]
    num_sent     = max(len(sentences), 1)

    sw   = sum(1 for w in words if w in STOPWORDS)
    cj   = sum(1 for w in words if w in CONJUNCTIONS)
    qw   = sum(1 for w in words if w in QUESTION_WORDS)
    neg  = sum(1 for w in words if w in NEGATION_WORDS)
    lw   = sum(len(w) > 6 for w in words)
    uniq = len(set(words))

    return {
        # Surface
        "char_len":           float(cl),
        "word_len":           float(wl),
        "avg_word_len":       avg_word_len,
        "num_digits":         float(num_digits),
        "num_punct":          float(num_punct),
        "num_sentences":      float(num_sent),
        "avg_sentence_len":   wl / num_sent,
        "digit_ratio":        num_digits / max(cl, 1),
        "punct_density":      num_punct  / max(cl, 1),
        # Lexical
        "stopword_ratio":     sw   / wl,
        "conjunction_count":  float(cj),
        "long_word_ratio":    lw   / wl,
        "unique_word_ratio":  uniq / wl,
        "question_word_ratio": qw  / wl,
        "negation_ratio":     neg  / wl,
        "type_token_ratio":   uniq / wl,
    }


def build_syntactic_features_from_conllu(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build per-sentence syntactic feature averages from .conllu files,
    then align them back to df by genre.

    Returns a DataFrame aligned to df's index, or None if .conllu not found.
    """
    from collections import defaultdict

    PARSED_DIR = "data/parsed"
    RELATIONS  = ["nsubj","obj","obl","advmod","amod","ccomp","acl","nmod"]
    POS_TAGS   = ["NOUN","VERB","ADJ","ADV","PRON","ADP","PART","CCONJ","SCONJ","PUNCT"]

    genre_syn_features = {}

    for genre in ["literature", "news", "social"]:
        path = os.path.join(PARSED_DIR, f"{genre}.conllu")
        if not os.path.exists(path):
            continue

        sentence_vecs = []
        current = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    if current:
                        vec = _sentence_to_vec(current, RELATIONS, POS_TAGS)
                        if vec:
                            sentence_vecs.append(vec)
                        current = []
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 8 or "-" in parts[0] or "." in parts[0]:
                    continue
                try:
                    current.append({
                        "id":     int(parts[0]),
                        "upos":   parts[3],
                        "head":   int(parts[6]),
                        "deprel": parts[7],
                    })
                except ValueError:
                    continue

        if not sentence_vecs:
            continue

        # Average across all sentences for this genre
        all_keys = list(sentence_vecs[0].keys())
        genre_avg = {k: float(np.mean([v[k] for v in sentence_vecs])) for k in all_keys}
        genre_syn_features[genre] = genre_avg

    if not genre_syn_features:
        return None

    # Broadcast genre-level averages to every row in df
    rows = []
    for _, row in df.iterrows():
        g = row["genre"]
        if g in genre_syn_features:
            rows.append(genre_syn_features[g])
        else:
            # zero-fill if genre not parsed
            rows.append({k: 0.0 for k in list(genre_syn_features.values())[0].keys()})

    syn_df = pd.DataFrame(rows, index=df.index)
    return syn_df


def _sentence_to_vec(tokens, RELATIONS, POS_TAGS) -> dict | None:
    n = len(tokens)
    if n < 3:
        return None

    from collections import defaultdict as dd
    dep_counts = dd(int)
    pos_counts = dd(int)
    for t in tokens:
        dep_counts[t["deprel"]] += 1
        pos_counts[t["upos"]]   += 1

    rel_ratios = {r: dep_counts.get(r, 0) / n for r in RELATIONS}
    pos_ratios = {f"pos_{p}": pos_counts.get(p, 0) / n for p in POS_TAGS}

    dists     = [abs(t["id"] - t["head"]) for t in tokens if t["head"] != 0]
    mdd       = float(np.mean(dists)) if dists else 0.0

    head_map  = {t["id"]: t["head"] for t in tokens}
    def _depth(nid, vis=None):
        if vis is None: vis = set()
        if nid in vis or nid == 0: return 0
        vis.add(nid)
        return 1 + _depth(head_map.get(nid, 0), vis)
    max_depth = max((_depth(t["id"]) for t in tokens), default=0)

    child_cnt   = dd(int)
    for t in tokens:
        if t["head"] != 0:
            child_cnt[t["head"]] += 1
    branching   = float(np.mean(list(child_cnt.values()))) if child_cnt else 0.0

    left        = sum(1 for t in tokens if t["head"] != 0 and t["id"] < t["head"])
    left_ratio  = left / max(len(dists), 1)

    arcs = [(t["id"], t["head"]) for t in tokens if t["head"] != 0]
    np_cnt = 0
    for i, h in arcs:
        lo, hi = min(i,h), max(i,h)
        for t in tokens:
            if lo < t["id"] < hi and not (lo <= t["head"] <= hi):
                np_cnt += 1
                break
    nonproj_rate = np_cnt / max(len(arcs), 1)

    return {
        **rel_ratios,
        "mdd": mdd, "max_depth": float(max_depth),
        "branching": branching, "left_ratio": left_ratio,
        "non_proj_rate": nonproj_rate,
        **pos_ratios,
    }


def build_feature_df(df: pd.DataFrame):
    """
    Build the full feature matrix.
    Merges surface + lexical + syntactic (if .conllu available).
    """
    logger.info("Extracting surface + lexical features ...")
    surface = df["text"].apply(extract_surface_features)
    feature_names = list(surface.iloc[0].keys())
    X = np.array(surface.apply(lambda x: list(x.values())).tolist(), dtype=float)

    # Attempt to merge syntactic features
    logger.info("Attempting to merge syntactic features from .conllu ...")
    syn_df = build_syntactic_features_from_conllu(df)
    if syn_df is not None:
        syn_cols = list(syn_df.columns)
        X = np.hstack([X, syn_df.values.astype(float)])
        feature_names += syn_cols
        logger.info(f"Merged {len(syn_cols)} syntactic features from parsed data")
    else:
        logger.info("No .conllu files found — using surface+lexical features only")

    logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, feature_names