"""
dependency_analysis.py
Extracts structural features from CoNLL-U parsed files and runs
statistical analysis across genres.

Metrics computed per sentence
------------------------------
- Frequency ratios of 8 dependency relations
- Mean Dependency Distance (MDD)  — Futrell et al. 2015
- Maximum tree depth
- Average branching factor
- Left-branching ratio
- Non-projectivity rate
"""

import os
import logging
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import f_oneway, kruskal
from itertools import combinations

logger = logging.getLogger(__name__)

RELATIONS = ["nsubj", "obj", "obl", "advmod", "amod", "ccomp", "acl", "nmod"]
PARSED_DIR = "data/parsed"
RESULTS_DIR = "results"


# ---------------------------------------------------------------
# CoNLL-U reader
# ---------------------------------------------------------------
def _iter_sentences(file_path: str):
    """Yield one sentence at a time as a list of token dicts."""
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
            if len(parts) < 8:
                continue
            # Skip multi-word tokens (IDs like "1-2")
            if "-" in parts[0] or "." in parts[0]:
                continue
            try:
                sentence.append({
                    "id":     int(parts[0]),
                    "form":   parts[1],
                    "upos":   parts[3],
                    "feats":  parts[5],
                    "head":   int(parts[6]),
                    "deprel": parts[7],
                })
            except ValueError:
                continue
    if sentence:
        yield sentence


# ---------------------------------------------------------------
# Per-sentence metrics
# ---------------------------------------------------------------
def _sentence_metrics(tokens: list) -> dict:
    """Compute all structural metrics for one sentence."""
    n = len(tokens)
    if n == 0:
        return None

    dep_counts = Counter(t["deprel"] for t in tokens)
    total = n

    # Relation ratios
    rel_ratios = {rel: dep_counts.get(rel, 0) / total for rel in RELATIONS}

    # Mean Dependency Distance (MDD)
    distances = [abs(t["id"] - t["head"]) for t in tokens if t["head"] != 0]
    mdd = float(np.mean(distances)) if distances else 0.0

    # Tree depth via parent traversal
    head_map = {t["id"]: t["head"] for t in tokens}

    def depth(node_id, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited or node_id == 0:
            return 0
        visited.add(node_id)
        parent = head_map.get(node_id, 0)
        return 1 + depth(parent, visited)

    max_depth = max((depth(t["id"]) for t in tokens), default=0)

    # Branching factor (children per node)
    children_count = Counter(t["head"] for t in tokens if t["head"] != 0)
    branching = float(np.mean(list(children_count.values()))) if children_count else 0.0

    # Left-branching ratio (dependent left of its head)
    left = sum(1 for t in tokens if t["head"] != 0 and t["id"] < t["head"])
    left_ratio = left / max(len(distances), 1)

    # Non-projectivity
    # Arc (i→h) is non-projective if ∃ k between i and h where head[k] is outside [i,h]
    arcs = [(t["id"], t["head"]) for t in tokens if t["head"] != 0]
    non_proj = 0
    for (i, h) in arcs:
        lo, hi = min(i, h), max(i, h)
        for t in tokens:
            if lo < t["id"] < hi:
                # token k is between i and h — check if its head is outside
                if not (lo <= t["head"] <= hi):
                    non_proj += 1
                    break
    non_proj_rate = non_proj / max(len(arcs), 1)

    # POS ratios
    pos_counts = Counter(t["upos"] for t in tokens)
    pos_ratios = {
        f"pos_{p}": pos_counts.get(p, 0) / total
        for p in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "PART"]
    }

    return {
        **rel_ratios,
        "mdd":            mdd,
        "max_depth":      float(max_depth),
        "branching":      branching,
        "left_ratio":     left_ratio,
        "non_proj_rate":  non_proj_rate,
        **pos_ratios,
    }


# ---------------------------------------------------------------
# Per-genre aggregation
# ---------------------------------------------------------------
def analyze_dependencies(parsed_dir: str = PARSED_DIR):
    """
    Read CoNLL-U files for each genre and compute per-sentence metrics.

    Returns
    -------
    results      : {genre: {metric: mean}}
    all_samples  : {genre: [per-sentence metric dicts]}
    """
    genres = ["literature", "news", "social"]
    all_samples = {}
    results = {}

    for genre in genres:
        path = os.path.join(parsed_dir, f"{genre}.conllu")
        if not os.path.exists(path):
            logger.warning(f"Missing: {path} — skipping {genre}")
            continue

        samples = []
        for tokens in _iter_sentences(path):
            m = _sentence_metrics(tokens)
            if m:
                samples.append(m)

        all_samples[genre] = samples
        if samples:
            all_keys = list(samples[0].keys())
            results[genre] = {
                k: float(np.mean([s[k] for s in samples]))
                for k in all_keys
            }
        logger.info(f"{genre}: {len(samples)} sentences analysed")

    # Persist averaged results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "dependency_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, all_samples


# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------
def print_key_patterns(results: dict):
    all_keys = list(next(iter(results.values())).keys())
    print("\n" + "=" * 60)
    print("DEPENDENCY & STRUCTURAL PATTERNS (genre means)")
    print("=" * 60)

    sections = [
        ("Dependency relation ratios", RELATIONS),
        ("Structural complexity", ["mdd", "max_depth", "branching", "left_ratio", "non_proj_rate"]),
        ("POS ratios", [k for k in all_keys if k.startswith("pos_")]),
    ]

    for title, keys in sections:
        print(f"\n--- {title} ---")
        header = f"{'Metric':<20}" + "".join(f"{g:>14}" for g in results)
        print(header)
        print("-" * len(header))
        for k in keys:
            row = f"{k:<20}"
            for g in results:
                v = results[g].get(k, 0.0)
                row += f"{v:>14.4f}"
            print(row)

    print("\n--- HIGHEST BY GENRE ---")
    for k in RELATIONS + ["mdd", "max_depth", "branching"]:
        vals = {g: results[g].get(k, 0.0) for g in results}
        best = max(vals, key=vals.get)
        print(f"{k:<20}: highest in {best}  {vals}")


def compute_statistical_significance(all_samples: dict):
    """ANOVA + Kruskal–Wallis + eta² for all metrics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    genres = list(all_samples.keys())
    if len(genres) < 2:
        logger.warning("Need ≥2 genres for significance testing")
        return

    all_keys = list(all_samples[genres[0]][0].keys())
    rows = []

    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 60)
    print(f"{'Metric':<22} {'F-stat':>9} {'p (ANOVA)':>12} {'η²':>8} {'KW p':>12} {'sig':>5}")
    print("-" * 70)

    for metric in all_keys:
        groups = [[s[metric] for s in all_samples[g]] for g in genres]

        # ANOVA
        try:
            f_stat, p_anova = f_oneway(*groups)
        except Exception:
            continue

        # Kruskal–Wallis (non-parametric)
        try:
            _, p_kw = kruskal(*groups)
        except Exception:
            p_kw = float("nan")

        # Eta-squared
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
        )
        ss_total = np.sum((all_data - grand_mean) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0

        sig = "*" if p_anova < 0.05 else ""
        print(f"{metric:<22} {f_stat:>9.2f} {p_anova:>12.6f} {eta2:>8.4f} {p_kw:>12.6f} {sig:>5}")
        rows.append({
            "metric": metric,
            "F": round(f_stat, 4),
            "p_anova": round(p_anova, 6),
            "eta2": round(eta2, 4),
            "p_kw": round(p_kw, 6),
            "significant": p_anova < 0.05,
        })

    import pandas as pd
    stats_df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "significance_tests.csv")
    stats_df.to_csv(out, index=False)
    logger.info(f"Significance results saved → {out}")
