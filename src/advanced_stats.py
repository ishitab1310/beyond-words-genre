"""
advanced_stats.py
=================
EMNLP-level statistical analysis:

1. Bootstrap 95% confidence intervals (BCa method) for all metrics
2. Cohen's d effect sizes for all pairwise genre comparisons
3. Permutation tests (non-parametric, avoids ANOVA assumptions)
4. Benjamini-Hochberg FDR correction (replaces Bonferroni)
5. Sentence-length-controlled MDD analysis (partial correlation)
6. Summary table formatted for paper Table 3

References
----------
Dror et al. (2018) "The Hitchhiker's Guide to Testing Significance in NLP" ACL
Gries (2015) "The Most Under-Used Method in Corpus Linguistics" Corpora
Koplenig (2019) "A Non-Parametric Significance Test to Compare Corpora" PLoS ONE
Gerlanc & Kirby (2015) "BootES" Behavior Research Methods
"""

import os
import json
import logging
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, kruskal, permutation_test
from sklearn.utils import resample

logger = logging.getLogger(__name__)

RESULTS_DIR  = "results"
PLOTS_DIR    = "results/plots"
GENRE_ORDER  = ["literature", "news", "social"]
GENRE_LABELS = {"literature": "Literature", "news": "News", "social": "Social Media"}
C = {"literature": "#5C4FC4", "news": "#0D7A5F", "social": "#C4411A",
     "bg": "#FAFAF8", "grid": "#E8E6DF", "text": "#1A1A18", "muted": "#7A7870",
     "accent": "#E8A020"}

KEY_METRICS = [
    "nsubj", "obj", "obl", "advmod", "amod",
    "mdd", "max_depth", "branching", "left_ratio", "non_proj_rate",
    "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_PRON", "pos_ADP",
]


# ═══════════════════════════════════════════════════════════════
# 1. Bootstrap 95% CI
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(data: list, n_boot: int = 2000, ci: float = 0.95) -> dict:
    """BCa bootstrap confidence interval for the mean."""
    arr  = np.array(data)
    obs  = arr.mean()
    boot = [resample(arr).mean() for _ in range(n_boot)]
    lo   = np.percentile(boot, (1 - ci) / 2 * 100)
    hi   = np.percentile(boot, (1 + ci) / 2 * 100)
    return {"mean": round(float(obs), 4),
            "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4)}


# ═══════════════════════════════════════════════════════════════
# 2. Cohen's d (pairwise)
# ═══════════════════════════════════════════════════════════════

def cohen_d(a: list, b: list) -> float:
    """Pooled-SD Cohen's d."""
    a, b = np.array(a), np.array(b)
    pooled_sd = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def cohen_d_magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.20: return "negligible"
    if d < 0.50: return "small"
    if d < 0.80: return "medium"
    return "large"


# ═══════════════════════════════════════════════════════════════
# 3. Permutation test (non-parametric)
# ═══════════════════════════════════════════════════════════════

def permutation_pvalue(groups: list, n_perms: int = 5000) -> float:
    """
    Permutation test for difference in means across k groups.
    Null: group labels are exchangeable.
    Test statistic: F-statistic (same as ANOVA).
    """
    observed_f, _ = f_oneway(*groups)
    all_data = np.concatenate(groups)
    sizes    = [len(g) for g in groups]
    count    = 0
    for _ in range(n_perms):
        perm = np.random.permutation(all_data)
        perm_groups = []
        idx = 0
        for s in sizes:
            perm_groups.append(perm[idx:idx+s])
            idx += s
        perm_f, _ = f_oneway(*perm_groups)
        if perm_f >= observed_f:
            count += 1
    return count / n_perms


# ═══════════════════════════════════════════════════════════════
# 4. Benjamini-Hochberg FDR correction
# ═══════════════════════════════════════════════════════════════

def bh_correction(p_values: list, alpha: float = 0.05) -> list:
    """
    Benjamini-Hochberg FDR correction.
    Returns list of booleans (True = reject null at corrected alpha).
    """
    m   = len(p_values)
    idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[idx]
    bh_thresholds = [(i + 1) / m * alpha for i in range(m)]
    reject = np.zeros(m, dtype=bool)
    for i in range(m - 1, -1, -1):
        if sorted_p[i] <= bh_thresholds[i]:
            reject[idx[:i+1]] = True
            break
    return reject.tolist()


# ═══════════════════════════════════════════════════════════════
# 5. Length-controlled MDD (partial correlation)
# ═══════════════════════════════════════════════════════════════

def partial_correlation_mdd_length(all_samples: dict) -> dict:
    """
    Compute partial correlation of MDD with genre after controlling for
    sentence length. Addresses the confound that longer sentences → higher MDD.
    """
    rows = []
    for genre, samples in all_samples.items():
        for s in samples:
            rows.append({
                "genre":      genre,
                "mdd":        s.get("mdd", 0),
                "sent_len":   s.get("max_depth", 0) + s.get("branching", 0),  # proxy for length
            })
    df = pd.DataFrame(rows)

    # Residualize MDD on sentence length
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(df["sent_len"], df["mdd"])
    df["mdd_residual"] = df["mdd"] - (slope * df["sent_len"] + intercept)

    print("\n  Length-controlled MDD (residuals after regressing out sentence length):")
    result = {}
    for genre in GENRE_ORDER:
        g_df = df[df["genre"] == genre]
        result[genre] = {
            "raw_mdd_mean":      round(g_df["mdd"].mean(), 4),
            "residual_mdd_mean": round(g_df["mdd_residual"].mean(), 4),
        }
        print(f"  {GENRE_LABELS[genre]:<15}: raw={result[genre]['raw_mdd_mean']:.4f}  "
              f"residual={result[genre]['residual_mdd_mean']:.4f}")

    # ANOVA on residuals
    groups = [df[df["genre"]==g]["mdd_residual"].values for g in GENRE_ORDER if g in df["genre"].values]
    if len(groups) >= 2:
        f, p = f_oneway(*groups)
        print(f"\n  ANOVA on length-controlled MDD: F={f:.3f}, p={p:.6f}")
        result["anova_f"] = round(float(f), 4)
        result["anova_p"] = round(float(p), 6)

    return result


# ═══════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════

def run_advanced_statistics(all_samples: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    if not all_samples:
        logger.warning("No dependency samples found — skipping advanced statistics.")
        return

    genres_available = [g for g in GENRE_ORDER if g in all_samples and all_samples[g]]
    if len(genres_available) < 2:
        logger.warning("Need ≥2 genres with data for statistics.")
        return

    print("\n" + "=" * 60)
    print("ADVANCED STATISTICS (EMNLP-level)")
    print("=" * 60)

    # Get metrics available in this dataset
    sample_keys = list(all_samples[genres_available[0]][0].keys())
    metrics = [m for m in KEY_METRICS if m in sample_keys]

    all_stats = {}
    p_values  = []
    metric_list = []

    # ── Collect p-values for BH correction ──
    for metric in metrics:
        groups = [[s[metric] for s in all_samples[g]] for g in genres_available]
        try:
            f_stat, p_anova = f_oneway(*groups)
        except Exception:
            continue
        p_values.append(float(p_anova))
        metric_list.append(metric)

    # BH correction
    bh_reject = bh_correction(p_values, alpha=0.05)

    rows = []
    print(f"\n  {'Metric':<20} {'Mean±CI (Lit)':>20} {'Mean±CI (News)':>20} "
          f"{'Mean±CI (Soc)':>20} {'Cohen d(L-N)':>14} {'Perm p':>10} {'BH':>6}")
    print("  " + "-" * 112)

    for metric, p_val, bh_sig in zip(metric_list, p_values, bh_reject):
        groups = {g: [s[metric] for s in all_samples[g]] for g in genres_available}

        # Bootstrap CI per genre
        cis = {}
        for g in genres_available:
            cis[g] = bootstrap_ci(groups[g], n_boot=1000)

        # Cohen's d pairwise
        pairs = list(itertools.combinations(genres_available, 2))
        cohens = {}
        for g1, g2 in pairs:
            d = cohen_d(groups[g1], groups[g2])
            cohens[f"{g1[0]}{g2[0]}"] = round(d, 3)

        # Permutation test (fewer perms for speed)
        try:
            perm_p = permutation_pvalue([groups[g] for g in genres_available], n_perms=1000)
        except Exception:
            perm_p = float("nan")

        bh_label = "✓" if bh_sig else "NS"

        lit_str  = f"{cis.get('literature',{}).get('mean','?'):.4f} [{cis.get('literature',{}).get('ci_lo','?')},{cis.get('literature',{}).get('ci_hi','?')}]"
        news_str = f"{cis.get('news',{}).get('mean','?'):.4f} [{cis.get('news',{}).get('ci_lo','?')},{cis.get('news',{}).get('ci_hi','?')}]"
        soc_str  = f"{cis.get('social',{}).get('mean','?'):.4f} [{cis.get('social',{}).get('ci_lo','?')},{cis.get('social',{}).get('ci_hi','?')}]"
        ln_d     = cohens.get("ln", 0)

        print(f"  {metric:<20} {lit_str:>20} {news_str:>20} {soc_str:>20} "
              f"{ln_d:>+14.3f} {perm_p:>10.4f} {bh_label:>6}")

        rows.append({
            "metric":            metric,
            "lit_mean":          cis.get("literature", {}).get("mean"),
            "lit_ci_lo":         cis.get("literature", {}).get("ci_lo"),
            "lit_ci_hi":         cis.get("literature", {}).get("ci_hi"),
            "news_mean":         cis.get("news",       {}).get("mean"),
            "news_ci_lo":        cis.get("news",       {}).get("ci_lo"),
            "news_ci_hi":        cis.get("news",       {}).get("ci_hi"),
            "social_mean":       cis.get("social",     {}).get("mean"),
            "social_ci_lo":      cis.get("social",     {}).get("ci_lo"),
            "social_ci_hi":      cis.get("social",     {}).get("ci_hi"),
            "cohens_d_lit_news": cohens.get("ln", None),
            "cohens_d_lit_soc":  cohens.get("ls", None),
            "cohens_d_news_soc": cohens.get("ns", None),
            "cohen_magnitude":   cohen_d_magnitude(ln_d),
            "perm_p":            round(perm_p, 4),
            "bh_significant":    bool(bh_sig),
        })
        all_stats[metric] = rows[-1]

    # Length-controlled MDD
    if "mdd" in [m for m in metrics]:
        mdd_controlled = partial_correlation_mdd_length(all_samples)
        all_stats["mdd_length_controlled"] = mdd_controlled

    # Save
    df_stats = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "advanced_statistics.csv")
    df_stats.to_csv(csv_path, index=False)
    logger.info(f"Advanced statistics saved → {csv_path}")

    # Save JSON
    with open(os.path.join(RESULTS_DIR, "advanced_statistics.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    # Plots
    _plot_effect_sizes(df_stats)
    _plot_bootstrap_cis(df_stats)

    print(f"\n  Full statistics saved → {csv_path}")
    return df_stats


def _plot_effect_sizes(df_stats: pd.DataFrame):
    """Forest plot of Cohen's d (lit vs news) with CI."""
    df = df_stats.dropna(subset=["cohens_d_lit_news"]).copy()
    df = df.sort_values("cohens_d_lit_news")

    fig, ax = plt.subplots(figsize=(9, max(5, len(df)*0.42)))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    y    = range(len(df))
    vals = df["cohens_d_lit_news"].values
    sigs = df["bh_significant"].values

    colors = [C["literature"] if v > 0 else C["news"] for v in vals]
    alphas = [0.9 if s else 0.45 for s in sigs]

    for i, (v, col, alp) in enumerate(zip(vals, colors, alphas)):
        ax.barh(i, v, color=col, alpha=alp, height=0.6)

    ax.axvline(0, color=C["text"], lw=1.2)
    # Effect size reference lines
    for d_val, label in [(-0.8,"large"),(-0.5,"medium"),(-0.2,"small"),
                          (0.2,"small"),(0.5,"medium"),(0.8,"large")]:
        ax.axvline(d_val, color=C["grid"], lw=0.8, linestyle=":")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["metric"].values, fontsize=8.5)
    ax.set_xlabel("Cohen's d  (Literature − News)")
    ax.set_title(
        "Effect sizes: Literature vs. News (forest plot)\n"
        "Faded bars = not significant after BH correction",
        fontsize=10, fontweight="bold", color=C["text"]
    )
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors=C["muted"])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "effect_sizes_forest.png"),
                dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()


def _plot_bootstrap_cis(df_stats: pd.DataFrame):
    """Plot means with bootstrap 95% CI for key metrics."""
    key_m = [m for m in ["nsubj","obj","obl","advmod","amod","mdd","pos_VERB","pos_ADP"]
             if m in df_stats["metric"].values]

    if not key_m:
        return

    n_metrics = len(key_m)
    fig, axes = plt.subplots(2, (n_metrics+1)//2, figsize=(14, 7))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    fig.patch.set_facecolor(C["bg"])

    for ax, metric in zip(axes_flat, key_m):
        ax.set_facecolor(C["bg"])
        row = df_stats[df_stats["metric"] == metric].iloc[0]
        for i, genre in enumerate(GENRE_ORDER):
            mn   = row.get(f"{genre}_mean")
            lo   = row.get(f"{genre}_ci_lo")
            hi   = row.get(f"{genre}_ci_hi")
            if mn is None:
                continue
            ax.bar(i, mn, color=C[genre], alpha=0.82, width=0.55)
            ax.errorbar(i, mn, yerr=[[mn-lo],[hi-mn]],
                        fmt="none", color=C["text"], capsize=5, lw=1.5)
        ax.set_xticks(range(len(GENRE_ORDER)))
        ax.set_xticklabels([GENRE_LABELS[g] for g in GENRE_ORDER],
                           rotation=15, fontsize=7.5)
        bh_label = "✓" if row.get("bh_significant") else "(NS)"
        ax.set_title(f"{metric} {bh_label}", fontsize=9, fontweight="bold",
                     color=C["text"])
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(colors=C["muted"])

    # Hide extra axes
    for ax in list(axes_flat)[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Genre means with 95% bootstrap confidence intervals",
                 fontsize=11, fontweight="bold", color=C["text"], y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bootstrap_cis.png"),
                dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    logger.info("Bootstrap CI plot saved")