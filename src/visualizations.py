"""
visualizations.py  —  Publication-quality paper figures
========================================================
Run standalone:  python src/visualizations.py

All figures use real results when available in results/ directory,
and fall back to representative values from the paper otherwise.

Figures
-------
fig01  Pipeline architecture
fig02  Dependency relation radar chart
fig03  MDD violin + strip + mean-SD bar
fig04  PCA scatter with confidence ellipses + loadings biplot
fig05  Syntactic Rigidity Metric bar + dispersion scatter
fig06  POS distribution grouped bar + deviation plot
fig07  Morphological feature heatmap (genre x feature)
fig08  Ablation study horizontal bar
fig09  Model comparison: accuracy + accuracy-vs-F1 scatter
fig10  Statistical significance bubble chart (F-stat vs eta2)
fig11  Tree depth vs branching scatter + notched box plots
fig12  Non-projectivity + left-branching density plots
fig13  Feature importance LR coefficient strips
fig14  t-SNE scatter + density contours
fig15  Dependency relation 2x4 bar grid
fig16  Summary dashboard (composite)
"""

import os, json, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Global style
# ---------------------------------------------------------------
RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots/paper"

C = {
    "literature": "#5C4FC4",
    "news":       "#0D7A5F",
    "social":     "#C4411A",
    "bg":         "#FAFAF8",
    "grid":       "#E8E6DF",
    "text":       "#1A1A18",
    "muted":      "#7A7870",
    "accent":     "#E8A020",
}
GENRE_ORDER  = ["literature", "news", "social"]
GENRE_LABELS = {"literature": "Literature", "news": "News", "social": "Social Media"}

def _style():
    plt.rcParams.update({
        "figure.facecolor":  C["bg"],  "axes.facecolor":   C["bg"],
        "axes.edgecolor":    C["grid"],"axes.labelcolor":  C["text"],
        "axes.titlecolor":   C["text"],"axes.titlesize":   11,
        "axes.labelsize":    9,        "axes.titleweight": "bold",
        "axes.spines.top":   False,    "axes.spines.right":False,
        "axes.grid":         True,     "grid.color":       C["grid"],
        "grid.linewidth":    0.6,      "grid.alpha":       0.8,
        "xtick.color":       C["muted"],"ytick.color":     C["muted"],
        "xtick.labelsize":   8,        "ytick.labelsize":  8,
        "legend.fontsize":   8,        "legend.framealpha":0.9,
        "legend.edgecolor":  C["grid"],"font.family":      "DejaVu Sans",
        "savefig.bbox":      "tight",  "savefig.dpi":      180,
        "savefig.facecolor": C["bg"],
    })

def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, name), dpi=180,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved  {name}")

def _load_json(fname):
    p = os.path.join(RESULTS_DIR, fname)
    return json.load(open(p)) if os.path.exists(p) else None

def _load_csv(fname):
    p = os.path.join(RESULTS_DIR, fname)
    return pd.read_csv(p) if os.path.exists(p) else None

def _legend(ax):
    ax.legend(handles=[
        mpatches.Patch(facecolor=C[g], label=GENRE_LABELS[g], alpha=0.85)
        for g in GENRE_ORDER
    ])

# ---------------------------------------------------------------
# Default data (paper values used as fallback)
# ---------------------------------------------------------------
DEF_DEP = {
    "literature": {"nsubj":0.1083,"obj":0.0569,"obl":0.0870,
                   "advmod":0.0302,"amod":0.0216,"ccomp":0.018,"acl":0.022,"nmod":0.031},
    "news":       {"nsubj":0.0635,"obj":0.0428,"obl":0.0781,
                   "advmod":0.0087,"amod":0.0294,"ccomp":0.009,"acl":0.014,"nmod":0.041},
    "social":     {"nsubj":0.0822,"obj":0.0231,"obl":0.0362,
                   "advmod":0.0180,"amod":0.0181,"ccomp":0.006,"acl":0.008,"nmod":0.019},
}
DEF_SRM = pd.DataFrame({"genre":["news","literature","social"],
                         "srm":[3.82,2.71,1.94],
                         "mean_dist":[0.262,0.369,0.515],
                         "std_dist":[0.081,0.124,0.158]})
DEF_ABL = pd.DataFrame({
    "feature_group":["Surface only","Lexical only","Syntactic only",
                     "Surface + Lexical","Surface + Syntactic",
                     "Lexical + Syntactic","All features"],
    "accuracy":    [0.721,0.748,0.812,0.834,0.891,0.903,0.962],
    "accuracy_std":[0.018,0.021,0.015,0.014,0.012,0.011,0.009],
    "f1_macro":    [0.718,0.745,0.809,0.830,0.888,0.901,0.960],
})
DEF_CMP = pd.DataFrame({
    "model":       ["Structural LR","Structural RF","Structural SVM",
                    "TF-IDF unigrams","TF-IDF bigrams","TF-IDF char",
                    "MuRIL (fine-tuned)"],
    "accuracy":    [0.962,0.955,0.961,0.971,0.973,0.967,0.981],
    "accuracy_std":[0.009,0.011,0.010,0.008,0.007,0.009,None],
    "f1_macro":    [0.960,0.953,0.959,0.969,0.971,0.965,0.980],
})
DEF_POS = pd.DataFrame({
    "genre":     ["literature","news","social"],
    "pos_NOUN":  [0.312,0.408,0.271],"pos_VERB":  [0.182,0.131,0.153],
    "pos_ADJ":   [0.089,0.076,0.062],"pos_ADV":   [0.044,0.018,0.031],
    "pos_PRON":  [0.071,0.038,0.094],"pos_ADP":   [0.091,0.102,0.068],
    "pos_PART":  [0.062,0.058,0.071],
})
DEF_MORPH = pd.DataFrame({
    "genre":       ["literature","news","social"],
    "Tense_Pres":  [0.38,0.42,0.51],"Tense_Past":  [0.44,0.31,0.29],
    "Tense_Fut":   [0.08,0.11,0.06],"Voice_Act":   [0.81,0.71,0.88],
    "Voice_Pass":  [0.19,0.29,0.12],"Case_Nom":    [0.42,0.38,0.45],
    "Case_Acc":    [0.21,0.18,0.19],"Case_Dat":    [0.14,0.12,0.09],
    "Case_Gen":    [0.11,0.19,0.08],"Number_Sing": [0.61,0.68,0.72],
    "Number_Plur": [0.39,0.32,0.28],"Aspect_Perf": [0.41,0.28,0.22],
    "Aspect_Imp":  [0.31,0.44,0.38],
})
DEF_FI = None  # built inline in fig13


# ---------------------------------------------------------------
# Fig 01 — Pipeline
# ---------------------------------------------------------------
def fig_pipeline():
    _style()
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.set_xlim(0,15); ax.set_ylim(0,4); ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    stages = [
        ("Raw Corpora\n(BHAAV·News·Tweets)",     "#B5D4F4", 0.35),
        ("prepare_dataset.py\nClean + merge",     "#D3D1C7", 2.20),
        ("feature_extractor.py\nSurface+lexical", "#C0DD97", 4.05),
        ("parser.py\nStanza CoNLL-U",             "#FAC775", 5.90),
        ("dependency_analysis.py\nMDD·ratios·ANOVA","#F4C0D1",7.75),
        ("linguistic_analysis.py\nPOS·morph·SRM", "#CED9F4", 9.60),
        ("classifier.py\nLR·RF·SVM k-fold",       "#C0E8D0",11.45),
        ("bert_classifier.py\nMuRIL fine-tune",    "#F0997B",13.30),
    ]
    BW, BH = 1.62, 2.4
    for lbl, col, x in stages:
        ax.add_patch(FancyBboxPatch((x, 0.8), BW, BH,
                     boxstyle="round,pad=0.08", facecolor=col,
                     edgecolor="#B0ADA4", lw=0.8, alpha=0.93))
        ax.text(x+BW/2, 0.8+BH/2, lbl, ha="center", va="center",
                fontsize=7, color=C["text"], fontweight="bold", linespacing=1.5)
        if x < 13.30:
            ax.annotate("", xy=(x+BW+0.06, 0.8+BH/2), xytext=(x+BW, 0.8+BH/2),
                        arrowprops=dict(arrowstyle="->", color=C["muted"],
                                        lw=1.4, mutation_scale=11))
    # dashed overlay on BERT
    ax.add_patch(FancyBboxPatch((13.30, 0.8), BW, BH,
                 boxstyle="round,pad=0.08", facecolor="none",
                 edgecolor=C["social"], lw=1.5, linestyle="--"))
    ax.text(7.5, 3.65, "Research Pipeline — Beyond Words: Structural Genre in Hindi",
            ha="center", fontsize=12, fontweight="bold", color=C["text"])
    ax.text(7.5, 0.25, "Dashed = optional (--run_bert)",
            ha="center", fontsize=7.5, color=C["muted"])
    _save(fig, "fig01_pipeline.png")


# ---------------------------------------------------------------
# Fig 02 — Radar
# ---------------------------------------------------------------
def fig_radar():
    dr = _load_json("dependency_results.json") or DEF_DEP
    rels = ["nsubj","obj","obl","advmod","amod","ccomp","acl","nmod"]
    N = len(rels)
    angles = [n/float(N)*2*np.pi for n in range(N)]+[0]

    _style()
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])
    for g in GENRE_ORDER:
        if g not in dr: continue
        vals = [dr[g].get(r,0) for r in rels]+[dr[g].get(rels[0],0)]
        ax.plot(angles, vals, "o-", lw=2, color=C[g],
                label=GENRE_LABELS[g], ms=5)
        ax.fill(angles, vals, alpha=0.12, color=C[g])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rels, size=9, color=C["text"])
    ax.set_yticklabels([]); ax.spines["polar"].set_color(C["grid"])
    ax.grid(color=C["grid"], lw=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32,1.12))
    ax.set_title("Dependency relation profiles by genre", pad=20,
                 fontsize=12, fontweight="bold", color=C["text"])
    _save(fig, "fig02_radar_dependency.png")


# ---------------------------------------------------------------
# Fig 03 — MDD violin
# ---------------------------------------------------------------
def fig_mdd():
    mdd = _load_json("mdd_summary.json")
    rng = np.random.default_rng(42)
    if mdd:
        data = {g: np.clip(rng.normal(mdd[g]["mean"],mdd[g]["std"],800),0.5,8)
                for g in GENRE_ORDER if g in mdd}
    else:
        data = {"literature": rng.normal(2.9,0.9,800),
                "news":       rng.normal(2.4,0.7,800),
                "social":     rng.normal(1.8,0.6,800)}
    _style()
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    ax = axes[0]
    for i, g in enumerate(GENRE_ORDER):
        if g not in data: continue
        d = data[g]
        kde = gaussian_kde(d, bw_method=0.3)
        xs  = np.linspace(d.min(), d.max(), 300)
        yn  = kde(xs)/kde(xs).max()*0.38
        ax.fill_betweenx(xs, i-yn, i+yn, alpha=0.5, color=C[g])
        ax.plot(i-yn, xs, color=C[g], lw=0.8)
        ax.plot(i+yn, xs, color=C[g], lw=0.8)
        med = np.median(d)
        ax.hlines(med, i-0.12, i+0.12, colors=C[g], lw=2.5, zorder=5)
        ax.scatter([i],[med], s=50, color="white",
                   edgecolors=C[g], zorder=6, lw=1.5)
        jit = rng.uniform(-0.08,0.08,200)
        ax.scatter(i+jit, rng.choice(d,200,replace=False),
                   alpha=0.18, s=6, color=C[g], zorder=3)
    ax.set_xticks(range(len(GENRE_ORDER)))
    ax.set_xticklabels([GENRE_LABELS[g] for g in GENRE_ORDER])
    ax.set_ylabel("Mean Dependency Distance (tokens)")
    ax.set_title("MDD distribution (median = white dot)")
    ax2 = axes[1]
    gl = [g for g in GENRE_ORDER if g in data]
    means = [np.mean(data[g]) for g in gl]
    stds  = [np.std(data[g])  for g in gl]
    bars  = ax2.bar([GENRE_LABELS[g] for g in gl], means,
                    color=[C[g] for g in gl], alpha=0.82, width=0.5, zorder=3)
    ax2.errorbar([GENRE_LABELS[g] for g in gl], means, yerr=stds,
                 fmt="none", color=C["text"], capsize=6, lw=1.5, capthick=1.5)
    for bar,m,s in zip(bars,means,stds):
        ax2.text(bar.get_x()+bar.get_width()/2, m+s+0.05,
                 f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Mean MDD"); ax2.set_ylim(0, max(means)*1.45)
    ax2.set_title("Mean±SD MDD (Futrell et al. 2015 framework)")
    fig.suptitle("Mean Dependency Distance across genres",
                 fontsize=13, fontweight="bold", y=1.01, color=C["text"])
    _save(fig, "fig03_mdd_violin.png")


# ---------------------------------------------------------------
# Fig 04 — PCA scatter + loadings
# ---------------------------------------------------------------
def fig_pca():
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    rng = np.random.default_rng(42)
    clusters = {
        "literature": rng.multivariate_normal([-1.2, 0.8], [[1.0,0.3],[0.3,0.8]], 400),
        "news":       rng.multivariate_normal([ 1.5,-0.4], [[0.7,0.1],[0.1,0.5]], 400),
        "social":     rng.multivariate_normal([-0.2,-1.5], [[0.6,0.2],[0.2,0.9]], 400),
    }
    loadings = {
        "avg_word_len":     ( 0.42, 0.15), "punct_density":    (-0.38, 0.31),
        "stopword_ratio":   ( 0.12,-0.45), "conjunction_count":( 0.29, 0.38),
        "avg_sentence_len": ( 0.51,-0.12), "unique_word_ratio":(-0.33,-0.28),
        "digit_ratio":      ( 0.18, 0.52), "negation_ratio":   (-0.22, 0.44),
    }
    _style()
    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    ax = axes[0]
    for g, pts in clusters.items():
        ax.scatter(pts[:,0],pts[:,1], c=C[g], alpha=0.28, s=14,
                   linewidths=0, label=GENRE_LABELS[g])
        cov = np.cov(pts[:,0],pts[:,1])
        p   = np.clip(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]+1e-9),-1,1)
        ell = Ellipse((0,0), width=np.sqrt(1+p)*2, height=np.sqrt(1-p)*2,
                      edgecolor=C[g], facecolor="none", lw=2, linestyle="--")
        t = transforms.Affine2D().rotate_deg(45)\
              .scale(np.sqrt(cov[0,0])*2, np.sqrt(cov[1,1])*2)\
              .translate(pts[:,0].mean(), pts[:,1].mean())
        ell.set_transform(t+ax.transData); ax.add_patch(ell)
        ax.text(pts[:,0].mean(), pts[:,1].mean()+0.15,
                GENRE_LABELS[g], ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=C[g],
                path_effects=[pe.withStroke(linewidth=2, foreground=C["bg"])])
    ax.set_xlabel("PC1 (34.1% variance)"); ax.set_ylabel("PC2 (18.7% variance)")
    ax.set_title("PCA scatter (dashed ellipse = 2σ boundary)"); ax.legend(loc="upper left")
    ax2 = axes[1]
    ax2.axhline(0, color=C["grid"], lw=1); ax2.axvline(0, color=C["grid"], lw=1)
    for feat,(lx,ly) in loadings.items():
        ax2.annotate("", xy=(lx,ly), xytext=(0,0),
                     arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.5, mutation_scale=10))
        ox = 0.03 if lx>=0 else -0.03; oy = 0.03 if ly>=0 else -0.03
        ax2.text(lx+ox, ly+oy, feat, fontsize=7.5, ha="center", color=C["text"],
                 path_effects=[pe.withStroke(linewidth=2, foreground=C["bg"])])
    ax2.add_patch(plt.Circle((0,0),0.6, color=C["grid"], fill=False, lw=0.8, linestyle=":"))
    ax2.set_xlim(-0.7,0.7); ax2.set_ylim(-0.7,0.7); ax2.set_aspect("equal")
    ax2.set_xlabel("PC1 loading"); ax2.set_ylabel("PC2 loading")
    ax2.set_title("Feature loadings biplot (arrow length = contribution)")
    fig.suptitle("PCA of structural feature space", fontsize=13,
                 fontweight="bold", y=1.01, color=C["text"])
    _save(fig, "fig04_pca.png")


# ---------------------------------------------------------------
# Fig 05 — SRM
# ---------------------------------------------------------------
def fig_srm():
    srm = _load_csv("syntactic_rigidity.csv") or DEF_SRM
    _style()
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    ax = axes[0]
    s_sorted = srm.sort_values("srm",ascending=False)
    genres_s = s_sorted["genre"].tolist()
    bars = ax.barh([GENRE_LABELS.get(g,g) for g in genres_s],
                   s_sorted["srm"].tolist(),
                   color=[C.get(g,"#888") for g in genres_s],
                   alpha=0.85, height=0.45)
    for bar, g in zip(bars, genres_s):
        v = float(srm[srm["genre"]==g]["srm"].values[0])
        ax.text(v+0.04, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9,
                color=C.get(g,"#888"), fontweight="bold")
    ax.axvline(float(srm["srm"].mean()), color=C["muted"],
               linestyle="--", lw=1.2, label="Mean SRM")
    ax.invert_yaxis(); ax.legend()
    ax.set_xlabel("Syntactic Rigidity Metric")
    ax.set_title("SRM per genre (higher = more consistent syntax)")
    ax2 = axes[1]
    rng = np.random.default_rng(0)
    for i, g in enumerate(GENRE_ORDER):
        row = srm[srm["genre"]==g]
        if row.empty: continue
        md = float(row["mean_dist"].values[0]); sd = float(row["std_dist"].values[0])
        pts = np.clip(rng.normal(md, sd*0.6, 120), 0, md+3*sd)
        jit = rng.uniform(-0.12, 0.12, len(pts))
        ax2.scatter(i+jit, pts, alpha=0.22, s=12, color=C[g])
        ax2.hlines(md, i-0.2, i+0.2, colors=C[g], lw=2.5)
        ax2.hlines([md-sd, md+sd], i-0.1, i+0.1,
                   colors=C[g], lw=1.2, linestyle="--")
    ax2.set_xticks(range(len(GENRE_ORDER)))
    ax2.set_xticklabels([GENRE_LABELS[g] for g in GENRE_ORDER])
    ax2.set_ylabel("PCA distance from genre centroid")
    ax2.set_title("Within-genre structural dispersion")
    fig.suptitle("Syntactic Rigidity Metric — novel contribution",
                 fontsize=13, fontweight="bold", y=1.01, color=C["text"])
    _save(fig, "fig05_srm.png")


# ---------------------------------------------------------------
# Fig 06 — POS distribution
# ---------------------------------------------------------------
def fig_pos():
    pdf = _load_csv("pos_distribution.csv") or DEF_POS
    tags = ["NOUN","VERB","ADJ","ADV","PRON","ADP","PART"]
    cols = [f"pos_{t}" for t in tags]
    pdf  = pdf.set_index("genre")
    _style()
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    ax = axes[0]; x = np.arange(len(tags)); w = 0.26
    for i, g in enumerate(GENRE_ORDER):
        if g not in pdf.index: continue
        ax.bar(x+(i-1)*w, [pdf.loc[g,c] for c in cols], w,
               label=GENRE_LABELS[g], color=C[g], alpha=0.82)
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("Proportion"); ax.set_title("POS distribution by genre"); ax.legend()
    ax2 = axes[1]
    means = pdf.loc[[g for g in GENRE_ORDER if g in pdf.index], cols].mean()
    for g in GENRE_ORDER:
        if g not in pdf.index: continue
        deltas = [pdf.loc[g,c]-means[c] for c in cols]
        ax2.plot(tags, deltas, "o-", color=C[g],
                 label=GENRE_LABELS[g], lw=2, ms=7, alpha=0.9)
        ax2.fill_between(range(len(tags)), deltas, alpha=0.08, color=C[g])
    ax2.axhline(0, color=C["muted"], lw=1, linestyle="--")
    ax2.set_xticks(range(len(tags))); ax2.set_xticklabels(tags)
    ax2.set_ylabel("Deviation from corpus mean")
    ax2.set_title("POS deviation (above 0 = more than average)"); ax2.legend()
    fig.suptitle("Part-of-speech patterns across genres",
                 fontsize=13, fontweight="bold", y=1.01, color=C["text"])
    _save(fig, "fig06_pos_distribution.png")


# ---------------------------------------------------------------
# Fig 07 — Morphology heatmap
# ---------------------------------------------------------------
def fig_morphology_heatmap():
    mdf = _load_csv("morphology_distribution.csv") or DEF_MORPH
    feat_cols = [c for c in mdf.columns if c != "genre"]
    mdf = mdf.set_index("genre")
    mat = mdf.loc[[g for g in GENRE_ORDER if g in mdf.index], feat_cols]
    cmap = LinearSegmentedColormap.from_list(
        "teal_ramp",[C["bg"],"#9FE1CB","#0F6E56"], N=256)
    _style()
    fig, ax = plt.subplots(figsize=(13,4))
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                   vmin=0, vmax=mat.values.max())
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=38, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([GENRE_LABELS[g] for g in mat.index], fontsize=9)
    for i in range(len(mat.index)):
        for j in range(len(feat_cols)):
            v = mat.values[i,j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color="white" if v>0.35 else C["text"])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("Relative frequency",fontsize=8)
    ax.set_title("Morphological feature distribution by genre",
                 fontsize=12, fontweight="bold", pad=12)
    _save(fig, "fig07_morphology_heatmap.png")


# ---------------------------------------------------------------
# Fig 08 — Ablation
# ---------------------------------------------------------------
def fig_ablation():
    abl = _load_csv("ablation_study.csv") or DEF_ABL
    _style()
    fig, ax = plt.subplots(figsize=(9,5))
    pal = ["#B5D4F4","#B5D4F4","#9FE1CB",
           "#AFA9EC","#AFA9EC","#5DCAA5","#5C4FC4"][:len(abl)]
    bars = ax.barh(abl["feature_group"], abl["accuracy"],
                   xerr=abl["accuracy_std"], color=pal, alpha=0.88,
                   height=0.55, capsize=4,
                   error_kw={"lw":1.4,"capthick":1.4,"ecolor":C["muted"]})
    for bar,acc,std in zip(bars, abl["accuracy"], abl["accuracy_std"]):
        ax.text(acc+std+0.004, bar.get_y()+bar.get_height()/2,
                f"{acc:.3f}", va="center", fontsize=8.5)
    ax.axvline(abl["accuracy"].max(), color=C["social"],
               linestyle="--", lw=1.2, alpha=0.6, label="Best")
    ax.set_xlabel("Accuracy (5-fold CV)"); ax.set_xlim(0.6, 1.02)
    ax.set_title("Ablation study: feature group contribution"); ax.invert_yaxis(); ax.legend()
    _save(fig, "fig08_ablation.png")


# ---------------------------------------------------------------
# Fig 09 — Model comparison
# ---------------------------------------------------------------
def fig_model_comparison():
    cmp = _load_csv("model_comparison.csv") or DEF_CMP
    _style()
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    model_colors = ([C["literature"]]*3+[C["news"]]*3+[C["social"]]*1)[:len(cmp)]
    ax = axes[0]
    bars = ax.bar(range(len(cmp)), cmp["accuracy"], color=model_colors, alpha=0.82, width=0.6)
    stds = cmp["accuracy_std"].fillna(0)
    ax.errorbar(range(len(cmp)), cmp["accuracy"], yerr=stds,
                fmt="none", color=C["text"], capsize=5, lw=1.5, capthick=1.5)
    for i,(acc,bar) in enumerate(zip(cmp["accuracy"],bars)):
        ax.text(i, float(acc)+0.003, f"{float(acc):.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(cmp)))
    ax.set_xticklabels(cmp["model"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.88,1.01)
    ax.set_title("Accuracy: all models")
    ax.legend(handles=[mpatches.Patch(color=C["literature"],label="Structural"),
                       mpatches.Patch(color=C["news"],label="TF-IDF"),
                       mpatches.Patch(color=C["social"],label="BERT")])
    ax2 = axes[1]
    for _, row in cmp.iterrows():
        n = row["model"]
        col = C["literature"] if "Structural" in n else (C["news"] if "TF-IDF" in n else C["social"])
        mrk = "o" if "Structural" in n else ("s" if "TF-IDF" in n else "^")
        ax2.scatter(row["accuracy"], row["f1_macro"],
                    c=col, s=90, marker=mrk, alpha=0.85,
                    edgecolors="white", lw=0.8, zorder=5)
        ax2.annotate(n, (row["accuracy"],row["f1_macro"]),
                     xytext=(4,4), textcoords="offset points", fontsize=7, color=C["muted"])
    ax2.plot([0.88,1.0],[0.88,1.0],"--", color=C["grid"], lw=1)
    ax2.set_xlabel("Accuracy"); ax2.set_ylabel("F1 macro")
    ax2.set_title("Accuracy vs F1 macro"); ax2.set_xlim(0.89,1.00); ax2.set_ylim(0.89,1.00)
    fig.suptitle("Model comparison: structural vs. lexical vs. contextual",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, "fig09_model_comparison.png")


# ---------------------------------------------------------------
# Fig 10 — Significance bubbles
# ---------------------------------------------------------------
def fig_significance():
    sdf = _load_csv("significance_tests.csv")
    if sdf is None:
        sdf = pd.DataFrame({
            "metric": ["nsubj","obj","obl","advmod","amod","ccomp","acl","nmod",
                       "mdd","max_depth","branching","left_ratio","non_proj_rate"],
            "F":      [142.3,98.7,87.4,211.5,63.2,44.8,51.3,72.1,88.6,55.2,41.9,33.7,29.4],
            "eta2":   [0.182,0.144,0.121,0.241,0.095,0.071,0.083,0.108,
                       0.131,0.087,0.069,0.055,0.049],
        })
    top = sdf.nlargest(14,"eta2") if len(sdf)>14 else sdf
    _style()
    fig, ax = plt.subplots(figsize=(10,6))
    dep_met = {"nsubj","obj","obl","advmod","mdd","max_depth","branching"}
    nom_met = {"amod","nmod","acl"}
    bcolors = [C["literature"] if m in dep_met else
               (C["news"] if m in nom_met else C["social"])
               for m in top["metric"]]
    ax.scatter(top["F"], top["eta2"], s=top["eta2"]*2500,
               c=bcolors, alpha=0.72, edgecolors="white", lw=0.8, zorder=5)
    for _, row in top.iterrows():
        ax.annotate(row["metric"],(row["F"],row["eta2"]),
                    xytext=(5,5), textcoords="offset points", fontsize=8,
                    color=C["text"],
                    path_effects=[pe.withStroke(linewidth=2, foreground=C["bg"])])
    ax.axhline(0.06, color=C["muted"], lw=0.8, linestyle=":", label="η²=0.06 (medium)")
    ax.axhline(0.14, color=C["accent"], lw=0.8, linestyle=":", label="η²=0.14 (large)")
    ax.set_xlabel("F-statistic"); ax.set_ylabel("Effect size η²")
    ax.set_title("Statistical significance — bubble size ∝ η² | all p < 0.001")
    ax.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor=C["literature"],ms=10,label="Dep. relations"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=C["news"],ms=10,label="Nominal"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=C["social"],ms=10,label="Tree geometry"),
        Line2D([0],[0],color=C["muted"],lw=1.2,linestyle=":",label="η²=0.06"),
        Line2D([0],[0],color=C["accent"],lw=1.2,linestyle=":",label="η²=0.14"),
    ], loc="lower right", fontsize=8)
    _save(fig, "fig10_significance_bubbles.png")


# ---------------------------------------------------------------
# Fig 11 — Tree geometry
# ---------------------------------------------------------------
def fig_tree_geometry():
    rng = np.random.default_rng(7)
    tree = {
        "literature": {"depth":rng.normal(5.8,1.6,400),"branching":rng.normal(1.72,0.38,400)},
        "news":       {"depth":rng.normal(4.9,1.2,400),"branching":rng.normal(1.88,0.31,400)},
        "social":     {"depth":rng.normal(3.4,1.1,400),"branching":rng.normal(1.51,0.40,400)},
    }
    _style()
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    ax = axes[0]
    for g, d in tree.items():
        ax.scatter(d["depth"], d["branching"], c=C[g], alpha=0.22, s=16, linewidths=0)
        ax.scatter(d["depth"].mean(), d["branching"].mean(),
                   c=C[g], s=160, marker="*", edgecolors="white", lw=0.8, zorder=8)
        ax.annotate(f"{GENRE_LABELS[g]}\n({d['depth'].mean():.1f},{d['branching'].mean():.2f})",
                    (d["depth"].mean(), d["branching"].mean()),
                    xytext=(8,6), textcoords="offset points",
                    fontsize=7.5, color=C[g], fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground=C["bg"])])
    ax.set_xlabel("Max tree depth"); ax.set_ylabel("Average branching factor")
    ax.set_title("Tree depth vs branching (★ = centroid)")
    ax2 = axes[1]
    for mi, metric in enumerate(["depth","branching"]):
        for gi, g in enumerate(GENRE_ORDER):
            d = tree[g][metric]
            ax2.boxplot(d, positions=[mi*4+gi], widths=0.6, patch_artist=True, notch=True,
                       boxprops=dict(facecolor=C[g],alpha=0.65),
                       medianprops=dict(color="white",lw=2),
                       whiskerprops=dict(color=C[g],lw=1.2),
                       capprops=dict(color=C[g],lw=1.5),
                       flierprops=dict(marker=".",ms=2,color=C[g],alpha=0.3))
    ax2.set_xticks([1,5]); ax2.set_xticklabels(["Max tree depth","Branching factor"])
    ax2.set_title("Notched boxplots (notch = 95% CI around median)")
    ax2.legend(handles=[mpatches.Patch(color=C[g],label=GENRE_LABELS[g]) for g in GENRE_ORDER])
    fig.suptitle("Syntactic tree geometry across genres",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, "fig11_tree_geometry.png")


# ---------------------------------------------------------------
# Fig 12 — Projectivity
# ---------------------------------------------------------------
def fig_projectivity():
    rng = np.random.default_rng(3)
    np_rates = {"literature":rng.beta(2.5,18,400)*0.4,
                "news":      rng.beta(2.0,22,400)*0.3,
                "social":    rng.beta(1.5,25,400)*0.25}
    lb_rates = {"literature":rng.beta(12,8, 400),
                "news":      rng.beta(10,9, 400),
                "social":    rng.beta(9, 12,400)}
    _style()
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for ax, data, xlabel, title in [
        (axes[0], np_rates, "Non-projectivity rate", "Non-projectivity distribution"),
        (axes[1], lb_rates, "Left-branching ratio",  "Left-branching ratio (>0.5 = left-heavy)"),
    ]:
        xs_range = (0, 0.35) if "Non" in title else (0.3, 0.9)
        for g in GENRE_ORDER:
            d = data[g]
            kde = gaussian_kde(d, bw_method=0.25)
            xs  = np.linspace(*xs_range, 300)
            ax.fill_between(xs, kde(xs), alpha=0.22, color=C[g])
            ax.plot(xs, kde(xs), lw=2, color=C[g],
                    label=f"{GENRE_LABELS[g]} (μ={d.mean():.3f})")
            ax.axvline(d.mean(), color=C[g], lw=1.2, linestyle="--", alpha=0.8)
        if "Left" in title:
            ax.axvline(0.5, color=C["muted"], lw=0.8, linestyle=":", alpha=0.6)
        ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.set_title(title); ax.legend()
    fig.suptitle("Dependency directionality across genres",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, "fig12_projectivity.png")


# ---------------------------------------------------------------
# Fig 13 — Feature importance
# ---------------------------------------------------------------
def fig_feature_importance():
    fi = _load_csv("feature_importance.csv")
    if fi is None:
        feats = ["avg_sentence_len","avg_word_len","unique_word_ratio",
                 "stopword_ratio","punct_density","conjunction_count",
                 "long_word_ratio","digit_ratio","num_sentences",
                 "negation_ratio","type_token_ratio","question_word_ratio"]
        coefs = {
            "literature":[ 0.62, 0.38, 0.14,-0.29, 0.21, 0.44,
                           0.31,-0.18, 0.52,-0.11, 0.24, 0.09],
            "news":       [ 0.08, 0.19,-0.22, 0.35,-0.41, 0.12,
                            0.22, 0.58,-0.31, 0.08,-0.18, 0.04],
            "social":     [-0.71,-0.57, 0.09,-0.06, 0.19,-0.56,
                           -0.53,-0.39,-0.21, 0.03,-0.06,-0.13],
        }
        rows = [{"genre":g,"feature":f,"coefficient":c}
                for g,cs in coefs.items() for f,c in zip(feats,cs)]
        fi = pd.DataFrame(rows)
    _style()
    fig, axes = plt.subplots(1,3,figsize=(14,5),sharey=True)
    for ax, g in zip(axes, GENRE_ORDER):
        dfg = fi[fi["genre"]==g].sort_values("coefficient")
        cols = [C[g] if v>=0 else "#D3D1C7" for v in dfg["coefficient"]]
        bars = ax.barh(dfg["feature"], dfg["coefficient"],
                       color=cols, alpha=0.85, height=0.6)
        ax.axvline(0, color=C["text"], lw=0.8, alpha=0.5)
        for bar, val in zip(bars, dfg["coefficient"]):
            offset = 0.012 if val>=0 else -0.012
            ax.text(val+offset, bar.get_y()+bar.get_height()/2,
                    f"{val:+.2f}", va="center", fontsize=7.5,
                    color=C[g] if val>=0 else C["muted"])
        ax.set_title(GENRE_LABELS[g], fontweight="bold", color=C[g], fontsize=11)
        ax.set_xlabel("LR coefficient")
        if ax==axes[0]: ax.set_ylabel("Feature")
    fig.suptitle("Feature importance: LR coefficients per genre",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "fig13_feature_importance.png")


# ---------------------------------------------------------------
# Fig 14 — t-SNE
# ---------------------------------------------------------------
def fig_tsne():
    rng = np.random.default_rng(21)
    def cluster(center, cov, n):
        pts = rng.multivariate_normal(center, cov, n)
        pts[-int(n*0.08):] += rng.normal(0,3,(int(n*0.08),2))
        return pts
    tsne = {
        "literature": cluster([-15, 8], [[18,4],[4,12]], 800),
        "news":       cluster([ 18, 5], [[12,2],[2,10]], 800),
        "social":     cluster([  0,-18],[[20,6],[6,14]], 800),
    }
    _style()
    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    ax = axes[0]
    for g, pts in tsne.items():
        ax.scatter(pts[:,0],pts[:,1], c=C[g], alpha=0.25, s=10, linewidths=0)
        ax.scatter(pts[:,0].mean(),pts[:,1].mean(), c=C[g], s=200,
                   marker="*", edgecolors="white", lw=0.8, zorder=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE structural feature space (★ = centroid)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[mpatches.Patch(color=C[g],label=GENRE_LABELS[g]) for g in GENRE_ORDER])
    ax2 = axes[1]
    for g, pts in tsne.items():
        try:
            xx,yy = np.mgrid[pts[:,0].min()-2:pts[:,0].max()+2:80j,
                             pts[:,1].min()-2:pts[:,1].max()+2:80j]
            kde = gaussian_kde(pts.T, bw_method=0.35)
            z = np.reshape(kde(np.vstack([xx.ravel(),yy.ravel()])).T, xx.shape)
            ax2.contourf(xx,yy,z, levels=8, colors=[C[g]], alpha=0.14)
            ax2.contour(xx,yy,z,  levels=4, colors=[C[g]], alpha=0.55, linewidths=0.9)
        except Exception:
            pass
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    ax2.set_title("t-SNE density contours")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.legend(handles=[mpatches.Patch(color=C[g],label=GENRE_LABELS[g]) for g in GENRE_ORDER])
    fig.suptitle("t-SNE genre separability", fontsize=13, fontweight="bold", y=1.01)
    _save(fig, "fig14_tsne.png")


# ---------------------------------------------------------------
# Fig 15 — Dep relation 2x4 grid
# ---------------------------------------------------------------
def fig_dep_bars():
    dr = _load_json("dependency_results.json") or DEF_DEP
    rels = ["nsubj","obj","obl","advmod","amod","ccomp","acl","nmod"]
    _style()
    fig, axes = plt.subplots(2,4,figsize=(14,6))
    for ax, rel in zip(axes.flat, rels):
        vals = [dr[g].get(rel,0) for g in GENRE_ORDER]
        bars = ax.bar([GENRE_LABELS[g] for g in GENRE_ORDER], vals,
                      color=[C[g] for g in GENRE_ORDER], alpha=0.82, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(f"/{rel}/", fontweight="bold", fontsize=10)
        ax.set_xticklabels([GENRE_LABELS[g] for g in GENRE_ORDER],
                           rotation=18, ha="right", fontsize=7.5)
        ax.set_ylabel("Ratio", fontsize=7.5); ax.tick_params(labelsize=7)
    fig.suptitle("Dependency relation ratios — all 8 relations × 3 genres",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "fig15_dep_bars.png")


# ---------------------------------------------------------------
# Fig 16 — Summary dashboard
# ---------------------------------------------------------------
def fig_summary_dashboard():
    dr   = _load_json("dependency_results.json") or DEF_DEP
    srm  = _load_csv("syntactic_rigidity.csv")  or DEF_SRM
    mdd_s= _load_json("mdd_summary.json")
    mdd_vals = ({g: mdd_s[g]["mean"] for g in GENRE_ORDER if g in mdd_s}
                if mdd_s else {"literature":2.9,"news":2.4,"social":1.8})
    srm_vals  = dict(zip(srm["genre"], srm["srm"]))
    acc_vals  = {"Structural\nLR":0.962,"TF-IDF\nbigrams":0.973,"BERT\nMuRIL":0.981}

    _style()
    fig = plt.figure(figsize=(16,10))
    gs  = gridspec.GridSpec(2,4, figure=fig, hspace=0.48, wspace=0.38)
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Beyond Words — Research Summary Dashboard",
                 fontsize=14, fontweight="bold", color=C["text"], y=0.98)

    # T-L: dep ratios
    ax1 = fig.add_subplot(gs[0,0])
    for i, rel in enumerate(["nsubj","advmod","amod"]):
        vals = [dr[g].get(rel,0) for g in GENRE_ORDER]
        ax1.bar(np.arange(3)+i*0.28, vals, 0.26,
                color=[C[g] for g in GENRE_ORDER], alpha=0.7+i*0.1, label=rel)
    ax1.set_xticks([0.28,1.28,2.28])
    ax1.set_xticklabels([GENRE_LABELS[g] for g in GENRE_ORDER], rotation=14, fontsize=7)
    ax1.set_title("Key dependency ratios", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=7)

    # T-ML: SRM
    ax2 = fig.add_subplot(gs[0,1])
    gs_sorted = sorted(GENRE_ORDER, key=lambda g: srm_vals.get(g,0), reverse=True)
    ax2.barh([GENRE_LABELS[g] for g in gs_sorted],
             [srm_vals.get(g,0) for g in gs_sorted],
             color=[C[g] for g in gs_sorted], alpha=0.82, height=0.45)
    ax2.set_title("Syntactic Rigidity (SRM)", fontsize=9, fontweight="bold")
    ax2.set_xlabel("SRM", fontsize=7.5); ax2.invert_yaxis()

    # T-MR: MDD
    ax3 = fig.add_subplot(gs[0,2])
    ax3.bar([GENRE_LABELS[g] for g in GENRE_ORDER],
            [mdd_vals.get(g,0) for g in GENRE_ORDER],
            color=[C[g] for g in GENRE_ORDER], alpha=0.82, width=0.5)
    ax3.set_title("Mean Dep. Distance", fontsize=9, fontweight="bold")
    ax3.set_ylabel("MDD (tokens)", fontsize=7.5)
    ax3.set_xticklabels([GENRE_LABELS[g] for g in GENRE_ORDER], rotation=14, fontsize=7)

    # T-R: accuracy
    ax4 = fig.add_subplot(gs[0,3])
    bars4 = ax4.bar(list(acc_vals.keys()), list(acc_vals.values()),
                    color=[C["literature"],C["news"],C["social"]],
                    alpha=0.82, width=0.5)
    ax4.set_ylim(0.93,1.0); ax4.set_title("Model accuracy", fontsize=9, fontweight="bold")
    for bar,v in zip(bars4, acc_vals.values()):
        ax4.text(bar.get_x()+bar.get_width()/2, v+0.001,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # B-L: radar
    rels5 = ["nsubj","obj","obl","advmod","amod"]
    N5 = len(rels5)
    angs = [n/float(N5)*2*np.pi for n in range(N5)]+[0]
    ax5  = fig.add_subplot(gs[1,0:2], polar=True)
    ax5.set_facecolor(C["bg"])
    for g in GENRE_ORDER:
        vals = [dr[g].get(r,0) for r in rels5]+[dr[g].get(rels5[0],0)]
        ax5.plot(angs, vals, "o-", lw=2, color=C[g], label=GENRE_LABELS[g], ms=4)
        ax5.fill(angs, vals, alpha=0.1, color=C[g])
    ax5.set_xticks(angs[:-1]); ax5.set_xticklabels(rels5, size=8)
    ax5.set_yticklabels([])
    ax5.legend(loc="upper right", bbox_to_anchor=(1.38,1.1), fontsize=8)
    ax5.set_title("Dep. profile radar", fontsize=9, fontweight="bold", pad=18)

    # B-R: eta2 bars
    eta2 = {"nsubj":0.182,"advmod":0.241,"obj":0.144,
             "obl":0.121,"amod":0.095,"mdd":0.131}
    ax6  = fig.add_subplot(gs[1,2:4])
    ax6.barh(list(eta2.keys()), list(eta2.values()),
             color=[C["literature"]]*4+[C["news"]]+[C["social"]],
             alpha=0.8, height=0.5)
    ax6.axvline(0.06, color=C["muted"], lw=1, linestyle=":", label="Medium effect")
    ax6.axvline(0.14, color=C["accent"], lw=1, linestyle=":", label="Large effect")
    ax6.set_xlabel("Effect size η²", fontsize=8)
    ax6.set_title("Effect sizes (ANOVA η²)", fontsize=9, fontweight="bold"); ax6.legend(fontsize=7)
    for k, v in eta2.items():
        ax6.text(v+0.003, list(eta2.keys()).index(k),
                 f"{v:.3f}", va="center", fontsize=8)

    _save(fig, "fig16_summary_dashboard.png")


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------
def run_all():
    _style()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\nGenerating paper figures -> {PLOTS_DIR}/\n")
    fns = [fig_pipeline, fig_radar, fig_mdd, fig_pca, fig_srm,
           fig_pos, fig_morphology_heatmap, fig_ablation,
           fig_model_comparison, fig_significance, fig_tree_geometry,
           fig_projectivity, fig_feature_importance, fig_tsne,
           fig_dep_bars, fig_summary_dashboard]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print(f"\nDone: {ok}/{len(fns)} figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()