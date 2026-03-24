"""
build_dataset_final.py
======================
Definitive version. HuggingFace sources only. No local files needed.

Run:
    cd ~/beyond-words-genre
    python build_dataset_final.py

Sources (all confirmed working in previous runs)
-------------------------------------------------
LITERATURE  (~5000 docs, 1 source)
  wikipedia_hi        wikimedia/wikipedia "20231101.hi"   filtered literary

NEWS  (~5000 docs, 2 sources)
  wikipedia_hi_news   wikimedia/wikipedia "20231101.hi"   filtered political/govt
  xlsum_hindi         csebuetnlp/xlsum "hindi"            BBC news articles

SOCIAL  (~2200 docs, 3 sources)  ← max achievable without local sarcasm
  bbc_nli_passages    midas/bbc_hindi_nli                 sentences → passages
  indic_sentiment     ai4bharat/IndicSentiment            product reviews
  hindi_discourse     midas/hindi_discourse               dialogic/argumentative
"""

import os, re, sys, json, logging, hashlib, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH      = "data/processed/final_dataset_v2.csv"
STATS_PATH       = "data/processed/dataset_stats_v2.json"
MIN_WORDS        = 50
MIN_WORDS_SOCIAL = 30   # social passages are shorter
MAX_WORDS        = 400
TARGET_PER_GENRE = 5000


# ── utilities ─────────────────────────────────────────────────

def wc(t): return len(str(t).split())

def clean(t):
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("\x00", "")
    return re.sub(r"\s+", " ", t).strip()

def is_hindi(t, ratio=0.25):
    t = str(t)
    if not t.strip(): return False
    dev   = sum(1 for c in t if "\u0900" <= c <= "\u097f")
    alpha = sum(1 for c in t if c.isalpha())
    return alpha > 0 and dev / alpha >= ratio

def dedup(df):
    seen, keep = set(), []
    for i, r in df.iterrows():
        h = hashlib.md5(str(r["text"])[:200].encode()).hexdigest()
        if h not in seen: seen.add(h); keep.append(i)
    logger.info(f"Dedup: {len(df)} -> {len(keep)}")
    return df.loc[keep].reset_index(drop=True)

def mkrow(text, genre, source):
    return {"text": text, "genre": genre, "source": source}

def to_df(rows): return pd.DataFrame(rows) if rows else pd.DataFrame()

def group_passages(texts, size=3, min_w=30):
    """
    Group list of short texts into passages.
    size=3: 3 sentences × ~11 words = ~33 words > 30 minimum.
    Use size=3 not 5 to maximise passage count from limited data.
    """
    out, buf = [], []
    for t in texts:
        t = t.strip()
        if not t: continue
        buf.append(t)
        if len(buf) >= size:
            p = " ".join(buf)
            if wc(p) >= min_w: out.append(p)
            buf = []
    if buf:                                  # leftover
        p = " ".join(buf)
        if wc(p) >= min_w: out.append(p)
    return out


# ═══════════════════════════════════════════════════════════════
# LITERATURE — Hindi Wikipedia (literary)
# ═══════════════════════════════════════════════════════════════

LIT_KW = ["कहानी","उपन्यास","कविता","कवि","लेखक","साहित्य","नाटक",
           "प्रेमचंद","निराला","महादेवी","पंत","दिनकर","रेणु",
           "कथा","कथाकार","उपन्यासकार","रचना","साहित्यकार","काव्य",
           "कहानीकार","नाटककार","आत्मकथा","जीवनी"]

def load_wikipedia_lit(n=5000):
    try: from datasets import load_dataset
    except ImportError: return pd.DataFrame()
    logger.info("[Wikipedia/lit] Loading...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.hi",
                          split="train", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"[Wikipedia/lit] {e}"); return pd.DataFrame()
    rows = []
    for item in tqdm(ds, desc="[Wikipedia/lit]"):
        preview = str(item.get("title","")) + " " + str(item.get("text",""))[:300]
        if not any(k in preview for k in LIT_KW): continue
        text  = clean(str(item.get("text","")))
        words = text.split()
        if not is_hindi(text) or len(words) < MIN_WORDS: continue
        start = 20 if len(words) > 40 else 0
        p = " ".join(words[start:start+350])
        if wc(p) >= MIN_WORDS:
            rows.append(mkrow(p, "literature", "wikipedia_hi"))
        if len(rows) >= n: break
    df = to_df(rows)
    logger.info(f"[Wikipedia/lit] {len(df)} articles"); return df


# ═══════════════════════════════════════════════════════════════
# NEWS 1 — Hindi Wikipedia (political/governmental)
# ═══════════════════════════════════════════════════════════════

NEWS_KW = ["सरकार","मंत्री","संसद","चुनाव","राजनीति","अर्थव्यवस्था",
           "नीति","कानून","न्यायालय","प्रधानमंत्री","मुख्यमंत्री",
           "घोषणा","रिपोर्ट","आयोग","बजट","योजना","केंद्र","राज्य"]

def load_wikipedia_news(n=5000):
    try: from datasets import load_dataset
    except ImportError: return pd.DataFrame()
    logger.info("[Wikipedia/news] Loading...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.hi",
                          split="train", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"[Wikipedia/news] {e}"); return pd.DataFrame()
    rows = []
    for item in tqdm(ds, desc="[Wikipedia/news]"):
        title   = str(item.get("title",""))
        text    = str(item.get("text",""))
        preview = title + " " + text[:300]
        if not any(k in preview for k in NEWS_KW): continue
        text  = clean(text)
        words = text.split()
        if not is_hindi(text) or len(words) < 80: continue   # skip stubs
        rows.append(mkrow(" ".join(words[:350]), "news", "wikipedia_hi_news"))
        if len(rows) >= n: break
    df = to_df(rows)
    logger.info(f"[Wikipedia/news] {len(df)} articles"); return df


# ═══════════════════════════════════════════════════════════════
# NEWS 2 — XL-Sum Hindi (BBC news, 70K articles)
# ═══════════════════════════════════════════════════════════════

def load_xlsum(n=5000):
    try: from datasets import load_dataset
    except ImportError: return pd.DataFrame()
    logger.info("[XL-Sum] Loading Hindi BBC news...")
    rows = []
    try:
        ds = load_dataset("csebuetnlp/xlsum", "hindi", trust_remote_code=True)
        for split in ["train", "validation", "test"]:
            sp = ds.get(split)
            if sp is None: continue
            for item in tqdm(sp, desc=f"[XL-Sum] {split}"):
                for field in ["text", "article"]:
                    if field not in item: continue
                    text  = clean(str(item[field]))
                    words = text.split()
                    if not is_hindi(text) or len(words) < MIN_WORDS: continue
                    rows.append(mkrow(" ".join(words[:350]), "news", "xlsum_hindi"))
                    break
            if len(rows) >= n: break
    except Exception as e:
        logger.warning(f"[XL-Sum] {str(e)[:100]}")
    df = to_df(rows)
    logger.info(f"[XL-Sum] {len(df)} articles"); return df


# ═══════════════════════════════════════════════════════════════
# SOCIAL 1 — BBC Hindi NLI → passages
# KEY FIX: size=3 sentences per passage (3×11=33 > 30 min)
# Previously used size=4, min=50 which gave 0 passages
# ═══════════════════════════════════════════════════════════════

def load_bbc_nli():
    try: from datasets import load_dataset
    except ImportError: return pd.DataFrame()
    logger.info("[BBC NLI] Loading sentences...")
    sentences, seen = [], set()
    try:
        ds = load_dataset("midas/bbc_hindi_nli", trust_remote_code=True)
        for split in ["train", "test", "validation"]:
            sp = ds.get(split)
            if sp is None: continue
            for item in sp:
                s = clean(str(item.get("premise", "")))
                if s in seen or not is_hindi(s) or wc(s) < 4: continue
                seen.add(s); sentences.append(s)
    except Exception as e:
        logger.warning(f"[BBC NLI] {str(e)[:100]}"); return pd.DataFrame()

    logger.info(f"[BBC NLI] {len(sentences)} unique sentences, "
                f"avg {sum(wc(s) for s in sentences[:200])/min(200,len(sentences)):.1f} words")

    # size=3: 3 × ~11 words = ~33 > 30 minimum ✓
    passages = group_passages(sentences, size=3, min_w=MIN_WORDS_SOCIAL)
    rows = [mkrow(p, "social", "bbc_nli_passages")
            for p in passages if wc(p) <= MAX_WORDS]
    df = to_df(rows)
    logger.info(f"[BBC NLI] {len(sentences)} sentences -> {len(df)} passages")
    return df


# ═══════════════════════════════════════════════════════════════
# SOCIAL 2 — IndicSentiment (product reviews)
# ═══════════════════════════════════════════════════════════════

def load_indic_sentiment():
    try: from datasets import load_dataset
    except ImportError: return pd.DataFrame()
    logger.info("[IndicSentiment] Loading...")
    reviews = []
    try:
        ds = load_dataset("ai4bharat/IndicSentiment", "translation-hi",
                          trust_remote_code=True)
        for split in ["train", "test", "validation"]:
            sp = ds.get(split)
            if sp is None: continue
            for item in sp:
                text = None
                for col in ["INDIC REVIEW", "text", "review", "sentence"]:
                    if col in item and item[col]:
                        text = str(item[col]); break
                if not text: continue
                text = clean(text)
                if is_hindi(text) and wc(text) >= 4:
                    reviews.append(text)
    except Exception as e:
        logger.warning(f"[IndicSentiment] {str(e)[:100]}"); return pd.DataFrame()

    rows = []
    long  = [r for r in reviews if wc(r) >= MIN_WORDS_SOCIAL]
    short = [r for r in reviews if wc(r) < MIN_WORDS_SOCIAL]

    for r in long:
        if wc(r) <= MAX_WORDS:
            rows.append(mkrow(r, "social", "indic_sentiment"))

    for p in group_passages(short, size=4, min_w=MIN_WORDS_SOCIAL):
        if wc(p) <= MAX_WORDS:
            rows.append(mkrow(p, "social", "indic_sentiment"))

    df = to_df(rows)
    logger.info(f"[IndicSentiment] {len(reviews)} reviews -> {len(df)} passages")
    return df


# ═══════════════════════════════════════════════════════════════
# SOCIAL 3 — Hindi Discourse (dialogic + argumentative modes)
# Sentences from Hindi stories in conversational/argumentative register
# Distinct from literature because it captures dialogic informal language
# ═══════════════════════════════════════════════════════════════

def load_hindi_discourse():
    """
    midas/hindi_discourse: 4000+ sentences from Hindi stories
    labeled with discourse mode: narrative, descriptive, argumentative,
    dialogic, informative.

    We use 'dialogic' and 'argumentative' modes as social-register proxies:
    dialogic = conversational speech patterns
    argumentative = opinionated, informal rhetoric
    Both are structurally distinct from formal news/literary prose.
    """
    try: from datasets import load_dataset
    except ImportError: return pd.DataFrame()
    logger.info("[Hindi Discourse] Loading dialogic/argumentative sentences...")
    social_modes = {"D", "A"}      # Dialogic, Argumentative
    sentences = []
    try:
        ds = load_dataset("midas/hindi_discourse", trust_remote_code=True)
        for split in ["train", "test", "validation"]:
            sp = ds.get(split)
            if sp is None: continue
            for item in sp:
                mode = str(item.get("Discourse Mode", item.get("label", "")))
                text = clean(str(item.get("Sentence", item.get("text", ""))))
                if mode not in social_modes: continue
                if is_hindi(text) and wc(text) >= 4:
                    sentences.append(text)
    except Exception as e:
        logger.warning(f"[Hindi Discourse] {str(e)[:100]}"); return pd.DataFrame()

    logger.info(f"[Hindi Discourse] {len(sentences)} dialogic/argumentative sentences")

    # Group into passages
    passages = group_passages(sentences, size=3, min_w=MIN_WORDS_SOCIAL)
    rows = [mkrow(p, "social", "hindi_discourse")
            for p in passages if wc(p) <= MAX_WORDS]
    df = to_df(rows)
    logger.info(f"[Hindi Discourse] {len(sentences)} sentences -> {len(df)} passages")
    return df


# ═══════════════════════════════════════════════════════════════
# QUALITY FILTER
# ═══════════════════════════════════════════════════════════════

def quality_filter(df):
    n0 = len(df)
    df = df.copy()
    df["_wc"] = df["text"].apply(wc)

    def passes(row):
        min_w = MIN_WORDS_SOCIAL if row["genre"] == "social" else MIN_WORDS
        return min_w <= row["_wc"] <= MAX_WORDS

    df = df[df.apply(passes, axis=1)]
    df = df[df["text"].apply(is_hindi)]
    df = df[df["text"].apply(
        lambda t: len(set(t.split())) / max(len(t.split()), 1) > 0.22
    )]
    df = dedup(df)
    df = df.drop(columns=["_wc"], errors="ignore").reset_index(drop=True)
    logger.info(f"Quality filter: {n0} -> {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# BALANCE
# ═══════════════════════════════════════════════════════════════

def balance(df, target=TARGET_PER_GENRE):
    parts = []
    logger.info("\nBalancing:")
    for genre in df["genre"].unique():
        gdf = df[df["genre"] == genre]
        srcs = gdf["source"].unique()
        per_src = max(target // len(srcs), 200)
        for src in srcs:
            sdf = gdf[gdf["source"] == src]
            n   = min(len(sdf), per_src)
            parts.append(sdf.sample(n=n, random_state=42))
            logger.info(f"  {genre:12} / {src:32}: {n}")
    combined = pd.concat(parts, ignore_index=True)
    final    = []
    for genre in combined["genre"].unique():
        gdf = combined[combined["genre"] == genre]
        final.append(gdf.sample(n=min(len(gdf), target), random_state=42))
    return pd.concat(final, ignore_index=True)\
             .sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════

def print_stats(df):
    print("\n" + "="*65)
    print("FINAL DATASET STATISTICS")
    print("="*65)
    stats, ok = {}, True

    for genre in sorted(df["genre"].unique()):
        gdf = df[df["genre"] == genre]
        wcs = gdf["text"].apply(wc)
        nsrc = gdf["source"].nunique()
        print(f"\n  {genre.upper()}  — {len(gdf)} docs, {nsrc} sources")
        for src, cnt in gdf["source"].value_counts().items():
            print(f"    {src:<36} {cnt:>5}")
        print(f"    Words: mean={wcs.mean():.0f}  "
              f"median={wcs.median():.0f}  min={wcs.min()}")
        stats[genre] = {
            "n_docs": int(len(gdf)), "n_sources": int(nsrc),
            "sources": gdf["source"].value_counts().to_dict(),
            "word_mean": round(float(wcs.mean()), 1),
            "word_median": round(float(wcs.median()), 1),
        }
        if wcs.median() < 30: ok = False
        if len(gdf) < 1000:   ok = False

    print("\n  Jaccard vocabulary overlap:")
    genres = sorted(df["genre"].unique())
    vocabs = {}
    for g in genres:
        words = set()
        for t in df[df["genre"] == g]["text"].tolist()[:2000]:
            words.update(str(t).lower().split())
        vocabs[g] = words
    for i, g1 in enumerate(genres):
        for g2 in genres[i+1:]:
            j  = len(vocabs[g1] & vocabs[g2]) / len(vocabs[g1] | vocabs[g2])
            lv = "good" if j >= 0.10 else ("low" if j >= 0.05 else "very low")
            print(f"    {g1} <-> {g2}: J={j:.4f}  [{lv}]")
            stats[f"jaccard_{g1}_{g2}"] = round(j, 4)

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n  " + ("All checks passed." if ok else
          "WARNINGS above — check social doc count and word length."))

    if "social" in df["genre"].values:
        n_social = len(df[df["genre"] == "social"])
        if n_social < 2000:
            print(f"\n  Social genre has only {n_social} docs.")
            print("  To reach 5000, copy sarcasm CSVs from Windows to data/raw/")
            print("  and re-run without --skip-local.")
            print("  This is the maximum achievable from HuggingFace alone.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()

    logger.info("="*65)
    logger.info("Hindi Genre Corpus Builder — Definitive Final Version")
    logger.info("="*65)

    parts = []

    # LITERATURE
    logger.info("\n──── LITERATURE ────")
    df = load_wikipedia_lit(n=5000)
    if not df.empty: parts.append(df)

    # NEWS
    logger.info("\n──── NEWS ────")
    df = load_wikipedia_news(n=5000)
    if not df.empty: parts.append(df)
    df = load_xlsum(n=5000)
    if not df.empty: parts.append(df)

    # SOCIAL
    logger.info("\n──── SOCIAL ────")
    df = load_bbc_nli()           # ~1300 passages
    if not df.empty: parts.append(df)
    df = load_indic_sentiment()   # ~400 passages
    if not df.empty: parts.append(df)
    df = load_hindi_discourse()   # ~500 passages (new source)
    if not df.empty: parts.append(df)

    if not parts:
        logger.error("Nothing loaded."); sys.exit(1)

    raw = pd.concat(parts, ignore_index=True)
    print("\n  Raw counts:")
    for genre in ["literature", "news", "social"]:
        n   = len(raw[raw["genre"] == genre])
        src = raw[raw["genre"] == genre]["source"].nunique() if n > 0 else 0
        print(f"  {'OK' if n >= 500 else 'LOW!'} {genre}: {n} docs, {src} sources")

    filtered = quality_filter(raw)
    final    = balance(filtered)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info(f"\nSaved -> {OUTPUT_PATH}  ({len(final)} rows)")

    print_stats(final)


if __name__ == "__main__":
    main()