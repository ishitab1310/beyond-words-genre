"""
PATCH — apply this to build_dataset.py to fix missing social genre.

Three changes:
1. MIN_WORDS_SOCIAL = 8  (BBC NLI premises are short — don't filter them)
2. quality_filter respects per-genre min word threshold
3. load_indic_sentiment uses load_dataset fallback (parquet path was wrong)
4. Adds load_iit_sentiment as another source using IndicSentiment via HF library

Replace the relevant functions in build_dataset.py with these versions.
Or just run patch_dataset.py which applies everything automatically.
"""

import os, re, json, logging, hashlib
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = "data/processed/final_dataset_v2.csv"

MIN_WORDS        = 25   # for literature + news
MIN_WORDS_SOCIAL = 8    # social posts/sentences are short
MAX_WORDS        = 450
TARGET_PER_GENRE = 5000


def wc(text): return len(str(text).split())

def is_hindi(text, min_ratio=0.25):
    text = str(text)
    if not text.strip(): return False
    dev   = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    alpha = sum(1 for c in text if c.isalpha())
    return alpha > 0 and dev / alpha >= min_ratio

def clean(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()

def dedup(df):
    seen, keep = set(), []
    for i, r in df.iterrows():
        h = hashlib.md5(str(r["text"])[:200].encode()).hexdigest()
        if h not in seen: seen.add(h); keep.append(i)
    logger.info(f"Dedup: {len(df)} -> {len(keep)}")
    return df.loc[keep].reset_index(drop=True)

def mkrow(text, genre, source, sid=""):
    return {"text": text, "genre": genre, "source": source, "source_id": sid}

def to_df(rows): return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── FIXED quality_filter: per-genre min word threshold ───────

def quality_filter(df):
    n0 = len(df)
    df = df.copy()
    df["_wc"] = df["text"].apply(wc)

    # Per-genre minimum — social posts can be shorter
    def passes_length(row):
        min_w = MIN_WORDS_SOCIAL if row["genre"] == "social" else MIN_WORDS
        return min_w <= row["_wc"] <= MAX_WORDS

    df = df[df.apply(passes_length, axis=1)]
    df = df[df["text"].apply(is_hindi)]
    df = df[df["text"].apply(
        lambda t: len(set(t.split())) / max(len(t.split()), 1) > 0.20
    )]
    df = dedup(df)
    df = df.drop(columns=["_wc"], errors="ignore").reset_index(drop=True)
    logger.info(f"Quality filter: {n0} -> {len(df)}")
    return df


# ── FIXED load_indic_sentiment: use HF library not parquet URL ─

def load_indic_sentiment():
    """
    ai4bharat/IndicSentiment — Hindi product/service reviews.
    Short, informal, opinion-driven — genuine social register.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return pd.DataFrame()

    logger.info("[IndicSentiment] Loading via HuggingFace library...")
    rows = []
    try:
        # Try multiple config names
        for config in ["translation-hi", "hi", None]:
            try:
                ds = (load_dataset("ai4bharat/IndicSentiment", config,
                                   trust_remote_code=True)
                      if config else
                      load_dataset("ai4bharat/IndicSentiment", trust_remote_code=True))

                logger.info(f"[IndicSentiment] Loaded config={config}")
                for split_name in ["train", "test", "validation"]:
                    sp = ds.get(split_name)
                    if sp is None: continue
                    for item in sp:
                        # Try common column names for IndicSentiment
                        text = None
                        for col in ["INDIC REVIEW", "text", "sentence",
                                    "review", "content", "sentence1"]:
                            if col in item and item[col]:
                                text = str(item[col]); break
                        if not text: continue
                        text = clean(text)
                        if is_hindi(text) and wc(text) >= MIN_WORDS_SOCIAL:
                            rows.append(mkrow(text, "social",
                                              "indic_sentiment", ""))
                if rows:
                    logger.info(f"[IndicSentiment] {len(rows)} reviews")
                    break
            except Exception as e:
                logger.warning(f"[IndicSentiment] config={config}: {str(e)[:80]}")
    except Exception as e:
        logger.warning(f"[IndicSentiment] {str(e)[:80]}")

    df = to_df(rows)
    logger.info(f"[IndicSentiment] {len(df)} reviews total")
    return df


# ── ADDITIONAL: Hindi Twitter/social via iNLTK ───────────────

def load_hindi_twitter():
    """
    Try multiple small Hindi social media datasets that are confirmed
    to exist on HuggingFace.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return pd.DataFrame()

    rows = []

    # Dataset 1: Hindi hate speech tweets
    sources = [
        ("ai4bharat/hindi-hate-speech", None),
        ("Genius1237/xlsum", "hindi"),        # XL-Sum Hindi — news summaries
        ("csebuetnlp/xlsum", "hindi"),
    ]

    for ds_name, config in sources:
        try:
            logger.info(f"[Social extra] Trying {ds_name} config={config}")
            ds = (load_dataset(ds_name, config, trust_remote_code=True)
                  if config else load_dataset(ds_name, trust_remote_code=True))
            for split_name in ["train", "test", "validation"]:
                sp = ds.get(split_name)
                if sp is None: continue
                for item in sp:
                    text = None
                    for col in ["text", "text_hi", "sentence", "tweet",
                                "post", "summary", "article"]:
                        if col in item and item[col]:
                            text = str(item[col]); break
                    if not text: continue
                    text = clean(text)
                    if is_hindi(text) and wc(text) >= MIN_WORDS_SOCIAL:
                        rows.append(mkrow(text, "social", ds_name.split("/")[-1], ""))
            if rows:
                logger.info(f"[Social extra] {len(rows)} from {ds_name}")
                break
        except Exception as e:
            logger.warning(f"[Social extra] {ds_name}: {str(e)[:80]}")

    df = to_df(rows)
    logger.info(f"[Social extra] {len(df)} total")
    return df


# ── MAIN PATCH: rebuild social + merge with existing dataset ──

def rebuild_social_and_merge():
    """
    Loads existing dataset, rebuilds social genre properly,
    merges, and saves back to OUTPUT_PATH.
    """
    logger.info("="*60)
    logger.info("Rebuilding social genre...")
    logger.info("="*60)

    # Load existing (literature + news)
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        existing = existing[existing["genre"].isin(["literature", "news"])]
        logger.info(f"Existing lit+news: {len(existing)} rows")
    else:
        existing = pd.DataFrame()
        logger.warning("No existing dataset found — run build_dataset.py first")

    social_parts = []

    # 1. Sarcasm tweets (auto-detect)
    import glob as glob_mod
    sarc_guesses = ["data/raw", os.path.expanduser("~/beyond-words-genre/data/raw"),
                    os.path.expanduser("~/data")]
    for d in sarc_guesses:
        if not os.path.isdir(d): continue
        for fname in os.listdir(d):
            fl = fname.lower(); fpath = os.path.join(d, fname)
            if fname.endswith(".csv"):
                try:
                    df_raw = pd.read_csv(fpath, on_bad_lines="skip",
                                         encoding="utf-8", errors="ignore")
                    tcol = next((c for c in df_raw.columns if "text" in c.lower()),
                                df_raw.columns[0] if len(df_raw.columns) else None)
                    if tcol is None: continue
                    count = 0
                    for text in df_raw[tcol].dropna():
                        text = clean(str(text))
                        if is_hindi(text) and wc(text) >= 5:
                            label = "sarcastic" if "sarcastic" in fl and "non" not in fl \
                                    else "non_sarcastic"
                            social_parts.append(
                                mkrow(text, "social", "sarcasm_tweets", label))
                            count += 1
                    if count > 0:
                        logger.info(f"[Sarcasm] {fname}: {count} tweets")
                except Exception:
                    continue

    # 2. BBC NLI (short sentences — now allowed since MIN_WORDS_SOCIAL=8)
    try:
        from datasets import load_dataset
        ds = load_dataset("midas/bbc_hindi_nli", trust_remote_code=True)
        seen = set()
        for sname in ["train", "test", "validation"]:
            sp = ds.get(sname)
            if sp is None: continue
            for item in sp:
                p = clean(str(item.get("premise", "")))
                if p in seen: continue
                seen.add(p)
                if is_hindi(p) and wc(p) >= MIN_WORDS_SOCIAL:
                    social_parts.append(mkrow(p, "social", "bbc_hindi_nli", ""))
        logger.info(f"[BBC NLI] {sum(1 for r in social_parts if r['source']=='bbc_hindi_nli')} sentences")
    except Exception as e:
        logger.warning(f"[BBC NLI] {e}")

    # 3. IndicSentiment
    s_df = load_indic_sentiment()
    if not s_df.empty:
        social_parts.extend(s_df.to_dict("records"))

    # 4. Extra social if needed
    if len(social_parts) < 1000:
        extra = load_hindi_twitter()
        if not extra.empty:
            social_parts.extend(extra.to_dict("records"))

    if not social_parts:
        logger.error("STILL no social data. You must copy sarcasm files from Mac:")
        logger.error("  scp path/to/Sarcastic.csv gnode076:~/beyond-words-genre/data/raw/")
        logger.error("  Then re-run: python patch_dataset.py")
        return

    social_df = to_df(social_parts)
    logger.info(f"Social raw: {len(social_df)} from {social_df['source'].nunique()} sources")

    # Combine all
    all_data = pd.concat([existing, social_df], ignore_index=True) \
               if not existing.empty else social_df

    # Quality filter with per-genre min words
    filtered = quality_filter(all_data)

    # Balance
    TARGET = 5000
    parts  = []
    logger.info("\nBalancing:")
    for genre in filtered["genre"].unique():
        gdf  = filtered[filtered["genre"] == genre]
        srcs = gdf["source"].unique()
        per_src = max(TARGET // len(srcs), 300)
        for src in srcs:
            sdf = gdf[gdf["source"] == src]
            n   = min(len(sdf), per_src)
            parts.append(sdf.sample(n=n, random_state=42))
            logger.info(f"  {genre:12} / {src:30}: {n}")

    combined = pd.concat(parts, ignore_index=True)
    final = []
    for genre in combined["genre"].unique():
        gdf = combined[combined["genre"] == genre]
        final.append(gdf.sample(n=min(len(gdf), TARGET), random_state=42))

    result = pd.concat(final, ignore_index=True)\
               .sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n" + "="*60)
    print("PATCHED DATASET")
    print("="*60)
    for genre in sorted(result["genre"].unique()):
        gdf = result[result["genre"] == genre]
        wcs = gdf["text"].apply(wc)
        print(f"\n  {genre.upper()}  — {len(gdf)} docs, {gdf['source'].nunique()} sources")
        for src, cnt in gdf["source"].value_counts().items():
            print(f"    {src:<34} {cnt:>5}")
        print(f"    Words: mean={wcs.mean():.0f}  median={wcs.median():.0f}")

    if "social" not in result["genre"].values:
        print("\n  !! SOCIAL STILL MISSING — copy sarcasm files and re-run !!")
        print("  scp path/to/Sarcastic.csv gnode076:~/beyond-words-genre/data/raw/")
    else:
        print(f"\n  Saved -> {OUTPUT_PATH}  ({len(result)} rows)")
        print("  Next: python main.py --reparse --parser trankit --parse_size 2000")


if __name__ == "__main__":
    rebuild_social_and_merge()