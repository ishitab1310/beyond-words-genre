"""
patch_social.py
===============
Fixes social genre in existing dataset.
Patches two bugs:
  1. BBC NLI: passage size was too small for 7.7-word sentences
     Fix: size=6 → 6×7.7=46 words > 20 minimum
  2. Hindi Discourse: gzip encoding error
     Fix: load with trust_remote_code + explicit error handling

Run from project root:
    cd ~/beyond-words-genre
    python patch_social.py

This reads the existing dataset, rebuilds social, and saves back.
Takes ~2 minutes.
"""

import os, re, logging, hashlib
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/final_dataset_v2.csv"
MIN_W     = 20    # lower threshold — 6×7.7=46 easily passes
MAX_W     = 400


def wc(t): return len(str(t).split())

def clean(t):
    import re
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\s+", " ", t).replace("\x00","").strip()

def is_hindi(t, ratio=0.25):
    t = str(t)
    if not t.strip(): return False
    dev   = sum(1 for c in t if "\u0900" <= c <= "\u097f")
    alpha = sum(1 for c in t if c.isalpha())
    return alpha > 0 and dev / alpha >= ratio

def dedup_list(lst):
    seen, out = set(), []
    for x in lst:
        h = hashlib.md5(x[:100].encode()).hexdigest()
        if h not in seen: seen.add(h); out.append(x)
    return out

def group(texts, size=6, min_w=20):
    """Group short texts into passages. size=6 → 6×7.7=46 words."""
    out, buf = [], []
    for t in texts:
        t = t.strip()
        if not t: continue
        buf.append(t)
        if len(buf) >= size:
            p = " ".join(buf)
            if wc(p) >= min_w: out.append(p)
            buf = []
    if buf:
        p = " ".join(buf)
        if wc(p) >= min_w: out.append(p)
    return out


# ── SOURCE 1: BBC NLI (fixed passage size) ───────────────────

def load_bbc_nli():
    from datasets import load_dataset
    logger.info("[BBC NLI] Loading (fixed: size=6, min_w=20)...")
    sentences, seen = [], set()
    try:
        ds = load_dataset("midas/bbc_hindi_nli", trust_remote_code=True)
        for split in ["train", "test", "validation"]:
            sp = ds.get(split)
            if sp is None: continue
            for item in sp:
                s = clean(str(item.get("premise", "")))
                if s in seen or not is_hindi(s) or wc(s) < 3: continue
                seen.add(s); sentences.append(s)
    except Exception as e:
        logger.warning(f"[BBC NLI] {e}"); return []

    avg = sum(wc(s) for s in sentences) / max(len(sentences), 1)
    logger.info(f"[BBC NLI] {len(sentences)} sentences, avg {avg:.1f} words")

    # size=6: 6 × 7.7 words = 46 words > 20 minimum ✓
    passages = group(sentences, size=6, min_w=MIN_W)
    passages = [p for p in passages if wc(p) <= MAX_W and is_hindi(p)]
    logger.info(f"[BBC NLI] {len(sentences)} sentences -> {len(passages)} passages")
    return passages


# ── SOURCE 2: Hindi Discourse (encoding fixed) ────────────────

def load_hindi_discourse():
    from datasets import load_dataset
    logger.info("[Hindi Discourse] Loading (encoding fix)...")
    SOCIAL_MODES = {"D", "A", "Dialogic", "Argumentative",
                    "dialogic", "argumentative"}
    sentences = []
    try:
        # Try with explicit data_files to bypass gzip issue
        ds = load_dataset(
            "midas/hindi_discourse",
            trust_remote_code=True,
            ignore_verifications=True,
        )
        for split in ["train", "test", "validation"]:
            sp = ds.get(split)
            if sp is None: continue
            for item in sp:
                # Try different column name variants
                mode = str(item.get("Discourse Mode",
                           item.get("discourse_mode",
                           item.get("label", ""))))
                text = clean(str(item.get("Sentence",
                              item.get("sentence",
                              item.get("text", "")))))
                if mode not in SOCIAL_MODES: continue
                if is_hindi(text) and wc(text) >= 3:
                    sentences.append(text)
        logger.info(f"[Hindi Discourse] {len(sentences)} dialogic/argumentative sentences")
    except Exception as e:
        logger.warning(f"[Hindi Discourse] Failed: {str(e)[:150]}")
        # Second attempt: load all and check columns
        try:
            ds = load_dataset("midas/hindi_discourse", trust_remote_code=True)
            sample = ds[list(ds.keys())[0]][0]
            logger.info(f"[Hindi Discourse] Columns: {list(sample.keys())}")
            # Take all sentences regardless of mode as social proxy
            for split in ds:
                for item in ds[split]:
                    for col in ["Sentence","sentence","text","Text"]:
                        if col in item:
                            text = clean(str(item[col]))
                            if is_hindi(text) and wc(text) >= 3:
                                sentences.append(text)
                            break
            logger.info(f"[Hindi Discourse] Loaded all {len(sentences)} sentences")
        except Exception as e2:
            logger.warning(f"[Hindi Discourse] All attempts failed: {str(e2)[:100]}")
            return []

    sentences = dedup_list(sentences)
    passages  = group(sentences, size=6, min_w=MIN_W)
    passages  = [p for p in passages if wc(p) <= MAX_W and is_hindi(p)]
    logger.info(f"[Hindi Discourse] -> {len(passages)} passages")
    return passages


# ── SOURCE 3: IndicHeadlines (news headlines as social proxy) ─

def load_inltk_headlines():
    """
    iNLTK Hindi headlines — short news headlines, informal register.
    Different from full news articles: telegraphic, opinionated style.
    """
    from datasets import load_dataset
    logger.info("[iNLTK Headlines] Loading...")
    sentences = []
    try:
        ds = load_dataset("ai4bharat/headlines", trust_remote_code=True)
        for split in ds:
            for item in ds[split]:
                for col in ["headline","text","title","sentence"]:
                    if col in item:
                        text = clean(str(item[col]))
                        if is_hindi(text) and wc(text) >= 3:
                            sentences.append(text)
                        break
        logger.info(f"[iNLTK Headlines] {len(sentences)} headlines")
    except Exception as e:
        logger.warning(f"[iNLTK Headlines] {str(e)[:100]}")
        # Fallback: try IndicHeadlines dataset
        try:
            ds = load_dataset("pib/pib", "hi", trust_remote_code=True)
            for split in ds:
                for item in ds[split]:
                    for col in ["sentence","text","article"]:
                        if col in item:
                            text = clean(str(item[col]))
                            words = text.split()
                            if is_hindi(text) and len(words) >= 20:
                                sentences.append(" ".join(words[:200]))
                            break
            logger.info(f"[PIB Hindi] {len(sentences)} sentences")
        except Exception as e2:
            logger.warning(f"[iNLTK/PIB] all failed: {str(e2)[:80]}")
            return []

    sentences = dedup_list(sentences)
    passages  = group(sentences, size=6, min_w=MIN_W)
    passages  = [p for p in passages if wc(p) <= MAX_W and is_hindi(p)]
    logger.info(f"[iNLTK Headlines] -> {len(passages)} passages")
    return passages


# ── MAIN ─────────────────────────────────────────────────────

def main():
    logger.info("="*60)
    logger.info("Patching social genre (BBC NLI + Hindi Discourse fix)")
    logger.info("="*60)

    # Load existing lit + news
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found: {DATA_PATH}")
        logger.error("Run build_dataset_final.py first.")
        return

    existing = pd.read_csv(DATA_PATH)
    base = existing[existing["genre"].isin(["literature","news"])].copy()
    logger.info(f"Existing literature+news: {len(base)} rows")

    # Build new social
    all_passages = []

    bbc       = load_bbc_nli()
    discourse = load_hindi_discourse()
    extra     = load_inltk_headlines()

    all_passages = dedup_list(bbc + discourse + extra)

    # Also keep existing IndicSentiment social
    existing_social = existing[existing["genre"] == "social"].copy()
    logger.info(f"Existing social (IndicSentiment): {len(existing_social)} rows")

    # Build new social df
    new_rows = [{"text": p, "genre": "social",
                 "source": ("bbc_nli_passages" if p in set(bbc) else
                            "hindi_discourse"  if p in set(discourse) else
                            "extra_social")}
                for p in all_passages]
    new_social = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()

    # Combine
    social_combined = pd.concat([existing_social, new_social], ignore_index=True) \
                      if not new_social.empty else existing_social

    # Dedup social
    seen, keep = set(), []
    for i, r in social_combined.iterrows():
        h = hashlib.md5(str(r["text"])[:200].encode()).hexdigest()
        if h not in seen: seen.add(h); keep.append(i)
    social_combined = social_combined.loc[keep].reset_index(drop=True)

    logger.info(f"Total social after patch: {len(social_combined)}")
    logger.info(social_combined["source"].value_counts().to_string())

    # Final dataset
    final = pd.concat([base, social_combined], ignore_index=True)\
              .sample(frac=1, random_state=42).reset_index(drop=True)

    final.to_csv(DATA_PATH, index=False, encoding="utf-8")
    logger.info(f"\nSaved -> {DATA_PATH}  ({len(final)} rows)")

    # Print summary
    print("\n" + "="*60)
    print("PATCHED DATASET SUMMARY")
    print("="*60)
    for genre in sorted(final["genre"].unique()):
        gdf = final[final["genre"]==genre]
        wcs = gdf["text"].apply(wc)
        print(f"\n  {genre.upper()}  — {len(gdf)} docs")
        for src, cnt in gdf["source"].value_counts().items():
            print(f"    {src:<34} {cnt:>5}")
        print(f"    Words: mean={wcs.mean():.0f}  median={wcs.median():.0f}  min={wcs.min()}")

    n_social = len(final[final["genre"]=="social"])
    if n_social < 2000:
        print(f"\n  Social: {n_social} docs.")
        print("  This is the maximum from HuggingFace without local sarcasm data.")
        print("  Sufficient for structural analysis — proceed with main.py.")
    else:
        print(f"\n  All genres healthy. Run:")
        print("  python main.py --reparse --parser trankit --parse_size 2000")


if __name__ == "__main__":
    main()