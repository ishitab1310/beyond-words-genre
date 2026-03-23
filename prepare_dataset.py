"""
prepare_dataset.py
Combines BHAAV (literature), Hindi News (news), and Sarcasm tweets (social)
into a single processed CSV at data/processed/final_dataset.csv.

Edit the PATH constants below to match your local data locations.
"""

import os
import json
import logging
import pandas as pd
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# PATH CONFIG — edit these to match your local setup
# ---------------------------------------------------------------
BASE_PATH = r"F:/00Spring_2026/LD3/project/Datasets-20190922T151602Z-001/Datasets"
NEWS_PATH = r"F:/00Spring_2026/LD3/project/hindi_news_dataset.csv"
SARCASM_PATH = r"F:/00Spring_2026/LD3/project/Sarcasm_Hindi_Tweets-SARCASTIC.csv/Sarcastic.csv"
NON_SARCASM_PATH = r"F:/00Spring_2026/LD3/project/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv/NonSarcastic.csv"

OUTPUT_PATH = "data/processed/final_dataset.csv"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return pd.read_csv(f, on_bad_lines="skip")


def clean_text(text: str) -> str:
    return (
        str(text)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\x00", "")
        .strip()
    )


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 5]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------
def load_bhaav() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "Story Json Files")
    files = glob(os.path.join(path, "*.json"))
    data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            story = json.load(f)
        for entry in story:
            _, sentence, emotion = entry
            data.append({
                "text": sentence,
                "genre": "literature",
                "source": "bhaav",
                "label": emotion,
            })
    logger.info(f"[BHAAV] Loaded {len(data)} sentences")
    return pd.DataFrame(data)


def load_news() -> pd.DataFrame:
    df = safe_read_csv(NEWS_PATH)
    df = df.rename(columns={"Content": "text", "News Categories": "label"})
    df["genre"] = "news"
    df["source"] = "news"
    df = df[["text", "genre", "source", "label"]]
    logger.info(f"[NEWS] Loaded {len(df)} articles")
    return df


def load_sarcasm() -> pd.DataFrame:
    df1 = safe_read_csv(SARCASM_PATH).dropna(subset=["text"])
    df1["label"] = "sarcastic"
    df1["genre"] = "social"
    df1["source"] = "sarcasm"

    df2 = safe_read_csv(NON_SARCASM_PATH).dropna(subset=["text"])
    df2["label"] = "non_sarcastic"
    df2["genre"] = "social"
    df2["source"] = "sarcasm"

    df = pd.concat([df1, df2], ignore_index=True)[["text", "genre", "source", "label"]]
    logger.info(f"[SARCASM] Loaded {len(df)} tweets")
    return df


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    logger.info("Preparing dataset...")

    bhaav   = clean_df(load_bhaav())
    news    = clean_df(load_news())
    sarcasm = clean_df(load_sarcasm())

    final_df = pd.concat([bhaav, news, sarcasm], ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    logger.info(f"Dataset saved → {OUTPUT_PATH}")
    logger.info(f"Total rows: {len(final_df)}")
    logger.info(f"\n{final_df['genre'].value_counts().to_string()}")
    logger.info(f"\nSample:\n{final_df.head().to_string()}")


if __name__ == "__main__":
    main()
