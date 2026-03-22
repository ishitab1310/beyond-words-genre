import os
import json
import pandas as pd
from glob import glob

# -------------------------------
# PATH CONFIG (FIXED)
# -------------------------------
BASE_PATH = r"F:/00Spring_2026/LD3/project/Datasets-20190922T151602Z-001/Datasets"

NEWS_PATH = r"F:/00Spring_2026/LD3/project/hindi_news_dataset.csv"

SARCASM_PATH = r"F:/00Spring_2026/LD3/project/Sarcasm_Hindi_Tweets-SARCASTIC.csv/Sarcastic.csv"
NON_SARCASM_PATH = r"F:/00Spring_2026/LD3/project/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv/NonSarcastic.csv"

OUTPUT_PATH = "../data/processed/final_dataset.csv"


# -------------------------------
# SAFE CSV LOADER (IMPORTANT FIX)
# -------------------------------
def safe_read_csv(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return pd.read_csv(f, on_bad_lines="skip")


# -------------------------------
# 1. LOAD BHAAV DATA
# -------------------------------
def load_bhaav():
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
                    "label": emotion
                })

    print(f"[BHAAV] Loaded {len(data)} sentences")
    return pd.DataFrame(data)


# -------------------------------
# 2. LOAD NEWS DATA
# -------------------------------
def load_news():
    df = safe_read_csv(NEWS_PATH)

    df = df.rename(columns={
        "Content": "text",
        "News Categories": "label"
    })

    df["genre"] = "news"
    df["source"] = "news"

    df = df[["text", "genre", "source", "label"]]

    print(f"[NEWS] Loaded {len(df)} articles")
    return df


# -------------------------------
# 3. LOAD SARCASM DATA (FIXED)
# -------------------------------
def load_sarcasm():
    data = []

    # sarcastic = 1
    df1 = safe_read_csv(SARCASM_PATH)
    df1 = df1.dropna(subset=["text"])

    df1["label"] = 1
    df1["genre"] = "social"
    df1["source"] = "sarcasm"

    # non sarcastic = 0
    df2 = safe_read_csv(NON_SARCASM_PATH)
    df2 = df2.dropna(subset=["text"])

    df2["label"] = 0
    df2["genre"] = "social"
    df2["source"] = "sarcasm"

    df = pd.concat([df1, df2], ignore_index=True)
    df = df[["text", "genre", "source", "label"]]

    print(f"[SARCASM] Loaded {len(df)} tweets")
    return df


# -------------------------------
# 4. CLEAN TEXT
# -------------------------------
def clean_text(text):
    return (
        str(text)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\x00", "")   # REMOVE NULL BYTES (CRUCIAL FIX)
        .strip()
    )


def clean_df(df):
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 5]
    return df


# -------------------------------
# 5. MAIN PIPELINE
# -------------------------------
def main():
    print("Preparing dataset...\n")

    bhaav = clean_df(load_bhaav())
    news = clean_df(load_news())
    sarcasm = clean_df(load_sarcasm())

    final_df = pd.concat([bhaav, news, sarcasm], ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n Final dataset saved at:", OUTPUT_PATH)
    print("\nSample:\n", final_df.head())


if __name__ == "__main__":
    main()