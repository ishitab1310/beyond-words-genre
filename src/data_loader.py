"""
data_loader.py  (updated for dataset v2)
=========================================
Loads final_dataset_v2.csv with multiple sources per genre.
Supports:
  - Standard balanced sampling (for main experiments)
  - Source-aware splits (for cross-corpus evaluation)
  - Separate train/test by source (for generalization tests)
"""

import logging
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

# Update this path after running build_dataset.py
DATA_PATH_V2 = "data/processed/final_dataset_v2.csv"
DATA_PATH_V1 = "data/processed/final_dataset.csv"


def load_data(
    sample_size: int = 15000,
    random_state: int = 42,
    data_path: str = None,
) -> pd.DataFrame:
    """
    Load and balance the dataset.
    Tries v2 first (multi-source), falls back to v1.
    """
    if data_path is None:
        import os
        data_path = DATA_PATH_V2 if os.path.exists(DATA_PATH_V2) else DATA_PATH_V1
        if data_path == DATA_PATH_V1:
            logger.warning(
                "Using v1 dataset (single source per genre). "
                "Run build_dataset.py to create the improved v2 dataset."
            )

    df = pd.read_csv(data_path, encoding="utf-8")
    df = df.dropna(subset=["text", "genre"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]

    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = "unknown"

    genres   = df["genre"].unique()
    per_genre = sample_size // len(genres)

    balanced = (
        df.groupby("genre", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), per_genre), random_state=random_state))
        .reset_index(drop=True)
    )

    logger.info(f"Loaded {len(balanced)} samples from {data_path}")
    logger.info(f"\nGenre distribution:\n{balanced['genre'].value_counts().to_string()}")

    if "source" in balanced.columns:
        logger.info(f"\nSource distribution:\n{balanced['source'].value_counts().to_string()}")

    return balanced


def load_cross_corpus_splits(data_path: str = None):
    """
    Returns train/test splits where train and test come from DIFFERENT sources.

    For each genre:
      - Source A  → training set
      - Source B  → test set (held-out)

    This directly tests cross-corpus generalization.
    Returns: (df_train, df_test)
    """
    import os
    if data_path is None:
        data_path = DATA_PATH_V2 if os.path.exists(DATA_PATH_V2) else None

    if data_path is None:
        raise FileNotFoundError(
            "Dataset v2 not found. Run build_dataset.py first."
        )

    df = pd.read_csv(data_path, encoding="utf-8")
    df = df.dropna(subset=["text", "genre"])

    if "source" not in df.columns:
        raise ValueError("Dataset v2 required — must have 'source' column.")

    train_parts = []
    test_parts  = []

    # Define which source is held-out per genre
    # Modify this mapping based on what sources you actually have
    HOLDOUT_SOURCE = {
        "literature": "wikipedia_hi",       # hold out Wikipedia, train on BHAAV
        "news":       "wikipedia_hi_news",  # hold out Wikipedia, train on BBC
        "social":     "iitpatna_reviews",   # hold out reviews, train on tweets
    }

    for genre in df["genre"].unique():
        gdf     = df[df["genre"] == genre]
        holdout = HOLDOUT_SOURCE.get(genre)
        sources_available = gdf["source"].unique().tolist()

        if holdout and holdout in sources_available:
            test_parts.append(gdf[gdf["source"] == holdout])
            train_parts.append(gdf[gdf["source"] != holdout])
        else:
            # Fall back to random 80/20 if only one source
            n_test = len(gdf) // 5
            test_parts.append(gdf.sample(n=n_test, random_state=42))
            train_parts.append(gdf.drop(gdf.sample(n=n_test, random_state=42).index))
            logger.warning(
                f"Genre '{genre}': only 1 source available, using random split. "
                f"Available sources: {sources_available}"
            )

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test  = pd.concat(test_parts,  ignore_index=True)

    logger.info(f"Cross-corpus train: {len(df_train)} | test: {len(df_test)}")
    logger.info(f"Train sources: {df_train['source'].value_counts().to_dict()}")
    logger.info(f"Test sources:  {df_test['source'].value_counts().to_dict()}")

    return df_train, df_test


def get_source_info(df: pd.DataFrame) -> dict:
    """Return a summary of sources per genre."""
    if "source" not in df.columns:
        return {}
    info = {}
    for genre in df["genre"].unique():
        gdf = df[df["genre"] == genre]
        info[genre] = {
            "n_docs":   len(gdf),
            "sources":  gdf["source"].value_counts().to_dict(),
            "n_sources": gdf["source"].nunique(),
        }
    return info