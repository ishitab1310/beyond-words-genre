"""
data_loader.py
Loads and balances the final_dataset.csv produced by prepare_dataset.py.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/final_dataset.csv"


def load_data(sample_size: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Load the processed dataset and return a stratified, balanced sample.

    Parameters
    ----------
    sample_size : int
        Total number of rows to keep. Rows are split *equally* across genres.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame  with columns: text, genre, source, label
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df.dropna(subset=["text", "genre"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 5]

    genres = df["genre"].unique()
    per_genre = sample_size // len(genres)

    balanced = (
        df.groupby("genre", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), per_genre), random_state=random_state))
        .reset_index(drop=True)
    )

    logger.info(f"Loaded {len(balanced)} samples")
    logger.info(f"\n{balanced['genre'].value_counts().to_string()}")
    return balanced
