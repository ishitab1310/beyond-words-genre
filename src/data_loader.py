import pandas as pd

DATA_PATH = "../data/processed/final_dataset.csv"


def load_data(sample_size=50000):
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "genre"])

    # sample equally from each genre
    df = (
        df.groupby("genre", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), sample_size // 3), random_state=42))
        .reset_index(drop=True)
    )

    print(f"[LOADER] Loaded {len(df)} samples")
    print(df["genre"].value_counts())

    return df