from src.data_loader import load_data
from src.parser import parse_and_save
from src.feature_extractor import build_feature_df
from src.pca_analysis import run_pca
from src.classifier import train_classifier
from src.dependency_analysis import analyze_dependencies, print_key_patterns
from src.tfidf_baseline import run_tfidf_baseline


import os

def main():
    df = load_data(sample_size=15000)

    X, feature_names = build_feature_df(df)
    y = df["genre"].values

    run_pca(X, y)
    train_classifier(X, y, feature_names)

    # TF-IDF baseline
    run_tfidf_baseline(df)

    # Parsing (only once)
    for genre in ["literature", "news", "social"]:
        save_path = f"data/parsed/{genre}.conllu"

        if not os.path.exists(save_path):
            texts = df[df["genre"] == genre]["text"].sample(200, random_state=42).tolist()
            parse_and_save(texts, save_path)

    # Dependency analysis
    results, all_samples = analyze_dependencies()

    print_key_patterns(results)

    from src.dependency_analysis import compute_statistical_significance
    compute_statistical_significance(all_samples)
    print("\n KEY COMPARISONS")
    

    for dep in ["nsubj", "obj", "advmod", "amod"]:
        values = {genre: results[genre][dep] for genre in results}
        highest = max(values, key=values.get)
        print(f"{dep}: highest in {highest} → {values}")


if __name__ == "__main__":
    main()