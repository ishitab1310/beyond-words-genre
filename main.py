"""
Beyond Words: Structural Information Content of Genre in Hindi
Main pipeline runner
"""

import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main(args):
    from src.data_loader import load_data
    from src.feature_extractor import build_feature_df
    from src.pca_analysis import run_pca
    from src.classifier import train_classifier, run_ablation_study
    from src.tfidf_baseline import run_tfidf_baseline
    from src.parser import parse_and_save
    from src.dependency_analysis import (
        analyze_dependencies,
        print_key_patterns,
        compute_statistical_significance,
    )
    from src.bert_classifier import run_bert_classifier
    from src.linguistic_analysis import run_full_linguistic_analysis

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    logger.info("Loading data...")
    df = load_data(sample_size=args.sample_size)

    # ---------------------------------------------------------------
    # 2. Surface + structural feature extraction
    # ---------------------------------------------------------------
    logger.info("Extracting features...")
    X, feature_names = build_feature_df(df)
    y = df["genre"].values

    # ---------------------------------------------------------------
    # 3. PCA visualization
    # ---------------------------------------------------------------
    logger.info("Running PCA...")
    run_pca(X, y, feature_names)

    # ---------------------------------------------------------------
    # 4. Structural classifier (with k-fold CV + ablation)
    # ---------------------------------------------------------------
    logger.info("Training structural classifier...")
    train_classifier(X, y, feature_names)

    logger.info("Running ablation study...")
    run_ablation_study(X, y, feature_names)

    # ---------------------------------------------------------------
    # 5. TF-IDF lexical baseline
    # ---------------------------------------------------------------
    logger.info("Running TF-IDF baseline...")
    run_tfidf_baseline(df)

    # ---------------------------------------------------------------
    # 6. Dependency parsing (cached)
    # ---------------------------------------------------------------
    logger.info("Running dependency parsing...")
    for genre in ["literature", "news", "social"]:
        save_path = f"data/parsed/{genre}.conllu"
        if not os.path.exists(save_path) or args.reparse:
            texts = (
                df[df["genre"] == genre]["text"]
                .sample(min(args.parse_size, len(df[df["genre"] == genre])), random_state=42)
                .tolist()
            )
            parse_and_save(texts, save_path)
            logger.info(f"Parsed {genre}: {len(texts)} sentences → {save_path}")
        else:
            logger.info(f"Using cached parse: {save_path}")

    # ---------------------------------------------------------------
    # 7. Dependency analysis
    # ---------------------------------------------------------------
    logger.info("Analyzing dependency patterns...")
    results, all_samples = analyze_dependencies()
    print_key_patterns(results)
    compute_statistical_significance(all_samples)

    # ---------------------------------------------------------------
    # 8. Full linguistic analysis (POS, morphology, rigidity, MDD)
    # ---------------------------------------------------------------
    logger.info("Running full linguistic analysis...")
    run_full_linguistic_analysis()

    # ---------------------------------------------------------------
    # 9. BERT classifier (optional — slow)
    # ---------------------------------------------------------------
    if args.run_bert:
        logger.info("Fine-tuning BERT classifier...")
        run_bert_classifier(df, model_name=args.bert_model)
    else:
        logger.info("Skipping BERT (pass --run_bert to enable)")

    logger.info("Pipeline complete. Results saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hindi Genre Classification Pipeline")
    parser.add_argument("--sample_size", type=int, default=15000,
                        help="Total samples (split equally across genres)")
    parser.add_argument("--parse_size", type=int, default=2000,
                        help="Sentences per genre to parse with Stanza")
    parser.add_argument("--reparse", action="store_true",
                        help="Force re-parsing even if .conllu files exist")
    parser.add_argument("--run_bert", action="store_true",
                        help="Fine-tune and evaluate BERT model")
    parser.add_argument("--bert_model", type=str,
                        default="google/muril-base-cased",
                        help="HuggingFace model ID for BERT experiment")
    args = parser.parse_args()
    main(args)
