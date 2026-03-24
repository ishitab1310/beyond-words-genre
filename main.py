"""
main.py — Beyond Words: Structural Information Content of Genre in Hindi
Full EMNLP-level pipeline.

Run modes
---------
# Standard (fast, ~30 min):
python main.py

# With Trankit morphology re-parsing (slow, run once overnight):
python main.py --reparse --parser trankit

# With BERT:
python main.py --run_bert --bert_model google/muril-base-cased

# Full EMNLP mode (everything):
python main.py --full --run_bert
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
    from src.dependency_analysis import (
        analyze_dependencies, print_key_patterns,
        compute_statistical_significance,
    )
    from src.linguistic_analysis import run_full_linguistic_analysis
    from src.evaluation import aggregate_results

    # ── new EMNLP modules ──────────────────────────────────────
    from src.deconfound import (
        run_vocabulary_overlap,
        run_corpus_identity_probe,
    )
    from src.probing_experiments import run_all_probing_experiments
    from src.advanced_stats import run_advanced_statistics
    from src.cross_corpus_eval import run_cross_corpus_evaluation

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # 1. Load data
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Data loading")
    logger.info("=" * 60)
    df = load_data(sample_size=args.sample_size)

    # ──────────────────────────────────────────────────────────
    # 2. Corpus identity deconfounding (NEW)
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Corpus identity deconfounding analysis")
    logger.info("=" * 60)
    run_vocabulary_overlap(df)
    run_corpus_identity_probe(df)

    # ──────────────────────────────────────────────────────────
    # 3. Surface + structural feature extraction
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Feature extraction")
    logger.info("=" * 60)
    X, feature_names = build_feature_df(df)
    y = df["genre"].values

    # ──────────────────────────────────────────────────────────
    # 4. PCA
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — PCA visualization")
    logger.info("=" * 60)
    run_pca(X, y, feature_names)

    # ──────────────────────────────────────────────────────────
    # 5. Probing experiments (NEW — core EMNLP contribution)
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Probing experiments")
    logger.info("  Word-shuffle | POS-only | Func-word | Dep-only")
    logger.info("=" * 60)
    run_all_probing_experiments(df)

    # ──────────────────────────────────────────────────────────
    # 6. Full structural classifier + ablation
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Structural classifier (k-fold CV + ablation)")
    logger.info("=" * 60)
    train_classifier(X, y, feature_names)
    run_ablation_study(X, y, feature_names)

    # ──────────────────────────────────────────────────────────
    # 7. TF-IDF baseline
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7 — TF-IDF lexical baseline")
    logger.info("=" * 60)
    run_tfidf_baseline(df)

    # ──────────────────────────────────────────────────────────
    # 8. Dependency parsing (cached)
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8 — Dependency parsing")
    logger.info("=" * 60)
    _run_parsing(df, args)

    # ──────────────────────────────────────────────────────────
    # 9. Dependency analysis + linguistic analysis
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9 — Dependency & linguistic analysis")
    logger.info("=" * 60)
    results, all_samples = analyze_dependencies()
    print_key_patterns(results)
    compute_statistical_significance(all_samples)
    run_full_linguistic_analysis()

    # ──────────────────────────────────────────────────────────
    # 10. Advanced statistics (NEW)
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 10 — Advanced statistics (bootstrap CIs, Cohen's d, permutation)")
    logger.info("=" * 60)
    run_advanced_statistics(all_samples)

    # ──────────────────────────────────────────────────────────
    # 11. Cross-corpus evaluation (NEW)
    # ──────────────────────────────────────────────────────────
    if args.full or args.cross_corpus:
        logger.info("=" * 60)
        logger.info("STEP 11 — Cross-corpus generalization evaluation")
        logger.info("=" * 60)
        run_cross_corpus_evaluation(df)

    # ──────────────────────────────────────────────────────────
    # 12. BERT classifier (optional)
    # ──────────────────────────────────────────────────────────
    if args.run_bert:
        logger.info("=" * 60)
        logger.info("STEP 12 — BERT fine-tuning (MuRIL)")
        logger.info("=" * 60)
        from src.bert_classifier import run_bert_classifier
        run_bert_classifier(df, model_name=args.bert_model)

    # ──────────────────────────────────────────────────────────
    # 13. Aggregated results table
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 13 — Aggregated model comparison")
    logger.info("=" * 60)
    aggregate_results()

    logger.info("\nPipeline complete. All results in results/")
    logger.info("Now run:  python src/visualizations.py")


def _run_parsing(df, args):
    """Dispatch to Stanza or Trankit parser."""
    if args.parser == "trankit":
        from src.trankit_parser import parse_and_save_trankit as parse_and_save
    else:
        from src.parser import parse_and_save

    for genre in ["literature", "news", "social"]:
        save_path = f"data/parsed/{genre}.conllu"
        if not os.path.exists(save_path) or args.reparse:
            texts = (
                df[df["genre"] == genre]["text"]
                .sample(
                    min(args.parse_size, len(df[df["genre"] == genre])),
                    random_state=42,
                )
                .tolist()
            )
            parse_and_save(texts, save_path)
        else:
            logger.info(f"Using cached parse: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beyond Words — Hindi Genre Classification (EMNLP pipeline)"
    )
    parser.add_argument("--sample_size", type=int, default=15000)
    parser.add_argument("--parse_size",  type=int, default=2000)
    parser.add_argument("--reparse",     action="store_true")
    parser.add_argument("--parser",      choices=["stanza", "trankit"], default="stanza",
                        help="Parser backend. Use 'trankit' for morphological features.")
    parser.add_argument("--run_bert",    action="store_true")
    parser.add_argument("--bert_model",  type=str, default="google/muril-base-cased")
    parser.add_argument("--cross_corpus", action="store_true",
                        help="Run cross-corpus generalization evaluation")
    parser.add_argument("--full",        action="store_true",
                        help="Run everything: BERT + cross-corpus + all probing experiments")
    args = parser.parse_args()

    if args.full:
        args.run_bert = True
        args.cross_corpus = True

    main(args)