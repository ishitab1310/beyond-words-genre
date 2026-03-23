"""
feature_extractor.py
Extracts surface, linguistic, and structural features from Hindi text.

Feature groups
--------------
SURFACE   : char/word counts, sentence stats, punctuation density, digit ratio
LEXICAL   : stopword ratio, conjunction count, long-word ratio, unique-word ratio
POS       : noun/verb/adjective/adverb/pronoun ratio (requires parsed .conllu)
SYNTAX    : loaded from pre-computed syntactic feature CSVs if available,
            otherwise falls back to surface+lexical only
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Hindi stopwords and conjunctions
# ---------------------------------------------------------------
STOPWORDS = set([
    "है", "का", "के", "की", "और", "को", "में", "से", "यह", "वह",
    "पर", "एक", "हैं", "था", "थी", "थे", "कि", "जो", "भी", "तो",
    "हो", "इस", "उस", "ने", "ही", "हम", "आप", "वे", "इन", "उन",
])

CONJUNCTIONS = set([
    "और", "लेकिन", "क्योंकि", "पर", "तथा", "या", "परंतु", "किंतु",
    "अथवा", "इसलिए", "जिससे", "ताकि", "मगर",
])

QUESTION_WORDS = set([
    "क्या", "कौन", "कहाँ", "कब", "कैसे", "क्यों", "कितना", "कितनी",
])

NEGATION_WORDS = set([
    "नहीं", "न", "मत", "ना", "नहिं",
])


def extract_surface_features(text: str) -> dict:
    """Character- and word-level surface features."""
    text = str(text)
    words = text.split()

    char_len = len(text)
    word_len = len(words)
    if word_len == 0:
        # Return zero-vector for empty text
        return {k: 0.0 for k in [
            "char_len", "word_len", "avg_word_len", "num_digits",
            "num_punct", "num_sentences", "avg_sentence_len",
            "stopword_ratio", "digit_ratio", "punct_density",
            "conjunction_count", "long_word_ratio", "unique_word_ratio",
            "question_word_ratio", "negation_ratio", "type_token_ratio",
        ]}

    avg_word_len = float(np.mean([len(w) for w in words]))
    num_digits   = sum(c.isdigit() for c in text)
    num_punct    = sum(not c.isalnum() and not c.isspace() for c in text)

    # Sentence segmentation (Devanagari danda + ASCII period)
    sentences = [s.strip() for s in text.replace(".", "।").split("।") if s.strip()]
    num_sentences    = max(len(sentences), 1)
    avg_sentence_len = word_len / num_sentences

    stopword_count   = sum(1 for w in words if w in STOPWORDS)
    conjunction_count = sum(1 for w in words if w in CONJUNCTIONS)
    question_count   = sum(1 for w in words if w in QUESTION_WORDS)
    negation_count   = sum(1 for w in words if w in NEGATION_WORDS)
    long_word_count  = sum(len(w) > 6 for w in words)
    unique_words     = len(set(words))

    return {
        "char_len":           float(char_len),
        "word_len":           float(word_len),
        "avg_word_len":       avg_word_len,
        "num_digits":         float(num_digits),
        "num_punct":          float(num_punct),
        "num_sentences":      float(num_sentences),
        "avg_sentence_len":   avg_sentence_len,
        "stopword_ratio":     stopword_count   / word_len,
        "digit_ratio":        num_digits       / max(char_len, 1),
        "punct_density":      num_punct        / max(char_len, 1),
        "conjunction_count":  float(conjunction_count),
        "long_word_ratio":    long_word_count  / word_len,
        "unique_word_ratio":  unique_words     / word_len,
        "question_word_ratio": question_count  / word_len,
        "negation_ratio":     negation_count   / word_len,
        "type_token_ratio":   unique_words     / word_len,
    }


def build_feature_df(df: pd.DataFrame):
    """
    Build the feature matrix from the text column.
    Also tries to merge pre-computed syntactic features (from linguistic_analysis).

    Returns
    -------
    X : np.ndarray  shape (n_samples, n_features)
    feature_names : list[str]
    """
    logger.info("Extracting surface features...")
    surface = df["text"].apply(extract_surface_features)
    feature_names = list(surface.iloc[0].keys())
    X = np.array(surface.apply(lambda x: list(x.values())).tolist())

    # ---------------------------------------------------------------
    # Optionally merge syntactic features if they were pre-computed
    # ---------------------------------------------------------------
    syn_path = "results/syntactic_features.csv"
    if os.path.exists(syn_path):
        logger.info(f"Merging syntactic features from {syn_path}")
        syn_df = pd.read_csv(syn_path)
        # align by index — syntactic features were computed on the same df
        if len(syn_df) == len(df):
            syn_cols = [c for c in syn_df.columns if c != "genre"]
            X = np.hstack([X, syn_df[syn_cols].values])
            feature_names += syn_cols
        else:
            logger.warning("Syntactic feature CSV length mismatch — skipping merge")

    logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, feature_names


# Expose feature group names for ablation study
SURFACE_FEATURES = [
    "char_len", "word_len", "avg_word_len", "num_digits", "num_punct",
    "num_sentences", "avg_sentence_len", "digit_ratio", "punct_density",
]

LEXICAL_FEATURES = [
    "stopword_ratio", "conjunction_count", "long_word_ratio",
    "unique_word_ratio", "question_word_ratio", "negation_ratio", "type_token_ratio",
]
