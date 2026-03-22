import numpy as np

STOPWORDS = set(["है", "का", "के", "की", "और", "को", "में", "से"])
CONJUNCTIONS = set(["और", "लेकिन", "क्योंकि", "पर", "तथा", "या"])


def extract_features(text):
    text = str(text)
    words = text.split()

    char_len = len(text)
    word_len = len(words)

    # Basic stats
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    num_digits = sum(c.isdigit() for c in text)
    num_punct = sum(not c.isalnum() and not c.isspace() for c in text)

    # Sentence features
    sentences = text.replace(".", "।").split("।")
    sentences = [s for s in sentences if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    avg_sentence_len = word_len / num_sentences if num_sentences else 0

    # Linguistic ratios
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    stopword_ratio = stopword_count / word_len if word_len else 0

    digit_ratio = num_digits / char_len if char_len else 0
    punct_density = num_punct / char_len if char_len else 0

    conjunction_count = sum(1 for w in words if w in CONJUNCTIONS)

    long_word_ratio = sum(len(w) > 6 for w in words) / word_len if word_len else 0
    unique_word_ratio = len(set(words)) / word_len if word_len else 0

    return {
        "avg_word_len": avg_word_len,
        "num_digits": num_digits,
        "num_punct": num_punct,
        "num_sentences": num_sentences,
        "avg_sentence_len": avg_sentence_len,
        "stopword_ratio": stopword_ratio,
        "digit_ratio": digit_ratio,
        "punct_density": punct_density,
        "conjunction_count": conjunction_count,
        "long_word_ratio": long_word_ratio,
        "unique_word_ratio": unique_word_ratio,
    }


def build_feature_df(df):
    features = df["text"].apply(extract_features)

    feature_names = list(features.iloc[0].keys())
    features_df = features.apply(lambda x: list(x.values())).tolist()

    return np.array(features_df), feature_names