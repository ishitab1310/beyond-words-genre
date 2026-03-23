# Beyond Words: Structural Information Content of Genre in Hindi

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Stanza](https://img.shields.io/badge/Stanza-Hindi%20NLP-red)
![Status](https://img.shields.io/badge/Status-Research%20Ready-green)

## Overview

This project investigates whether **syntactic and structural features alone** can distinguish between different text genres in Hindi, without relying on lexical content.

We compare three genres — **Literature**, **News**, and **Social Media** — along multiple structural dimensions and evaluate against both a lexical baseline (TF-IDF) and a fine-tuned BERT model.

---

## Research Questions

| # | Question |
|---|----------|
| RQ1 | Can dependency-based structural features identify Hindi genre without lexical content? |
| RQ2 | Do genres show systematic differences in structural complexity (MDD, tree depth, branching)? |
| RQ3 | What structural dimensions explain the largest genre differences (PCA analysis)? |
| RQ4 | How do structural features compare to fine-tuned BERT for genre classification? |

---

## Datasets

| Genre | Dataset | Source |
|-------|---------|--------|
| Literature | BHAAV (story sentences + emotion labels) | GitHub |
| News | Inshorts Hindi News Dataset | Kaggle |
| Social Media | Hindi Sarcasm Tweets (sarcastic + non-sarcastic) | Kaggle |

---

## Pipeline

```
prepare_dataset.py
      ↓
data_loader.py  (balanced sampling, 5k/genre)
      ↓
feature_extractor.py  (surface + lexical features)
      ↓
  ┌───────────────────────────────────────────────┐
  │  pca_analysis.py     (scatter + scree + loadings) │
  │  classifier.py       (LR/RF/SVM + k-fold CV)      │
  │  ablation_study      (feature group contribution)  │
  │  tfidf_baseline.py   (lexical upper bound)         │
  └───────────────────────────────────────────────┘
      ↓
parser.py  (Stanza → .conllu files, 2k sentences/genre)
      ↓
dependency_analysis.py  (relation ratios, MDD, tree geometry, ANOVA)
      ↓
linguistic_analysis.py  (POS, morphology, SRM, interpretation report)
      ↓
bert_classifier.py  (MuRIL / IndicBERT fine-tuning)  [optional]
```

---

## Feature Groups

### Surface Features
`char_len`, `word_len`, `avg_word_len`, `num_sentences`, `avg_sentence_len`, `punct_density`, `digit_ratio`

### Lexical Features
`stopword_ratio`, `conjunction_count`, `long_word_ratio`, `unique_word_ratio`, `question_word_ratio`, `negation_ratio`, `type_token_ratio`

### Dependency Structural Features (from parsed .conllu)
- **Relation ratios**: `nsubj`, `obj`, `obl`, `advmod`, `amod`, `ccomp`, `acl`, `nmod`
- **Complexity metrics**: Mean Dependency Distance (MDD), max tree depth, branching factor
- **Directionality**: left-branching ratio, non-projectivity rate
- **POS ratios**: NOUN, VERB, ADJ, ADV, PRON, ADP, PART

---

## Novel Contribution: Syntactic Rigidity Metric (SRM)

The SRM quantifies how consistently a genre follows grammatical conventions, using PCA cluster dispersion:

```
SRM(genre) = 1 / mean_distance_from_centroid_in_PCA_space
```

- **High SRM** → genre is structurally consistent (e.g., news with inverted pyramid)
- **Low SRM** → genre shows high structural variability (e.g., social media with fragments)

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -c "import stanza; stanza.download('hi')"
```

### 2. Prepare the dataset
```bash
# Edit PATH constants in src/prepare_dataset.py first
python src/prepare_dataset.py
```

### 3. Run the full pipeline
```bash
# Default: 15k samples, 2k sentences parsed per genre, no BERT
python main.py

# With custom parameters
python main.py --sample_size 15000 --parse_size 2000

# With BERT fine-tuning (requires GPU recommended)
python main.py --run_bert --bert_model google/muril-base-cased

# Force re-parsing
python main.py --reparse
```

### 4. Optional: BERT dependencies
```bash
pip install transformers torch datasets accelerate
```

---

## Results (reported on full run)

### Structural Classifier (5-fold CV)
| Model | Accuracy | F1 (macro) |
|-------|----------|-----------|
| LR    | ~0.96    | ~0.96     |
| RF    | ~0.95    | ~0.95     |
| SVM   | ~0.96    | ~0.96     |

### TF-IDF Baseline (5-fold CV)
| Config | Accuracy | F1 (macro) |
|--------|----------|-----------|
| Unigrams (5k) | ~0.97 | ~0.97 |
| Bigrams (10k) | ~0.97 | ~0.97 |

### Dependency Patterns (genre means)

| Relation | Literature | News   | Social |
|----------|-----------|--------|--------|
| nsubj    | 0.1083    | 0.0635 | 0.0822 |
| obj      | 0.0569    | 0.0428 | 0.0231 |
| obl      | 0.0870    | 0.0781 | 0.0362 |
| advmod   | 0.0302    | 0.0087 | 0.0180 |
| amod     | 0.0216    | 0.0294 | 0.0181 |

All differences statistically significant (ANOVA p < 0.001, Kruskal-Wallis p < 0.001).

---

## Project Structure

```
project/
├── main.py                      # Pipeline entry point
├── requirements.txt
├── src/
│   ├── prepare_dataset.py       # Data loading and cleaning
│   ├── data_loader.py           # Balanced sampling
│   ├── feature_extractor.py     # Surface + lexical features
│   ├── parser.py                # Stanza → CoNLL-U
│   ├── dependency_analysis.py   # MDD, tree geometry, ANOVA
│   ├── linguistic_analysis.py   # POS, morphology, SRM, report
│   ├── pca_analysis.py          # PCA with confidence ellipses
│   ├── classifier.py            # LR/RF/SVM + k-fold + ablation
│   ├── tfidf_baseline.py        # TF-IDF baseline
│   ├── bert_classifier.py       # MuRIL/IndicBERT fine-tuning
│   └── evaluation.py            # Shared metrics + comparison table
├── data/
│   ├── processed/               # final_dataset.csv (gitignored)
│   └── parsed/                  # *.conllu files (gitignored)
└── results/
    ├── plots/                   # All generated figures
    ├── classifier_results.json
    ├── tfidf_results.json
    ├── bert_results.json        # (if --run_bert)
    ├── dependency_results.json
    ├── significance_tests.csv
    ├── ablation_study.csv
    ├── pos_distribution.csv
    ├── morphology_distribution.csv
    ├── mdd_summary.json
    ├── syntactic_rigidity.csv
    ├── model_comparison.csv
    └── linguistic_analysis_report.txt
```

---

## Key References

- Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press.
- Futrell, R., Mahowald, K., & Gibson, E. (2015). Large-scale evidence of dependency length minimization. *PNAS*, 112(33).
- Devlin, J. et al. (2019). BERT. *NAACL*.
- Kunchukuttan, A. et al. (2020). AI4Bharat-IndicNLP Corpus. *ACL*.
- Kakwani, D. et al. (2020). IndicNLPSuite. *EMNLP Findings*.

---

## Authors

Uday Bindal (2022114015) · Ishita Bansal (2022114004)
