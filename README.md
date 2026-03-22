'''
# Beyond Words: Structural Information Content of Genre in Hindi
![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.27-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.1-lightgrey)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-2.3-yellow)
![Stanza](https://img.shields.io/badge/Stanza-NLP-red)
![TF-IDF](https://img.shields.io/badge/TF--IDF-Baseline-lightblue)
![Statistical Analysis](https://img.shields.io/badge/ANOVA-Significance-purple)
### Introduction

This project investigates whether **syntactic and structural features alone** can distinguish between different text genres in Hindi, without relying on lexical content.

We analyze three genres:
- Literature
- News
- Social Media

The core idea:
> Can structure (not words) encode genre?

---

### Pipeline

1. **Data Loading**
   - Balanced sampling across genres
   - ~15,000 samples (5K per genre)

2. **Feature Extraction**
   - Surface features:
     - avg_word_len
     - num_sentences
     - punct_density
   - Linguistic features:
     - stopword_ratio
     - conjunction_count
   - Structural features:
     - POS ratios (noun, verb, etc.)
     - Dependency relations (nsubj, obj, etc.)

3. **Dimensionality Reduction**
   - PCA visualization of feature space

4. **Classification**
   - Structural feature-based classifier
   - TF-IDF baseline comparison

5. **Dependency Analysis**
   - Parsing using Stanza
   - Extract syntactic patterns per genre

6. **Statistical Testing**
   - ANOVA to verify significance of differences

---

### Results

#### 🔹 Structural Classifier
Accuracy: **96%**

#### 🔹 TF-IDF Baseline
Accuracy: **97%**

> Structural features alone achieve competitive performance without using lexical content.

---

###  Dependency Patterns

| Relation | Literature | News | Social |
|----------|----------|------|--------|
| nsubj    | 0.1083   | 0.0635 | 0.0822 |
| obj      | 0.0569   | 0.0428 | 0.0231 |
| obl      | 0.0870   | 0.0781 | 0.0362 |
| advmod   | 0.0302   | 0.0087 | 0.0180 |
| amod     | 0.0216   | 0.0294 | 0.0181 |

---

### Statistical Significance

All differences are statistically significant:

- nsubj: p < 0.001  
- obj: p < 0.001  
- obl: p < 0.001  
- advmod: p < 0.001  
- amod: p < 0.001  

> This confirms that syntactic patterns vary systematically across genres.

---

### Key Insights

1. **Literature**
   - Highest `nsubj` and `obj`
   - Indicates narrative, action-driven structure

2. **News**
   - Highest `amod`
   - Reflects descriptive precision and factual reporting

3. **Social Media**
   - Lower `obj` and `obl`
   - Suggests fragmented and informal structure

4. **Adverbial usage**
   - Higher in literature → expressive storytelling

---

### Limitations

- Dependency parsing performed on a subset (≈200 samples per genre)
- Results may be influenced by text length and dataset bias
- Structural features do not capture semantic nuance

---

### How to Run

```bash
python main.py
```
### Project Structure
```
src/
 ├── data_loader.py
 ├── feature_extractor.py
 ├── parser.py
 ├── dependency_analysis.py
 ├── classifier.py
 ├── tfidf_baseline.py
 └── pca_analysis.py

data/
 ├── processed/
 └── parsed/

results/
 └── plots/
 ```
### Contribution

This work shows that:

- Genre is encoded not only in words, but in structure.

- Even without lexical information, syntactic patterns provide strong signals for genre classification.

### Future Work

- Combine structural + lexical features

- Extend to more genres

- Cross-lingual analysis

- Deep learning models on dependency graphs

