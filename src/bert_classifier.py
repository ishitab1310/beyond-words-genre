"""
bert_classifier.py
Fine-tunes a pretrained Hindi/multilingual BERT model for genre classification.
Compares against the structural feature classifier.

Supported models (pass via --bert_model):
  google/muril-base-cased       — Multilingual Universal Representations for Indian Languages
  ai4bharat/indic-bert          — IndicBERT (trained on 12 Indian languages)
  bert-base-multilingual-cased  — mBERT fallback

Requirements:
  pip install transformers torch datasets accelerate
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
GENRES      = ["literature", "news", "social"]
LABEL2ID    = {g: i for i, g in enumerate(GENRES)}
ID2LABEL    = {i: g for g, i in LABEL2ID.items()}


def run_bert_classifier(
    df: pd.DataFrame,
    model_name: str = "google/muril-base-cased",
    n_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    sample_size: int = 3000,
):
    """
    Fine-tune a BERT-style model on the genre classification task.

    Parameters
    ----------
    df         : DataFrame with 'text' and 'genre' columns
    model_name : HuggingFace model identifier
    n_epochs   : Training epochs (3 is usually sufficient)
    batch_size : Per-device batch size
    max_length : Token sequence length limit
    sample_size: Subsample to this size per genre for tractable training
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
    except ImportError as e:
        logger.error(
            f"BERT dependencies not installed: {e}\n"
            "Run: pip install transformers torch datasets accelerate"
        )
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---------------------------------------------------------------
    # Subsample for tractable fine-tuning
    # ---------------------------------------------------------------
    sampled = (
        df.groupby("genre", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), sample_size), random_state=42))
        .reset_index(drop=True)
    )
    sampled["label"] = sampled["genre"].map(LABEL2ID)

    train_df, eval_df = train_test_split(
        sampled, test_size=0.2, random_state=42, stratify=sampled["label"]
    )

    logger.info(f"Train: {len(train_df)} | Eval: {len(eval_df)}")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    eval_ds  = Dataset.from_pandas(eval_df[["text", "label"]]).map(tokenize, batched=True)
    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(GENRES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": float(np.mean([
                sum(1 for p, l in zip(preds, labels) if p == l and l == i) /
                max(sum(1 for l in labels if l == i), 1)
                for i in range(len(GENRES))
            ])),
        }

    output_dir = os.path.join(RESULTS_DIR, "bert_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",              # <--- Changed: Do not save intermediate checkpoints
        load_best_model_at_end=False,    # <--- Changed: Must be False if save_strategy="no"
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Fine-tuning {model_name}...")
    trainer.train()

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    predictions = trainer.predict(eval_ds)
    preds  = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    true_genres = [ID2LABEL[i] for i in labels]
    pred_genres = [ID2LABEL[i] for i in preds]

    acc = accuracy_score(labels, preds)
    report = classification_report(true_genres, pred_genres, target_names=GENRES, output_dict=True)

    print("\n" + "=" * 60)
    print(f"BERT CLASSIFIER RESULTS — {model_name}")
    print("=" * 60)
    print(classification_report(true_genres, pred_genres, target_names=GENRES))

    # Confusion matrix
    cm = confusion_matrix(true_genres, pred_genres, labels=GENRES)
    disp = ConfusionMatrixDisplay(cm, display_labels=GENRES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Purples", colorbar=False)
    ax.set_title(f"Confusion matrix — {model_name.split('/')[-1]} (fine-tuned)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_bert.png"), dpi=150)
    plt.close()

    # Save results
    results = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "classification_report": report,
        "train_size": len(train_df),
        "eval_size":  len(eval_df),
        "n_epochs":   n_epochs,
        "max_length": max_length,
    }
    with open(os.path.join(RESULTS_DIR, "bert_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"BERT results saved → {RESULTS_DIR}/bert_results.json")
    return results


def compare_models(results_dir: str = RESULTS_DIR):
    """
    Load classifier, TF-IDF, and BERT results and print a comparison table.
    """
    files = {
        "Structural (LR)": "classifier_results.json",
        "TF-IDF baseline": "tfidf_results.json",
        "BERT":            "bert_results.json",
    }

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1 macro':>10}")
    print("  " + "-" * 47)

    for name, fname in files.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        if name == "Structural (LR)":
            acc = data.get("lr", {}).get("accuracy_mean", "-")
            f1  = data.get("lr", {}).get("f1_macro_mean", "-")
        elif name == "TF-IDF baseline":
            acc = data.get("TF-IDF (bigrams, 10k)", {}).get("accuracy_mean", "-")
            f1  = data.get("TF-IDF (bigrams, 10k)", {}).get("f1_macro_mean", "-")
        elif name == "BERT":
            acc = data.get("accuracy", "-")
            f1  = data.get("classification_report", {}).get("macro avg", {}).get("f1-score", "-")

        print(f"  {name:<25} {str(acc):>10} {str(f1):>10}")
