"""
trankit_parser.py
=================
Trankit-based Hindi parser — replaces Stanza for morphological features.

Why Trankit instead of Stanza?
  Stanza v1.7 Hindi model does not populate the FEATS column reliably.
  Trankit uses XLM-RoBERTa with language adapters and achieves +2.18%
  on Hindi UFeats (F1 = 93.21%) vs Stanza baseline.

Install:
  pip install trankit

Reference:
  Van Nguyen, Veyseh & Nguyen (2021) "Trankit: A Light-Weight
  Transformer-based Toolkit for Multilingual NLP" EACL 2021
"""

import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            import trankit
            logger.info("Loading Trankit Hindi pipeline ...")
            _pipeline = trankit.Pipeline(
                lang="hindi",
                gpu=True,
                cache_dir="./trankit_cache",
            )
            logger.info("Trankit pipeline loaded.")
        except ImportError:
            raise ImportError(
                "Trankit not installed. Run: pip install trankit\n"
                "Then re-run with --parser trankit"
            )
    return _pipeline


def _feats_to_str(feats_dict) -> str:
    """Convert Trankit feats dict to CoNLL-U string."""
    if not feats_dict:
        return "_"
    items = []
    for k, v in sorted(feats_dict.items()):
        if isinstance(v, list):
            v = ",".join(v)
        items.append(f"{k}={v}")
    return "|".join(items)


def parse_and_save_trankit(texts: list, save_path: str):
    """
    Parse texts using Trankit and save CoNLL-U with populated FEATS column.

    The FEATS column will contain morphological features like:
      Gender=Masc|Number=Sing|Case=Nom|Person=3
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nlp = get_pipeline()

    total_sentences = 0

    with open(save_path, "w", encoding="utf-8") as f:
        for text in tqdm(texts, desc=f"Trankit → {os.path.basename(save_path)}"):
            try:
                doc = nlp(str(text))
            except Exception as e:
                logger.warning(f"Parse error: {e}")
                continue

            if "sentences" not in doc:
                continue

            for sentence in doc["sentences"]:
                total_sentences += 1
                tokens = sentence.get("tokens", [])
                for token in tokens:
                    token_id  = token.get("id", 0)
                    form      = token.get("text", "_")
                    lemma     = token.get("lemma", "_") or "_"
                    upos      = token.get("upos",  "_") or "_"
                    xpos      = token.get("xpos",  "_") or "_"
                    feats     = _feats_to_str(token.get("feats", {}))
                    head      = token.get("head",    0)
                    deprel    = token.get("deprel", "_") or "_"

                    # Skip multi-word tokens
                    if not isinstance(token_id, int):
                        continue

                    line = "\t".join([
                        str(token_id), form, lemma, upos, xpos,
                        feats, str(head), deprel, "_", "_"
                    ])
                    f.write(line + "\n")
                f.write("\n")

    logger.info(f"Trankit: saved {total_sentences} sentences → {save_path}")