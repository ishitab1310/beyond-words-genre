"""
parser.py
Parses Hindi text using Stanza and saves CoNLL-U format files.
Includes morphological features in column 6 (standard CoNLL-U slot).
"""

import os
import logging
import stanza
from tqdm import tqdm

logger = logging.getLogger(__name__)

_nlp = None


def get_pipeline():
    global _nlp
    if _nlp is None:
        logger.info("Loading Stanza Hindi pipeline (tokenize, pos, lemma, morph, depparse)...")
        _nlp = stanza.Pipeline(
            "hi",
            processors="tokenize,pos,lemma,depparse",
            use_gpu=False,
            verbose=False,
        )
    return _nlp


def parse_and_save(texts: list, save_path: str, batch_size: int = 32):
    """
    Parse a list of Hindi strings and write a CoNLL-U file.

    CoNLL-U columns written:
      ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
    FEATS is populated from word.feats (Stanza morphological features).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nlp = get_pipeline()

    total_sentences = 0

    with open(save_path, "w", encoding="utf-8") as f:
        for text in tqdm(texts, desc=f"Parsing → {os.path.basename(save_path)}"):
            try:
                doc = nlp(str(text))
            except Exception as e:
                logger.warning(f"Parse error (skipping): {e}")
                continue

            for sentence in doc.sentences:
                total_sentences += 1
                for word in sentence.words:
                    feats = word.feats if word.feats else "_"
                    line = "\t".join([
                        str(word.id),
                        word.text,
                        word.lemma if word.lemma else "_",
                        word.upos  if word.upos  else "_",
                        "_",          # XPOS
                        feats,        # FEATS  ← morphological features
                        str(word.head),
                        word.deprel if word.deprel else "_",
                        "_",          # DEPS
                        "_",          # MISC
                    ])
                    f.write(line + "\n")
                f.write("\n")  # sentence boundary

    logger.info(f"Saved {total_sentences} sentences → {save_path}")
