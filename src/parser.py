import os
import stanza


nlp = stanza.Pipeline(
    'hi',
    processors='tokenize,pos,lemma,depparse',
    use_gpu=False,
    verbose=False
)


def parse_and_save(texts, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for text in texts:
            doc = nlp(text)

            for sentence in doc.sentences:
                for word in sentence.words:
                    line = [
                        str(word.id),
                        word.text,
                        word.lemma,
                        word.upos,
                        "_",
                        "_",
                        str(word.head),
                        word.deprel,
                        "_",
                        "_"
                    ]
                    f.write("\t".join(line) + "\n")

                f.write("\n")  # sentence break

    print(f"Saved parsed file at {save_path}")