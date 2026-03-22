import os
from collections import Counter
from scipy.stats import f_oneway
import numpy as np


def parse_conllu_per_sentence(file_path):
    samples = []

    current_counts = Counter()
    total = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # sentence boundary
            if not line:
                if total > 0:
                    ratios = {
                        "nsubj": current_counts.get("nsubj", 0) / total,
                        "obj": current_counts.get("obj", 0) / total,
                        "obl": current_counts.get("obl", 0) / total,
                        "advmod": current_counts.get("advmod", 0) / total,
                        "amod": current_counts.get("amod", 0) / total,
                    }
                    samples.append(ratios)

                # reset
                current_counts = Counter()
                total = 0
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            dep = parts[7]
            current_counts[dep] += 1
            total += 1

    return samples


def analyze_dependencies():
    base_path = "data/parsed"
    genres = ["literature", "news", "social"]

    results = {}
    all_samples = {}

    for genre in genres:
        file_path = os.path.join(base_path, f"{genre}.conllu")

        samples = parse_conllu_per_sentence(file_path)
        all_samples[genre] = samples

        # average for reporting
        avg = {
            key: sum(s[key] for s in samples) / len(samples)
            for key in ["nsubj", "obj", "obl", "advmod", "amod"]
        }

        results[genre] = avg

    return results, all_samples


def print_key_patterns(results):
    keys = ["nsubj", "obj", "obl", "advmod", "amod"]

    print("\n Dependency Patterns (Ratios):\n")

    for genre, ratios in results.items():
        print(f"\n--- {genre.upper()} ---")
        for k in keys:
            print(f"{k}: {ratios.get(k, 0):.4f}")

    print("\n KEY COMPARISONS\n")

    for dep in keys:
        values = {genre: results[genre][dep] for genre in results}
        highest = max(values, key=values.get)
        print(f"{dep}: highest in {highest} → {values}")

def compute_eta_squared(lit, news, soc):
 
    all_data = np.array(lit + news + soc)
    grand_mean = np.mean(all_data)

    ss_between = (
        len(lit) * (np.mean(lit) - grand_mean) ** 2 +
        len(news) * (np.mean(news) - grand_mean) ** 2 +
        len(soc) * (np.mean(soc) - grand_mean) ** 2
    )

    ss_total = np.sum((all_data - grand_mean) ** 2)
    return ss_between / ss_total

def compute_statistical_significance(all_samples):
    print("\n STATISTICAL SIGNIFICANCE (ANOVA)\n")

    deps = ["nsubj", "obj", "obl", "advmod", "amod"]

    for dep in deps:
        lit = [s[dep] for s in all_samples["literature"]]
        news = [s[dep] for s in all_samples["news"]]
        soc = [s[dep] for s in all_samples["social"]]

        stat, p = f_oneway(lit, news, soc)
        eta = compute_eta_squared(lit, news, soc)
        print(f"{dep}: p={p:.6f}, eta²={eta:.4f}", end=" ")
        if p < 0.05:
            print("Significant")
        else:
            print("Not Significant")

