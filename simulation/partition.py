"""
Splits data/processed/train/ across NUM_CLIENTS simulated hospitals using
a Dirichlet distribution, so each hospital ends up with a skewed, uneven
mix of the 4 disease classes -- mimicking real-world hospital differences.

Saves:
  simulation/client_indices.json      -- which files go to which client
  simulation/client_distributions.png -- bar chart to visually confirm skew

Run from project root:
    python simulation/partition.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import CLASSES

TRAIN_DIR = Path(__file__).parent.parent / "data" / "processed" / "train"
OUT_JSON = Path(__file__).parent / "client_indices.json"
OUT_PLOT = Path(__file__).parent / "client_distributions.png"

NUM_CLIENTS = 20
ALPHA = 1.0  # lower = more skewed/non-IID, higher = closer to uniform
SEED = 42


def dirichlet_partition(class_file_lists, num_clients, alpha, seed):
    """class_file_lists: dict label_idx -> list of filepaths.
    Returns: list (len num_clients) of list of (filepath, label_idx)."""
    rng = np.random.default_rng(seed)
    client_samples = [[] for _ in range(num_clients)]

    for label_idx, files in class_file_lists.items():
        files = list(files)
        rng.shuffle(files)
        n = len(files)

        # sample proportions for this class across clients from a Dirichlet
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # convert proportions to actual counts, ensure they sum to n
        counts = (proportions * n).astype(int)
        counts[-1] = n - counts[:-1].sum()  # fix rounding drift

        start = 0
        for client_id, count in enumerate(counts):
            chunk = files[start:start + count]
            client_samples[client_id].extend([(str(f), label_idx) for f in chunk])
            start += count

    return client_samples


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit(
            f"{TRAIN_DIR} not found. Run data/prepare_data.py first."
        )

    class_file_lists = {}
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = TRAIN_DIR / class_name
        files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        class_file_lists[label_idx] = files
        print(f"{class_name}: {len(files)} training images found")

    client_samples = dirichlet_partition(class_file_lists, NUM_CLIENTS, ALPHA, SEED)

    # Save to JSON (filepath, label) pairs per client
    with open(OUT_JSON, "w") as f:
        json.dump(client_samples, f)
    print(f"\nSaved partition for {NUM_CLIENTS} clients to {OUT_JSON}")

    # Build a stacked bar chart: one bar per client, segments per class
    counts_matrix = np.zeros((NUM_CLIENTS, len(CLASSES)), dtype=int)
    for client_id, samples in enumerate(client_samples):
        for _, label_idx in samples:
            counts_matrix[client_id, label_idx] += 1

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(NUM_CLIENTS)
    for label_idx, class_name in enumerate(CLASSES):
        ax.bar(range(NUM_CLIENTS), counts_matrix[:, label_idx], bottom=bottom, label=class_name)
        bottom += counts_matrix[:, label_idx]

    ax.set_xlabel("Simulated Hospital (Client) ID")
    ax.set_ylabel("Number of Training Images")
    ax.set_title(f"Non-IID Class Distribution Across {NUM_CLIENTS} Simulated Hospitals (alpha={ALPHA})")
    ax.set_xticks(range(NUM_CLIENTS))
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"Saved distribution chart to {OUT_PLOT}")
    print("\nOpen that PNG -- you should see clearly uneven class mixes per hospital.")


if __name__ == "__main__":
    main()