"""
Groups the 20 simulated hospitals into clusters based on how similar their
class-label distributions are. This uses ONLY label counts per hospital
(no images, no model weights) -- the same info a real hospital could safely
share with a coordinating server without exposing any patient data.

Reads:  simulation/client_indices.json (from partition.py)
Saves:  clustering/cluster_assignments.json
        clustering/cluster_visualization.png

Run from project root:
    python clustering/cluster_clients.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent.parent / "simulation"))
from config import CLASSES

BASE_DIR = Path(__file__).parent
CLIENT_INDICES_PATH = BASE_DIR.parent / "simulation" / "client_indices.json"
OUT_JSON = BASE_DIR / "cluster_assignments.json"
OUT_PLOT = BASE_DIR / "cluster_visualization.png"

NUM_CLUSTERS = 4  # tweak this -- try 3, 4, or 5 and see what looks cleanest
SEED = 42


def compute_label_distributions(client_samples):
    """Returns an (num_clients x num_classes) matrix of normalized class
    proportions -- this is the only thing that leaves each 'hospital'."""
    num_clients = len(client_samples)
    num_classes = len(CLASSES)
    counts = np.zeros((num_clients, num_classes))

    for client_id, samples in enumerate(client_samples):
        for _, label_idx in samples:
            counts[client_id, label_idx] += 1

    # normalize each row to a proportion (so hospital size doesn't affect similarity)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for any empty client
    proportions = counts / row_sums
    return proportions, counts


def main():
    if not CLIENT_INDICES_PATH.exists():
        raise SystemExit(f"{CLIENT_INDICES_PATH} not found. Run simulation/partition.py first.")

    with open(CLIENT_INDICES_PATH) as f:
        client_samples = json.load(f)

    proportions, counts = compute_label_distributions(client_samples)
    num_clients = len(client_samples)

    # --- cluster hospitals based on their label proportions ---
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init=10)
    cluster_ids = kmeans.fit_predict(proportions)

    # save assignments: cluster_id -> list of client_ids
    clusters = {int(c): [] for c in range(NUM_CLUSTERS)}
    for client_id, cluster_id in enumerate(cluster_ids):
        clusters[int(cluster_id)].append(client_id)

    with open(OUT_JSON, "w") as f:
        json.dump(clusters, f, indent=2)

    print(f"Clustered {num_clients} hospitals into {NUM_CLUSTERS} groups:")
    for cluster_id, client_ids in clusters.items():
        print(f"  Cluster {cluster_id}: hospitals {client_ids}")
    print(f"\nSaved assignments to {OUT_JSON}")

    # --- visualize: PCA down to 2D so we can actually see the grouping ---
    pca = PCA(n_components=2, random_state=SEED)
    coords_2d = pca.fit_transform(proportions)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: clusters in 2D space
    scatter = axes[0].scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=cluster_ids, cmap="tab10", s=200, edgecolors="black"
    )
    for client_id in range(num_clients):
        axes[0].annotate(str(client_id), (coords_2d[client_id, 0], coords_2d[client_id, 1]),
                          ha="center", va="center", fontsize=8, fontweight="bold")
    axes[0].set_title("Hospitals Grouped by Data Similarity (PCA projection)")
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")

    # Right plot: original stacked bar chart, but sorted/colored by cluster
    order = np.argsort(cluster_ids)
    bottom = np.zeros(num_clients)
    for label_idx, class_name in enumerate(CLASSES):
        axes[1].bar(range(num_clients), counts[order, label_idx], bottom=bottom, label=class_name)
        bottom += counts[order, label_idx]
    axes[1].set_xticks(range(num_clients))
    axes[1].set_xticklabels([str(order[i]) for i in range(num_clients)], fontsize=8)
    axes[1].set_xlabel("Hospital ID (grouped by cluster)")
    axes[1].set_ylabel("Number of Training Images")
    axes[1].set_title("Same Hospitals, Reordered by Cluster")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"Saved visualization to {OUT_PLOT}")
    print("\nOpen that PNG -- the left plot should show visually separated groups,")
    print("and the right plot should show hospitals of similar composition sitting next to each other.")


if __name__ == "__main__":
    main()