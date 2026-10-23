"""
Runs FedAvg SEPARATELY within each hospital cluster (instead of one global
FedAvg across all 20 hospitals like run_baseline.py did), then compares
per-hospital accuracy against the baseline.

Requires: simulation/client_indices.json (from partition.py)
           clustering/cluster_assignments.json (from cluster_clients.py)

Run from project root:
    python clustering/run_clustered.py
"""

import json
import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import flwr as fl

import os
SIMULATION_DIR = str(Path(__file__).parent.parent / "simulation")
sys.path.append(SIMULATION_DIR)
os.environ["PYTHONPATH"] = SIMULATION_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

from model import ChestCNN, XrayDataset, get_device
from config import CLASSES

# ---- config: match run_baseline.py settings so results are comparable ----
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.005
MIN_CLIENTS_PER_ROUND = 2  # small clusters may not have 5 members
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
SIM_DIR = BASE_DIR.parent / "simulation"
TEST_DIR = BASE_DIR.parent / "data" / "processed" / "test"
CLIENT_INDICES_PATH = SIM_DIR / "client_indices.json"
CLUSTER_ASSIGNMENTS_PATH = BASE_DIR / "cluster_assignments.json"
RESULTS_PATH = BASE_DIR / "results_clustered.json"
PLOT_PATH = BASE_DIR / "accuracy_clustered_vs_baseline.png"
BASELINE_RESULTS_PATH = SIM_DIR / "results_baseline.json"

DEVICE = get_device()
print(f"Using device: {DEVICE}")


def load_test_data():
    samples = []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = TEST_DIR / class_name
        for ext in ("*.png", "*.jpg"):
            for f in class_dir.glob(ext):
                samples.append((str(f), label_idx))
    return samples


def get_params(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_params(model, params):
    keys = model.state_dict().keys()
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
    model.load_state_dict(state_dict, strict=True)


def train_local(model, loader, epochs, lr):
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


def evaluate(model, loader):
    model.to(DEVICE)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, samples):
        self.model = ChestCNN()
        self.loader = None
        if len(samples) > 0:
            self.loader = DataLoader(XrayDataset(samples), batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        if self.loader is None:
            # No data for this hospital in this split -- skip training,
            # return unchanged params so aggregation doesn't crash.
            return get_params(self.model), 1, {}
        train_local(self.model, self.loader, LOCAL_EPOCHS, LEARNING_RATE)
        return get_params(self.model), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        if self.loader is None:
            return 0.0, 1, {"accuracy": 0.0}
        acc = evaluate(self.model, self.loader)
        return 0.0, len(self.loader.dataset), {"accuracy": acc}


def make_client_fn(cluster_client_data):
    def client_fn(context: fl.common.Context):
        cid = int(context.node_config["partition-id"])
        samples = cluster_client_data[cid]
        return HospitalClient(samples).to_client()
    return client_fn


def run_fedavg_for_cluster(cluster_id, hospital_ids, all_client_samples, test_loader):
    """Runs a standalone FedAvg simulation using only the hospitals in this cluster."""
    cluster_client_data = [all_client_samples[hid] for hid in hospital_ids]
    num_clients = len(cluster_client_data)
    clients_per_round = min(MIN_CLIENTS_PER_ROUND, num_clients)

    print(f"\n=== Cluster {cluster_id}: {num_clients} hospitals {hospital_ids} ===")

    if num_clients < 2:
        print(f"Skipping cluster {cluster_id} -- needs at least 2 hospitals to federate.")
        return None

    global_model = ChestCNN()
    history = {"round": [], "accuracy": []}

    def evaluate_fn(server_round, parameters, config):
        set_params(global_model, parameters)
        acc = evaluate(global_model, test_loader)
        history["round"].append(server_round)
        history["accuracy"].append(acc)
        return 0.0, {"accuracy": acc}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(get_params(global_model)),
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(cluster_client_data),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if DEVICE.type == "cpu" else 1.0},
    )

    final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
    print(f"Cluster {cluster_id} final accuracy: {final_acc:.4f}")
    return history


def main():
    with open(CLIENT_INDICES_PATH) as f:
        raw_client_samples = json.load(f)
    all_client_samples = [[(s[0], s[1]) for s in samples] for samples in raw_client_samples]

    with open(CLUSTER_ASSIGNMENTS_PATH) as f:
        clusters = json.load(f)

    test_samples = load_test_data()
    test_loader = DataLoader(XrayDataset(test_samples), batch_size=32, shuffle=False)
    print(f"Loaded {len(test_samples)} centralized test images.")

    all_results = {}
    for cluster_id_str, hospital_ids in clusters.items():
        history = run_fedavg_for_cluster(int(cluster_id_str), hospital_ids, all_client_samples, test_loader)
        if history is not None:
            all_results[cluster_id_str] = history

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all cluster results to {RESULTS_PATH}")

    # --- comparison plot: baseline vs each cluster ---
    plt.figure(figsize=(10, 6))

    if BASELINE_RESULTS_PATH.exists():
        with open(BASELINE_RESULTS_PATH) as f:
            baseline = json.load(f)
        plt.plot(baseline["round"], baseline["accuracy"], marker="o", label="Baseline (1 global model)",
                  color="black", linewidth=2, linestyle="--")
    else:
        print("Note: simulation/results_baseline.json not found, skipping baseline comparison line.")

    for cluster_id_str, history in all_results.items():
        plt.plot(history["round"], history["accuracy"], marker="o", label=f"Cluster {cluster_id_str}")

    plt.xlabel("Federated Round")
    plt.ylabel("Centralized Test Accuracy")
    plt.title("Clustered FedAvg vs Baseline (Single Global Model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved comparison plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()