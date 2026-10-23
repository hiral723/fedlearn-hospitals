"""
Runs plain FedAvg federated training across the 20 non-IID simulated
hospitals created by partition.py, and evaluates the global model on
the centralized test set after every round.
"""

import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import flwr as fl

from model import ChestCNN, XrayDataset, CLASSES, get_device

NUM_CLIENTS = 20
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.005
CLIENTS_PER_ROUND = 5

BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR.parent / "data" / "processed" / "test"
CLIENT_INDICES_PATH = BASE_DIR / "client_indices.json"
RESULTS_PATH = BASE_DIR / "results_baseline.json"
PLOT_PATH = BASE_DIR / "accuracy_baseline.png"

DEVICE = get_device()
print(f"Using device: {DEVICE}")


def load_client_data():
    with open(CLIENT_INDICES_PATH) as f:
        client_samples = json.load(f)
    return [[(s[0], s[1]) for s in samples] for samples in client_samples]


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
    def __init__(self, client_id, samples):
        self.client_id = client_id
        self.model = ChestCNN()
        self.loader = None
        if len(samples) > 0:
            self.loader = DataLoader(XrayDataset(samples), batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        if self.loader is None:
            return get_params(self.model), 1, {}
        train_local(self.model, self.loader, LOCAL_EPOCHS, LEARNING_RATE)
        return get_params(self.model), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        if self.loader is None:
            return 0.0, 1, {"accuracy": 0.0}
        acc = evaluate(self.model, self.loader)
        return 0.0, len(self.loader.dataset), {"accuracy": acc}


def make_client_fn(client_data):
    def client_fn(context: fl.common.Context):
        cid = int(context.node_config["partition-id"])
        samples = client_data[cid]
        return HospitalClient(cid, samples).to_client()
    return client_fn


def main():
    client_data = load_client_data()
    test_samples = load_test_data()
    test_loader = DataLoader(XrayDataset(test_samples), batch_size=32, shuffle=False)
    print(f"Loaded {len(test_samples)} centralized test images.")

    global_model = ChestCNN()
    history = {"round": [], "accuracy": []}

    def evaluate_fn(server_round, parameters, config):
        set_params(global_model, parameters)
        acc = evaluate(global_model, test_loader)
        history["round"].append(server_round)
        history["accuracy"].append(acc)
        print(f"Round {server_round}: centralized test accuracy = {acc:.4f}")
        return 0.0, {"accuracy": acc}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CLIENTS_PER_ROUND / NUM_CLIENTS,
        fraction_evaluate=0.0,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(get_params(global_model)),
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(client_data),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if DEVICE.type == "cpu" else 1.0},
    )

    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    plt.figure(figsize=(9, 5))
    plt.plot(history["round"], history["accuracy"], marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("Centralized Test Accuracy")
    plt.title(f"Baseline FedAvg on Non-IID Data ({NUM_CLIENTS} hospitals)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved accuracy plot to {PLOT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    input("Press Enter to exit...")