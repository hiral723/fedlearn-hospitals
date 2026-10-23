"""
A genuinely live, real federated training run -- implemented as plain
FedAvg in pure PyTorch (no Flower/Ray), safe to run inside a Flask
background thread.
"""

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SIM_DIR = str(Path(__file__).parent.parent.parent / "simulation")
sys.path.append(SIM_DIR)
from model import ChestCNN, XrayDataset, CLASSES, get_device  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLIENT_INDICES_PATH = PROJECT_ROOT / "simulation" / "client_indices.json"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"

DEVICE = get_device()


def load_test_loader(batch_size=32):
    samples = []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = TEST_DIR / class_name
        for ext in ("*.png", "*.jpg"):
            for f in class_dir.glob(ext):
                samples.append((str(f), label_idx))
    return DataLoader(XrayDataset(samples), batch_size=batch_size, shuffle=False)


def average_state_dicts(state_dicts, weights):
    total_weight = sum(weights)
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            avg_state[key] += sd[key].float() * (w / total_weight)
        avg_state[key] = avg_state[key].to(state_dicts[0][key].dtype)
    return avg_state


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def train_local(model, samples, epochs, lr, batch_size):
    loader = DataLoader(XrayDataset(samples), batch_size=batch_size, shuffle=True)
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return len(loader.dataset)


def run_live_training(hospital_ids, num_rounds=5, local_epochs=1, lr=0.005,
                       batch_size=16, progress_callback=None, stop_flag=None):
    with open(CLIENT_INDICES_PATH) as f:
        all_client_samples = json.load(f)

    hospital_data = {hid: [(s[0], s[1]) for s in all_client_samples[hid]] for hid in hospital_ids}
    test_loader = load_test_loader()

    global_model = ChestCNN().to(DEVICE)
    initial_acc = evaluate(global_model, test_loader)
    if progress_callback:
        progress_callback(0, initial_acc, [])

    for round_num in range(1, num_rounds + 1):
        if stop_flag is not None and stop_flag.should_stop:
            break

        local_states = []
        local_weights = []
        for hid in hospital_ids:
            local_model = ChestCNN().to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            n_samples = train_local(local_model, hospital_data[hid], local_epochs, lr, batch_size)
            local_states.append(local_model.state_dict())
            local_weights.append(max(n_samples, 1))

        new_global_state = average_state_dicts(local_states, local_weights)
        global_model.load_state_dict(new_global_state)

        acc = evaluate(global_model, test_loader)
        if progress_callback:
            progress_callback(round_num, acc, hospital_ids)

    return global_model
