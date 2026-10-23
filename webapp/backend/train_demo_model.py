"""
Trains the demo model on the FULL training dataset (all classes, all images).
Saves: webapp/backend/demo_model.pt
Run from project root: python webapp/backend/train_demo_model.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SIM_DIR = str(Path(__file__).parent.parent.parent / "simulation")
sys.path.append(SIM_DIR)
from model import ChestCNN, XrayDataset, CLASSES, get_device  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_OUT_PATH = Path(__file__).parent / "demo_model.pt"

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = get_device()
print(f"Using device: {DEVICE}")


def load_samples(base_dir):
    samples = []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = base_dir / class_name
        for ext in ("*.png", "*.jpg"):
            for f in class_dir.glob(ext):
                samples.append((str(f), label_idx))
    return samples


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


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit(f"{TRAIN_DIR} not found. Run data/prepare_data.py first.")

    train_samples = load_samples(TRAIN_DIR)
    test_samples = load_samples(TEST_DIR)
    print(f"Training on FULL dataset: {len(train_samples)} images")
    print(f"Evaluating on {len(test_samples)} test images")

    train_loader = DataLoader(XrayDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(XrayDataset(test_samples), batch_size=32, shuffle=False)

    model = ChestCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}/{EPOCHS}: avg loss = {running_loss / len(train_loader):.4f}, test accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            print(f"  -> New best ({acc:.4f}), saved checkpoint")

    print(f"\nDone. Best test accuracy: {best_acc:.4f}")
    print(f"Best model saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
