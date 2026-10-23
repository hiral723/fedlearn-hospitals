"""
Runs the trained demo model over the full test set and computes a
confusion matrix plus per-class precision/recall/F1.

Requires: webapp/backend/demo_model.pt
Saves:    webapp/backend/metrics.json
Run from project root: python webapp/backend/compute_metrics.py
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SIM_DIR = str(Path(__file__).parent.parent.parent / "simulation")
sys.path.append(SIM_DIR)
from model import ChestCNN, XrayDataset, CLASSES, get_device  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_PATH = Path(__file__).parent / "demo_model.pt"
OUT_PATH = Path(__file__).parent / "metrics.json"

DEVICE = get_device()


def load_samples(base_dir):
    samples = []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = base_dir / class_name
        for ext in ("*.png", "*.jpg"):
            for f in class_dir.glob(ext):
                samples.append((str(f), label_idx))
    return samples


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found. Run train_demo_model.py first.")

    model = ChestCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_samples = load_samples(TEST_DIR)
    loader = DataLoader(XrayDataset(test_samples), batch_size=32, shuffle=False)
    print(f"Evaluating on {len(test_samples)} test images...")

    num_classes = len(CLASSES)
    confusion = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            for true_label, pred_label in zip(y.tolist(), preds.tolist()):
                confusion[true_label][pred_label] += 1

    per_class = {}
    total_correct = 0
    total_samples = 0
    for i, class_name in enumerate(CLASSES):
        tp = confusion[i][i]
        fn = sum(confusion[i]) - tp
        fp = sum(confusion[r][i] for r in range(num_classes)) - tp
        support = sum(confusion[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[class_name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                                  "f1": round(f1, 4), "support": support}
        total_correct += tp
        total_samples += support

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    metrics = {"classes": CLASSES, "confusion_matrix": confusion, "per_class": per_class,
               "overall_accuracy": round(overall_accuracy, 4), "total_test_samples": total_samples}

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOverall accuracy: {overall_accuracy:.4f}")
    for class_name, m in per_class.items():
        print(f"  {class_name}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}")
    print(f"\nSaved metrics to {OUT_PATH}")


if __name__ == "__main__":
    main()
