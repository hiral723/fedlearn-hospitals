"""
CNN architecture for 4-class chest X-ray classification, plus a Dataset
class that can load only a subset of images (used to hand each simulated
hospital its own slice of the training data).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from config import CLASSES, NUM_CLASSES

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class XrayDataset(Dataset):
    """Wraps a list of (filepath, label_index) pairs."""

    def __init__(self, samples, transform=TRANSFORM):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


class ChestCNN(nn.Module):
    """5 conv layers + 2 FC layers, mirrors the reference architecture
    used in the base paper (Cetinkaya et al.)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.1),
            )

        self.features = nn.Sequential(
            conv_block(3, 16, 7),
            conv_block(16, 32, 5),
            conv_block(32, 64, 3),
            conv_block(64, 128, 3),
            conv_block(128, 128, 3),
        )
        # 128x128 input -> after 5x stride-2 pooling -> 4x4 spatial
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")