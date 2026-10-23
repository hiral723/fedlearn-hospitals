"""
Reads the raw COVID-19 Radiography Database (downloaded via Kaggle CLI),
resizes every image, and writes a clean 90/10 train/test split organized
by class into data/processed/train/<class> and data/processed/test/<class>.

Run this once, from the project root:
    python data/prepare_data.py
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"
IMAGE_SIZE = (128, 128)
TEST_FRACTION = 0.10
SEED = 42

# The Kaggle dataset uses these folder names. If your download unpacked
# with different folder names, check data/raw/ and edit this dict to match.
CLASS_FOLDERS = {
    "COVID": "COVID",
    "Normal": "Normal",
    "Lung_Opacity": "Lung_Opacity",
    "Viral_Pneumonia": "Viral Pneumonia",
}


def find_images(class_dir: Path):
    """Handles both `raw/COVID/*.png` and `raw/COVID/images/*.png` layouts."""
    images_subdir = class_dir / "images"
    search_dir = images_subdir if images_subdir.exists() else class_dir
    exts = (".png", ".jpg", ".jpeg")
    return [p for p in search_dir.iterdir() if p.suffix.lower() in exts]


def main():
    random.seed(SEED)

    if not RAW_DIR.exists():
        raise SystemExit(
            f"Couldn't find {RAW_DIR}. Did you run the kaggle download + unzip "
            f"step from the README? Expected raw class folders inside data/raw/."
        )

    for split in ("train", "test"):
        for clean_name in CLASS_FOLDERS:
            (OUT_DIR / split / clean_name).mkdir(parents=True, exist_ok=True)

    total_train, total_test = 0, 0

    for clean_name, folder_name in CLASS_FOLDERS.items():
        class_dir = RAW_DIR / folder_name
        if not class_dir.exists():
            print(f"WARNING: expected folder '{folder_name}' not found in {RAW_DIR}, skipping.")
            continue

        images = find_images(class_dir)
        random.shuffle(images)

        n_test = int(len(images) * TEST_FRACTION)
        test_images = images[:n_test]
        train_images = images[n_test:]

        for split_name, split_images in (("train", train_images), ("test", test_images)):
            for img_path in tqdm(split_images, desc=f"{clean_name}/{split_name}"):
                try:
                    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                    out_path = OUT_DIR / split_name / clean_name / img_path.name
                    img.save(out_path)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

        total_train += len(train_images)
        total_test += len(test_images)
        print(f"{clean_name}: {len(train_images)} train, {len(test_images)} test")

    print(f"\nDone. Total: {total_train} train images, {total_test} test images.")
    print(f"Output written to {OUT_DIR}")


if __name__ == "__main__":
    main()