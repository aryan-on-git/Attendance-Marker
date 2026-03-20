import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "processed_dataset"
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
SPLIT_RATIO = 0.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def reset_class_folder(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Missing processed dataset folder: {SOURCE_DIR}")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    for person_dir in sorted(SOURCE_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        images = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        random.shuffle(images)

        if not images:
            print(f"{person_dir.name}: skipped (no images)")
            continue

        if len(images) == 1:
            split_index = 1
        else:
            split_index = int(len(images) * SPLIT_RATIO)
            split_index = max(1, min(len(images) - 1, split_index))

        train_images = images[:split_index]
        val_images = images[split_index:]

        train_person_dir = TRAIN_DIR / person_dir.name
        val_person_dir = VAL_DIR / person_dir.name
        reset_class_folder(train_person_dir)
        reset_class_folder(val_person_dir)

        for img in train_images:
            shutil.copy2(img, train_person_dir / img.name)

        for img in val_images:
            shutil.copy2(img, val_person_dir / img.name)

        print(f"{person_dir.name}: {len(train_images)} train, {len(val_images)} val")

    print("Dataset split completed.")


if __name__ == "__main__":
    main()
