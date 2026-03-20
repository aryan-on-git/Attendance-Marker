import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
MODEL_PATH = ROOT / "models" / "face_classifier.h5"
LABELS_PATH = ROOT / "models" / "labels.json"
IMG_SIZE = (96, 96)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_samples(base_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    if not base_dir.exists():
        return samples

    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                samples.append((image_path, class_dir.name))
    return samples


def load_features(samples: list[tuple[Path, str]], image_size: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    x_data: list[np.ndarray] = []
    y_labels: list[str] = []

    for image_path, label in samples:
        try:
            img = Image.open(image_path).convert("RGB").resize(image_size)
        except Exception:
            continue

        arr = np.asarray(img, dtype=np.float32) / 255.0
        x_data.append(arr.reshape(-1))
        y_labels.append(label)

    if not x_data:
        return np.empty((0, image_size[0] * image_size[1] * 3), dtype=np.float32), []

    return np.vstack(x_data), y_labels


def save_model(
    model_path: Path,
    image_size: tuple[int, int],
    classes: np.ndarray,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(model_path, "w") as h5f:
        h5f.attrs["model_type"] = "sklearn_logistic_regression"
        h5f.attrs["image_size"] = json.dumps(image_size)
        h5f.create_dataset("classes", data=classes.astype("S"))
        h5f.create_dataset("scaler_mean", data=scaler.mean_.astype(np.float32))
        h5f.create_dataset("scaler_scale", data=scaler.scale_.astype(np.float32))
        h5f.create_dataset("coef", data=classifier.coef_.astype(np.float32))
        h5f.create_dataset("intercept", data=classifier.intercept_.astype(np.float32))


def main() -> None:
    train_samples = collect_samples(TRAIN_DIR)
    if not train_samples:
        raise SystemExit(f"No training images found in {TRAIN_DIR}")

    x_train, y_train_labels = load_features(train_samples, IMG_SIZE)
    if x_train.size == 0:
        raise SystemExit("Training images could not be decoded.")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    labels = {name: idx for idx, name in enumerate(label_encoder.classes_)}

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"Label mapping saved: {labels}")

    if len(label_encoder.classes_) < 2:
        raise SystemExit("Need at least 2 classes with valid images to train a classifier.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    classifier.fit(x_train_scaled, y_train)

    train_preds = classifier.predict(x_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Train accuracy: {train_acc:.4f}")

    val_samples = collect_samples(VAL_DIR)
    if val_samples:
        x_val, y_val_labels = load_features(val_samples, IMG_SIZE)
        class_to_idx = {name: idx for idx, name in enumerate(label_encoder.classes_)}

        keep_indices = [
            idx for idx, label in enumerate(y_val_labels) if label in class_to_idx
        ]
        if keep_indices:
            x_val_kept = x_val[keep_indices]
            y_val = np.array([class_to_idx[y_val_labels[idx]] for idx in keep_indices], dtype=np.int64)
            x_val_scaled = scaler.transform(x_val_kept)
            val_preds = classifier.predict(x_val_scaled)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"Val accuracy: {val_acc:.4f}")
        else:
            print("Val set contains no classes present in train set; skipping val accuracy.")
    else:
        print("No validation images found; skipping val accuracy.")

    save_model(
        model_path=MODEL_PATH,
        image_size=IMG_SIZE,
        classes=label_encoder.classes_,
        scaler=scaler,
        classifier=classifier,
    )
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
