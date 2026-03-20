import json
from pathlib import Path

import cv2
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "face_classifier.h5"
LABELS_PATH = ROOT / "models" / "labels.json"
CONFIDENCE_THRESHOLD = 0.55


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exps = np.exp(x)
    return exps / np.sum(exps)


def load_classifier(model_path: Path) -> tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(model_path, "r") as h5f:
        image_size = tuple(json.loads(h5f.attrs["image_size"]))
        scaler_mean = h5f["scaler_mean"][:]
        scaler_scale = h5f["scaler_scale"][:]
        coef = h5f["coef"][:]
        intercept = h5f["intercept"][:]

    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    return image_size, scaler_mean, scaler_scale, coef, intercept


def predict_proba(
    feature_vector: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    x = (feature_vector - scaler_mean) / scaler_scale

    if coef.shape[0] == 1 and num_classes == 2:
        score = float(np.dot(coef[0], x) + intercept[0])
        p1 = float(sigmoid(np.array([score]))[0])
        return np.array([1.0 - p1, p1], dtype=np.float32)

    logits = coef @ x + intercept
    return softmax(logits).astype(np.float32)


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise SystemExit(f"Labels file not found: {LABELS_PATH}")

    with LABELS_PATH.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    idx_to_name = {int(v): k for k, v in labels.items()}
    num_classes = len(idx_to_name)
    if num_classes < 2:
        raise SystemExit("At least 2 classes are required for recognition.")

    image_size, scaler_mean, scaler_scale, coef, intercept = load_classifier(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise SystemExit("Failed to load Haar cascade face detector.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
            )

            for (x, y, w, h) in faces:
                face = frame[y : y + h, x : x + w]
                if face.size == 0:
                    continue

                face = cv2.resize(face, image_size)
                face = face.astype(np.float32) / 255.0
                feature_vector = face.reshape(-1)

                probs = predict_proba(
                    feature_vector=feature_vector,
                    scaler_mean=scaler_mean,
                    scaler_scale=scaler_scale,
                    coef=coef,
                    intercept=intercept,
                    num_classes=num_classes,
                )

                class_id = int(np.argmax(probs))
                confidence = float(np.max(probs))
                name = idx_to_name.get(class_id, "Unknown")

                if confidence < CONFIDENCE_THRESHOLD:
                    display_name = "Unknown"
                else:
                    display_name = name

                text = f"{display_name} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
