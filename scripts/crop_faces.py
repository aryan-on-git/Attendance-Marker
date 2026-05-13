from pathlib import Path
import argparse

import cv2

try:
    from mtcnn import MTCNN
except Exception:  # pragma: no cover - fallback when MTCNN or TF is unavailable
    MTCNN = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "dataset"
DEFAULT_OUTPUT_DIR = ROOT / "processed_dataset"
IMAGE_SIZE = (224, 224)


def clamp_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(0, w)
    h = max(0, h)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    return x, y, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop faces from dataset images.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input dataset folder (default: dataset/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for cropped faces (default: processed_dataset/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output

    if not input_dir.exists():
        raise SystemExit(f"Missing input dataset folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detector = MTCNN() if MTCNN is not None else None
    cascade = None
    if detector is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)

    for person_dir in sorted(input_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        save_path = output_dir / person_dir.name
        save_path.mkdir(parents=True, exist_ok=True)

        saved = 0
        for img_path in sorted(person_dir.iterdir()):
            if not img_path.is_file():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if detector is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb)
                if not faces:
                    continue
                x, y, w, h = faces[0]["box"]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) == 0:
                    continue
                x, y, w, h = faces[0]
            img_h, img_w = img.shape[:2]
            x1, y1, x2, y2 = clamp_box(x, y, w, h, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                continue

            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, IMAGE_SIZE)
            cv2.imwrite(str(save_path / img_path.name), face)
            saved += 1

        print(f"{person_dir.name}: saved {saved} cropped faces")

    print("Face cropping complete.")


if __name__ == "__main__":
    main()
