from pathlib import Path

import cv2
from mtcnn import MTCNN


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "processed_dataset"
IMAGE_SIZE = (224, 224)


def clamp_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(0, w)
    h = max(0, h)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    return x, y, x2, y2


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(f"Missing input dataset folder: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = MTCNN()

    for person_dir in sorted(INPUT_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        save_path = OUTPUT_DIR / person_dir.name
        save_path.mkdir(parents=True, exist_ok=True)

        saved = 0
        for img_path in sorted(person_dir.iterdir()):
            if not img_path.is_file():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if not faces:
                continue

            x, y, w, h = faces[0]["box"]
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
