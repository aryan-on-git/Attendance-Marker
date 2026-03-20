import re
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
IMAGE_SIZE = (224, 224)


def sanitize_name(raw_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name.strip().lower())
    return cleaned.strip("_")


def main() -> None:
    name = sanitize_name(input("Enter student name: "))
    if not name:
        raise SystemExit("Student name cannot be empty.")

    dataset_path = DATASET_DIR / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise SystemExit("Failed to load Haar cascade face detector.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    count = len(list(dataset_path.glob("*.jpg")))
    print("Press 's' to save image, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            largest_face = None

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if largest_face is None or (w * h) > (largest_face[2] * largest_face[3]):
                    largest_face = (x, y, w, h)

            cv2.imshow("Face Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and largest_face is not None:
                x, y, w, h = largest_face
                face = frame[y : y + h, x : x + w]
                face_img = cv2.resize(face, IMAGE_SIZE)
                out_path = dataset_path / f"{count:04d}.jpg"
                cv2.imwrite(str(out_path), face_img)
                print(f"Saved image {count} -> {out_path}")
                count += 1
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()