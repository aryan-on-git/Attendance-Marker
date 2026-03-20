from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "videos"
DATASET_DIR = ROOT / "dataset"
IMAGE_SIZE = (224, 224)
FRAME_SKIP = 5
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_DIR.exists():
        raise SystemExit(f"Missing video folder: {VIDEO_DIR}")

    video_files = [p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    if not video_files:
        raise SystemExit(f"No videos found in {VIDEO_DIR}")

    for video_file in sorted(video_files):
        name = video_file.stem.lower().strip()
        person_folder = DATASET_DIR / name
        person_folder.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_file}")
            continue

        frame_id = 0
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:
                img = cv2.resize(frame, IMAGE_SIZE)
                out_path = person_folder / f"{count:04d}.jpg"
                cv2.imwrite(str(out_path), img)
                count += 1

            frame_id += 1

        cap.release()
        print(f"Extracted {count} frames for {name}")

    print("All videos processed.")


if __name__ == "__main__":
    main()
