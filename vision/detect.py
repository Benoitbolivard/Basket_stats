"""
Minimal YOLOv8 detection CLI.
Usage:
    poetry run python -m vision.detect --source data/video.mp4 \\
        --output detections.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def run_detection(source: str | Path, output: str | Path) -> None:
    model = YOLO("yolov8n.pt")  # tiny model (~6 Mo)
    results = model(source, save=False, stream=True)

    detections: list[dict] = []
    for r in results:  # type: ignore[assignment]
        for box in r.boxes:
            detections.append(
                {
                    "frame": int(r.path.stem),
                    "class_id": int(box.cls),
                    "conf": float(box.conf),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()],
                }
            )

    with Path(output).open("w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, help="video or image folder"
    )
    parser.add_argument("--output", default="detections.json")
    args = parser.parse_args()

    run_detection(args.source, args.output)


if __name__ == "__main__":
    main()
