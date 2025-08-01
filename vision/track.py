"""
DeepSORT tracking for basketball objects.
Usage:
    poetry run python -m vision.track \
        --detections detections.json --video input.mp4 --output tracked.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


class DeepSORTTracker:
    """Simple DeepSORT-like tracker using OpenCV."""

    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        self.next_object_id = 0
        self.objects = {}  # {object_id: (centroid, disappeared_count)}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: tuple[int, int]) -> int:
        """Register a new object."""
        self.objects[self.next_object_id] = (centroid, 0)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id: int) -> None:
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: list[list[int]]) -> dict[int, tuple[int, int]]:
        """Update tracked objects with new detections."""
        if len(rects) == 0:
            # No objects detected, increment disappeared count
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Initialize centroids array
        centroids = [self._get_centroid(rect) for rect in rects]

        # If no objects tracked, register all
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i])
        else:
            # Get centroids of tracked objects
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[object_id][0] for object_id in object_ids]

            # Compute distances between existing centroids and new centroids
            distances = self._compute_distances(object_centroids, centroids)

            # Find smallest distances
            rows = self._argsort([min(row) for row in distances])
            cols = [row.index(min(row)) for row in distances]

            used_rows = set()
            used_cols = set()

            # Loop over the combination of the (row, column) index tuples
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distances[row][col] > self.max_distance:
                    continue

                # Update object
                object_id = object_ids[row]
                self.objects[object_id] = (centroids[col], 0)
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unused rows and columns
            unused_rows = set(range(len(distances))).difference(used_rows)
            unused_cols = set(range(len(distances[0]))).difference(used_cols)

            # If more objects than centroids, check if some disappeared
            if len(distances) >= len(distances[0]):
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More centroids than objects, register new ones
                for col in unused_cols:
                    self.register(centroids[col])

        return self.objects

    def _get_centroid(self, rect: list[int]) -> tuple[int, int]:
        """Get centroid from bounding box."""
        x1, y1, x2, y2 = rect
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _compute_distances(
        self, centroids1: list[tuple[int, int]], centroids2: list[tuple[int, int]]
    ) -> list[list[float]]:
        """Compute Euclidean distances between centroids."""
        distances = []
        for c1 in centroids1:
            row = []
            for c2 in centroids2:
                dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                row.append(dist)
            distances.append(row)
        return distances

    def _argsort(self, seq: list[float]) -> list[int]:
        """Return indices that would sort a list."""
        return sorted(range(len(seq)), key=seq.__getitem__)


def run_tracking(
    detections_path: str | Path, video_path: str | Path, output_path: str | Path
) -> None:
    """Run tracking on detected objects."""
    # Load detections
    with open(detections_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    # Group detections by frame
    frames = {}
    for det in detections:
        frame = det["frame"]
        if frame not in frames:
            frames[frame] = []
        frames[frame].append(det["bbox"])

    # Initialize tracker
    tracker = DeepSORTTracker()
    tracked_objects = {}

    # Process each frame
    for frame_num in sorted(frames.keys()):
        bboxes = frames[frame_num]
        objects = tracker.update(bboxes)

        # Store tracked objects for this frame
        tracked_objects[frame_num] = []
        for object_id, (centroid, disappeared) in objects.items():
            tracked_objects[frame_num].append(
                {
                    "object_id": object_id,
                    "centroid": centroid,
                    "disappeared": disappeared,
                }
            )

    # Save results
    Path(output_path).write_text(
        json.dumps(tracked_objects, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="Path to detections.json")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="tracked.json", help="Output file path")
    args = parser.parse_args()

    run_tracking(args.detections, args.video, args.output)


if __name__ == "__main__":
    main()
