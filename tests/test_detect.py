"""
Ce test ne lance PAS réellement YOLO (trop lourd pour la CI).
On monkey-patch vision.detect.run_detection afin de créer un
fichier detections.json factice, puis on vérifie son contenu.
"""
import json
import types
from pathlib import Path

import vision.detect as detect


def fake_run_detection(source: str, output: str | Path) -> None:  # noqa: D401
    data = [
        {"frame": 0, "class_id": 0, "conf": 0.9, "bbox": [0, 0, 10, 10]},
        {"frame": 0, "class_id": 0, "conf": 0.8, "bbox": [20, 20, 30, 30]},
    ]
    Path(output).write_text(json.dumps(data), encoding="utf-8")


def test_detection_monkeypatch(tmp_path) -> None:
    # monkey-patch
    detect.run_detection = types.FunctionType(
        fake_run_detection.__code__, globals(), "run_detection"
    )
    out = tmp_path / "detections.json"
    detect.run_detection("dummy.mp4", out)

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert len(parsed) == 2
    assert parsed[0]["bbox"] == [0, 0, 10, 10]
