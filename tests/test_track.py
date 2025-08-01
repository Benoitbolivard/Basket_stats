"""
Test factice pour le module de tracking DeepSORT.
"""
import json
import types
from pathlib import Path

import vision.track as track


def fake_run_tracking(
    detections_path: str, video_path: str, output_path: str | Path
) -> None:
    """Fausse fonction de tracking pour les tests."""
    data = {
        "0": [
            {"object_id": 0, "centroid": [100, 100], "disappeared": 0},
            {"object_id": 1, "centroid": [200, 200], "disappeared": 0},
        ],
        "1": [
            {"object_id": 0, "centroid": [105, 105], "disappeared": 0},
            {"object_id": 1, "centroid": [205, 205], "disappeared": 0},
        ],
    }
    Path(output_path).write_text(json.dumps(data), encoding="utf-8")


def test_tracking_monkeypatch(tmp_path) -> None:
    """Test du tracking avec monkey-patch."""
    # Remplace la vraie fonction par la fausse
    track.run_tracking = types.FunctionType(
        fake_run_tracking.__code__, globals(), "run_tracking"
    )

    out = tmp_path / "tracked.json"
    track.run_tracking("dummy_detections.json", "dummy_video.mp4", out)

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert "0" in parsed
    assert "1" in parsed
    assert len(parsed["0"]) == 2
    assert parsed["0"][0]["object_id"] == 0
