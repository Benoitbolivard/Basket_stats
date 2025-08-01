import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Basket Stats API",
    description="API for basketball statistics analysis",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Hello world"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "basket-stats-api"}


@app.get("/api/v1/health")
async def api_health():
    """API health check endpoint"""
    return {"status": "healthy", "api_version": "v1"}


@app.post("/api/v1/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """Analyze basketball video with detection and tracking."""
    if not video.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Video format not supported")

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        video_path = temp_path / video.filename
        detections_path = temp_path / "detections.json"
        tracked_path = temp_path / "tracked.json"

        # Save uploaded video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        try:
            # Run detection
            subprocess.run(
                [
                    "poetry",
                    "run",
                    "python",
                    "-m",
                    "vision.detect",
                    "--source",
                    str(video_path),
                    "--output",
                    str(detections_path),
                ],
                check=True,
                capture_output=True,
            )

            # Run tracking
            subprocess.run(
                [
                    "poetry",
                    "run",
                    "python",
                    "-m",
                    "vision.track",
                    "--detections",
                    str(detections_path),
                    "--video",
                    str(video_path),
                    "--output",
                    str(tracked_path),
                ],
                check=True,
                capture_output=True,
            )

            # Load results
            with open(detections_path, "r") as f:
                detections = json.load(f)

            with open(tracked_path, "r") as f:
                tracked = json.load(f)

            return {
                "status": "success",
                "detections_count": len(detections),
                "frames_tracked": len(tracked),
                "filename": video.filename,
            }

        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500, detail=f"Analysis failed: {e.stderr.decode()}"
            )


@app.get("/api/v1/status")
async def get_status():
    """Get analysis status and capabilities."""
    return {
        "status": "ready",
        "capabilities": ["video_detection", "object_tracking", "basketball_analysis"],
        "supported_formats": [".mp4", ".avi", ".mov"],
    }
