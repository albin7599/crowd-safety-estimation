"""Configuration settings for IADL-CDE CCTV Crowd Safety Monitoring Web App."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = Path(os.environ.get("IADL_MODELS_DIR", BASE_DIR / "models"))
UPLOAD_DIR = Path(os.environ.get("IADL_UPLOAD_DIR", BASE_DIR / "uploads"))
RESULTS_DIR = Path(os.environ.get("IADL_RESULTS_DIR", BASE_DIR / "results"))

# Ensure storage directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Upload limits and validation
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB upload limit
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi"}

# Retention policy
CLEANUP_MAX_AGE_SECONDS = 24 * 3600  # 24 hours

# Analysis defaults
DEFAULT_CALIB_SECONDS = 10
CALIB_MIN_SECONDS = 0
CALIB_MAX_SECONDS = 120

# Model files
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_DENSITY_FILE = "density.pt"
DEFAULT_PREDICTOR_FILE = "predictor.pt"

CAMERA_CONFIG_FILE = "config_cam.json"
CAMERA_PREDICTOR_FILE = "predictor_cam.pt"

# Honesty notes (shown across UI)
HONESTY_NOTES = [
    {
        "title": "Camera Calibration & Domain Transfer",
        "detail": "Anomaly model was trained on UCSD Ped2 (fixed walkway camera, 10 fps). On other cameras it needs calibration, and ideally fine-tuning."
    },
    {
        "title": "Crowd Density Estimation",
        "detail": "Density classes: sparse under 85 people, medium 85 to 270, dense over 270 (head-count based, trained on ShanghaiTech)."
    },
    {
        "title": "Advisory Nature of Alerts",
        "detail": "Alerts are automated suggestions, not proof. Expect false alarms in complex lighting or camera angles."
    }
]
