# IADL-CDE: CCTV Crowd Safety & Anomaly Detection Web Application

IADL-CDE is a real-time web application for CCTV crowd safety monitoring. It leverages a two-stage deep learning pipeline:
1. **Crowd Density Estimation**: Self-Modulating Convolutional Neural Network (SMCNN-v2 with ResNet-34 backbone) classifying crowd density into **Sparse**, **Medium**, and **Dense** classes.
2. **Spatial Anomaly Detection**: A U-Net spatio-temporal next-frame predictor trained to reconstruct normal pedestrian motion. Deviations between predicted and actual frames identify spatial anomalies (e.g., unauthorized vehicles, bicycles, skateboards, rapid irregular motion).

---

## Screenshot

```
+-----------------------------------------------------------------------------------------+
| [IADL-CDE] Crowd Safety System             📹 Analyze Video   ⚙️ Camera Adaptation  [CPU] |
+-----------------------------------------------------------------------------------------+
| [🔴 ANOMALY DETECTED] Reconstruction error (0.0384) exceeds safety threshold (0.0274)  |
|                                                                                         |
| +-----------------------------------------+  +----------------------------------------+ |
| |                                         |  | Real-time Telemetry     [DENSE (92%)]  | |
| |          LIVE ANNOTATED STREAM          |  | Score: 0.0384    | Threshold: 0.0274   | |
| |          [MJPEG Frame Output]           |  | Time: 14.2s      | Alert Boxes: 2      | |
| |                                         |  +----------------------------------------+ |
| |                                         |  | Alert Event Log            [1 Event]   | |
| +-----------------------------------------+  | #1 | 12.4s - 14.8s | 0.0384 | DENSE    | |
|                                              +----------------------------------------+ |
| +-----------------------------------------+  | ✅ Analysis Summary (Post-Processing)  | |
| | Anomaly Score vs Safety Threshold       |  | Density Dist: 10% Sp | 45% Med | 45% Dn| |
| | ~~~~~~ Score   - - - Threshold          |  | [ Download Video ]  [ Download CSV ]   | |
| +-----------------------------------------+  +----------------------------------------+ |
+-----------------------------------------------------------------------------------------+
```

---

## Key Features

- **Real-Time MJPEG Streaming**: Annotated video stream with bounding boxes, density tag, and visual anomaly indicator.
- **Dynamic Density HUD**: High-visibility color-coded badges:
  - 🟢 **Sparse**: Under 85 people
  - 🟠 **Medium**: 85 to 270 people
  - 🔴 **Dense**: Over 270 people
- **Adaptive Safety Thresholding**: Supports on-the-fly camera calibration or pre-saved baseline thresholds from `config.json`.
- **Live Anomaly Timeline (Chart.js)**: Real-time dual-line plot of frame anomaly score vs adaptive threshold.
- **Smart Event Merging**: Merges anomaly alerts occurring closer than 2.0 seconds into unified events with peak score tracking.
- **Post-Run Analytics & Export**: Summary breakdown of crowd density percentages, total alert events, alert frame percentage, annotated MP4 download, and detailed CSV telemetry log.
- **Single-Camera Fine-Tuning (`/camera`)**: Ingests 30–60s normal-only footage to adapt the U-Net predictor to specific camera angles and lighting without overwriting base model weights.

---

## Directory Structure

```
.
├── app.py                  # Flask web server & route handlers
├── config.py               # Centralized configuration & path definitions
├── jobs.py                 # Background worker queue, telemetry & model lifecycle
├── iadl_core.py            # Neural network models & analysis generator (unmodified)
├── requirements.txt        # Python package dependencies
├── models/
│   ├── config.json         # Base UCSD Ped2 model configuration & thresholds
│   ├── density.pt          # SMCNN-v2 crowd density weights
│   └── predictor.pt        # Spatio-temporal next-frame predictor weights
├── templates/
│   ├── base.html           # Unified navigation, health check, honesty notes
│   ├── index.html          # Drag-and-drop video upload page
│   ├── watch.html          # Live monitoring dashboard with Chart.js & metrics
│   └── camera.html         # Camera-specific fine-tuning & model switcher
├── static/
│   ├── css/
│   │   └── style.css       # Dark-mode dashboard styles & animations
│   └── js/
│       ├── upload.js       # File dropzone & client validation
│       ├── watch.js        # Polling telemetry & Chart.js live updates
│       └── camera.js       # Fine-tuning runner & model selector
├── tests/
│   └── test_app.py         # Automated test suite
└── uploads/ / results/     # Auto-created media directories (24h retention)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- PyTorch & Torchvision
- OpenCV & Flask

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model Weights
Ensure `./models/` contains `density.pt`, `predictor.pt`, and `config.json`.

---

## Running the Application

Start the Flask application:
```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | GET | Upload page with drag-and-drop and calibration settings |
| `POST /upload` | POST | Uploads video file, enqueues background job, returns `{job_id}` |
| `GET /watch/<id>` | GET | Live telemetry and stream dashboard for a job |
| `GET /stream/<id>` | GET | Live MJPEG multipart stream of annotated frames |
| `GET /stats/<id>` | GET | JSON object containing latest stats, alert event log, and status |
| `GET /stats/<id>/csv` | GET | Download per-frame CSV telemetry log |
| `GET /result/<id>` | GET | Download final annotated MP4 video |
| `GET /camera` | GET | Camera fine-tuning interface |
| `POST /camera/finetune` | POST | Triggers background adaptation on normal camera footage |
| `GET /camera/status` | GET | Returns fine-tuning state and available models |
| `POST /camera/switch` | POST | Toggles active model between default and camera-adapted |
| `GET /health` | GET | Returns JSON with hardware device (CUDA/CPU) and model status |

---

## Running Automated Tests

Run the unit tests:
```bash
python -m unittest discover -s tests -v
```

---

## Operational Notes & Limitations

1. **UCSD Ped2 Domain**: The base anomaly detector is trained on UCSD Ped2 (fixed walkway, 10 fps). On different camera perspectives, lighting, or framerates, use calibration (`calib_seconds > 0`) or fine-tune via `/camera`.
2. **Crowd Counts**: Density categories are derived from ShanghaiTech head-count distributions: Sparse (<85), Medium (85–270), Dense (>270).
3. **Advisory Alerts**: Alerts represent statistical anomalies in motion and appearance; they should be treated as automated suggestions rather than definitive proof of hazards.
