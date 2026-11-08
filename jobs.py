"""Job management, background worker queues, alert merging, and model lifecycle."""
import os
import time
import uuid
import json
import shutil
import threading
import subprocess
from queue import Queue
from pathlib import Path
import cv2
import torch

from config import (
    MODELS_DIR,
    UPLOAD_DIR,
    RESULTS_DIR,
    CLEANUP_MAX_AGE_SECONDS,
    DEFAULT_CONFIG_FILE,
    DEFAULT_DENSITY_FILE,
    DEFAULT_PREDICTOR_FILE,
    CAMERA_CONFIG_FILE,
    CAMERA_PREDICTOR_FILE
)
from iadl_core import Engine, VideoAnalyzer, finetune_camera, CLASSES

# Global Engine singleton and state
_engine_lock = threading.Lock()
_engine = None
_active_model = "default"  # "default" or "camera"


def get_engine():
    global _engine, _active_model
    with _engine_lock:
        if _engine is None:
            _load_engine_unlocked("default")
        return _engine


def get_active_model_info():
    with _engine_lock:
        eng = _engine
        cam_config_exists = (MODELS_DIR / CAMERA_CONFIG_FILE).exists()
        cam_pred_exists = (MODELS_DIR / CAMERA_PREDICTOR_FILE).exists()
        cam_available = cam_config_exists and cam_pred_exists
        return {
            "device": str(eng.dev) if eng else "unknown",
            "model_loaded": eng is not None,
            "active_model": _active_model,
            "camera_model_available": cam_available,
            "config": eng.cfg if eng else {}
        }


def _load_engine_unlocked(model_type="default"):
    global _engine, _active_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg_filename = CAMERA_CONFIG_FILE if model_type == "camera" else DEFAULT_CONFIG_FILE
    cfg_path = MODELS_DIR / cfg_filename
    
    if not cfg_path.exists():
        if model_type == "camera":
            raise FileNotFoundError("Camera model configuration not found. Please fine-tune first.")
        else:
            raise FileNotFoundError(f"Base config file not found at {cfg_path}")
    
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    from iadl_core import DensityNet, UNet
    den = DensityNet(pretrained=False)
    density_path = MODELS_DIR / DEFAULT_DENSITY_FILE
    den.load_state_dict(torch.load(density_path, map_location=device))
    
    pred_file = cfg.get("predictor_file", CAMERA_PREDICTOR_FILE if model_type == "camera" else DEFAULT_PREDICTOR_FILE)
    pred_path = MODELS_DIR / pred_file
    
    pr = UNet()
    pr.load_state_dict(torch.load(pred_path, map_location=device))
    
    _engine = Engine(den, pr, cfg, device)
    _active_model = model_type
    return _engine


def switch_engine_model(model_type="default"):
    with _engine_lock:
        return _load_engine_unlocked(model_type)


# Job representation
class AnalysisJob:
    def __init__(self, job_id, video_name, input_path, calib_seconds=10):
        self.id = job_id
        self.video_name = video_name
        self.input_path = input_path
        self.output_path = str(RESULTS_DIR / f"{job_id}_annotated.mp4")
        self.calib_seconds = calib_seconds
        
        self.status = "queued"  # queued, running, done, error
        self.error_message = None
        self.created_at = time.time()
        self.start_time = None
        self.finish_time = None
        
        self.progress_percent = 0.0
        self.latest = None
        self.stats_history = []
        self.alert_events = []
        self.summary = None
        
        self.current_frame_jpg = None
        self.frame_condition = threading.Condition()
        self.fps = 10.0
        self.total_input_frames = 0

    def to_dict(self):
        return {
            "id": self.id,
            "video_name": self.video_name,
            "status": self.status,
            "calib_seconds": self.calib_seconds,
            "progress_percent": self.progress_percent,
            "latest": self.latest,
            "alert_events": self.alert_events,
            "summary": self.summary,
            "error_message": self.error_message,
            "has_result_file": (self.status == "done" and os.path.exists(self.output_path))
        }


# Global store for jobs and queue
_jobs = {}
_jobs_lock = threading.Lock()
_job_queue = Queue()

# Fine-tuning job state
_finetune_state = {
    "status": "idle",  # idle, running, done, error
    "message": "Ready to fine-tune",
    "progress": 0,
    "result": None,
    "error": None
}
_finetune_lock = threading.Lock()


def register_job(video_name, input_path, calib_seconds=10):
    job_id = uuid.uuid4().hex[:12]
    job = AnalysisJob(job_id, video_name, input_path, calib_seconds)
    with _jobs_lock:
        _jobs[job_id] = job
    _job_queue.put(job_id)
    return job


def get_job(job_id):
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs():
    with _jobs_lock:
        return [j.to_dict() for j in _jobs.values()]


def get_finetune_status():
    with _finetune_lock:
        return dict(_finetune_state)


def set_finetune_status(status, message, progress=0, result=None, error=None):
    with _finetune_lock:
        _finetune_state["status"] = status
        _finetune_state["message"] = message
        _finetune_state["progress"] = progress
        if result is not None:
            _finetune_state["result"] = result
        if error is not None:
            _finetune_state["error"] = error


def _cleanup_old_files():
    """Remove files in UPLOAD_DIR and RESULTS_DIR older than 24 hours."""
    now = time.time()
    for directory in (UPLOAD_DIR, RESULTS_DIR):
        try:
            for item in directory.glob("*"):
                if item.is_file():
                    if now - item.stat().st_mtime > CLEANUP_MAX_AGE_SECONDS:
                        try:
                            item.unlink()
                        except Exception:
                            pass
        except Exception:
            pass


def _process_job(job):
    job.status = "running"
    job.start_time = time.time()
    
    cap = cv2.VideoCapture(job.input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    job.total_input_frames = total_frames
    
    eng = get_engine()
    analyzer = VideoAnalyzer(eng, calib_seconds=job.calib_seconds)
    
    tmp_out = job.output_path + ".tmp.mp4"
    writer = None
    counts = {"sparse": 0, "medium": 0, "dense": 0}
    alert_frames_count = 0
    
    try:
        for vis, st in analyzer.run(job.input_path):
            t0 = time.time()
            if writer is None:
                h, w = vis.shape[:2]
                out_fps = st.get("out_fps", 10.0)
                job.fps = out_fps
                writer = cv2.VideoWriter(
                    tmp_out,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    out_fps,
                    (w, h)
                )
            
            writer.write(vis)
            
            # Encode frame to JPEG for MJPEG stream
            ok, jpg_bytes = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with job.frame_condition:
                    job.current_frame_jpg = jpg_bytes.tobytes()
                    job.frame_condition.notify_all()
            
            # Update stats
            job.latest = st
            job.stats_history.append(st)
            
            density_label = st.get("density", "sparse")
            counts[density_label] = counts.get(density_label, 0) + 1
            
            if st.get("alert"):
                alert_frames_count += 1
                cur_time = round(float(st["time"]), 2)
                cur_score = round(float(st["score"]), 4)
                
                # Merge alerts closer than 2 seconds into one event
                if job.alert_events and (cur_time - job.alert_events[-1]["end_time"] < 2.0):
                    ev = job.alert_events[-1]
                    ev["end_time"] = cur_time
                    if cur_score > ev["peak_score"]:
                        ev["peak_score"] = cur_score
                        ev["density"] = density_label
                else:
                    job.alert_events.append({
                        "start_time": cur_time,
                        "end_time": cur_time,
                        "peak_score": cur_score,
                        "density": density_label
                    })
            
            # Progress calculation
            if total_frames > 0:
                frame_idx = st.get("frame", len(job.stats_history))
                pct = min(99.0, round((frame_idx / total_frames) * 100, 1))
                job.progress_percent = pct
            
            # Real-time yield / pacing
            target_step = 1.0 / max(1.0, st.get("out_fps", 10.0))
            elapsed = time.time() - t0
            if elapsed < target_step:
                time.sleep(target_step - elapsed)
        
        # Finalize video file
        if writer is not None:
            writer.release()
            writer = None
            
            # Convert to web-friendly H.264 if ffmpeg is available
            converted = False
            try:
                res = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_out,
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p", job.output_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if os.path.exists(job.output_path):
                    try:
                        os.remove(tmp_out)
                    except Exception:
                        pass
                    converted = True
            except Exception:
                converted = False
            
            if not converted:
                if os.path.exists(tmp_out):
                    if os.path.exists(job.output_path):
                        os.remove(job.output_path)
                    os.replace(tmp_out, job.output_path)
        
        # Summary computation
        total_st = len(job.stats_history)
        if total_st > 0:
            summary = {
                "total_frames": total_st,
                "density_percentages": {
                    "sparse": round((counts["sparse"] / total_st) * 100, 1),
                    "medium": round((counts["medium"] / total_st) * 100, 1),
                    "dense": round((counts["dense"] / total_st) * 100, 1),
                },
                "alert_events_count": len(job.alert_events),
                "alert_frames_percent": round((alert_frames_count / total_st) * 100, 1),
                "duration_seconds": job.stats_history[-1].get("time", 0.0)
            }
        else:
            summary = {
                "total_frames": 0,
                "density_percentages": {"sparse": 0.0, "medium": 0.0, "dense": 0.0},
                "alert_events_count": 0,
                "alert_frames_percent": 0.0,
                "duration_seconds": 0.0
            }
        
        job.summary = summary
        job.progress_percent = 100.0
        job.status = "done"
        job.finish_time = time.time()
        
    except Exception as e:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass
        job.status = "error"
        job.error_message = str(e)
    finally:
        with job.frame_condition:
            job.frame_condition.notify_all()


def _worker_loop():
    while True:
        try:
            _cleanup_old_files()
            job_id = _job_queue.get()
            job = get_job(job_id)
            if job and job.status == "queued":
                _process_job(job)
            _job_queue.task_done()
        except Exception as e:
            time.sleep(1)


# Start background worker thread
_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


# Background camera fine-tuning runner
def run_camera_finetune(normal_clip_path, seconds=40, epochs=15):
    def _task():
        try:
            set_finetune_status("running", "Extracting normal frames and training next-frame predictor...", progress=20)
            eng = get_engine()
            
            # Run finetune_camera from iadl_core
            pix_thr, frame_thr = finetune_camera(
                eng,
                normal_clip_path,
                seconds=seconds,
                epochs=epochs,
                target_fps=10.0
            )
            
            set_finetune_status("running", "Saving camera weights and configuration...", progress=85)
            
            # Save predictor_cam.pt
            cam_pred_path = MODELS_DIR / CAMERA_PREDICTOR_FILE
            torch.save(eng.pred.state_dict(), cam_pred_path)
            
            # Save config_cam.json without overwriting config.json
            cam_cfg = dict(eng.cfg, pix_thr=pix_thr, frame_thr=frame_thr, predictor_file=CAMERA_PREDICTOR_FILE)
            with open(MODELS_DIR / CAMERA_CONFIG_FILE, "w") as f:
                json.dump(cam_cfg, f, indent=2)
            
            set_finetune_status(
                "done",
                f"Camera adaptation complete! New thresholds: Pixel={pix_thr:.5f}, Frame={frame_thr:.5f}",
                progress=100,
                result={"pixel_thr": pix_thr, "frame_thr": frame_thr}
            )
        except Exception as e:
            set_finetune_status("error", f"Fine-tuning failed: {str(e)}", progress=0, error=str(e))
    
    t = threading.Thread(target=_task, daemon=True)
    t.start()
