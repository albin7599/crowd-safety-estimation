"""Flask Application for IADL-CDE CCTV Crowd Safety Monitoring."""
import os
import io
import csv
import time
from pathlib import Path
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file,
    Response,
    redirect,
    url_for,
    abort
)
from werkzeug.utils import secure_filename

import config
from jobs import (
    get_engine,
    get_active_model_info,
    switch_engine_model,
    register_job,
    get_job,
    list_jobs,
    run_camera_finetune,
    get_finetune_status
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Pre-load Engine at startup
try:
    _eng = get_engine()
    print(f"[IADL-CDE] Successfully loaded Engine on device: {_eng.dev}")
except Exception as e:
    print(f"[IADL-CDE] Warning: Engine initialization failed: {e}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.context_processor
def inject_global_data():
    return {
        "honesty_notes": config.HONESTY_NOTES,
        "model_info": get_active_model_info()
    }


# ----------------------------------------------------------------- Web Pages
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/watch/<job_id>")
def watch(job_id):
    job = get_job(job_id)
    if not job:
        abort(404, description=f"Job {job_id} not found")
    return render_template("watch.html", job=job)


@app.route("/camera")
def camera():
    return render_template("camera.html")


# ----------------------------------------------------------------- API Endpoints
@app.route("/health")
def health():
    info = get_active_model_info()
    return jsonify({
        "status": "ok" if info["model_loaded"] else "degraded",
        "device": info["device"],
        "model_loaded": info["model_loaded"],
        "active_model": info["active_model"],
        "camera_model_available": info["camera_model_available"]
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": "No video file provided"}), 400
        return redirect(url_for("index"))
    
    file = request.files["video"]
    if not file or not file.filename:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": "Empty filename"}), 400
        return redirect(url_for("index"))
    
    if not allowed_file(file.filename):
        err = f"Unsupported file type. Allowed formats: {', '.join(config.ALLOWED_EXTENSIONS)}"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": err}), 400
        return render_template("index.html", error=err)
    
    # Calibration parameter extraction
    calib_mode = request.form.get("calib_normal")  # checkbox
    calib_input = request.form.get("calib_seconds", str(config.DEFAULT_CALIB_SECONDS))
    
    try:
        calib_val = int(calib_input)
        if calib_mode == "false" or calib_mode is None and "calib_normal" in request.form:
            # If explicitly unchecked
            calib_seconds = 0
        else:
            calib_seconds = max(config.CALIB_MIN_SECONDS, min(config.CALIB_MAX_SECONDS, calib_val))
    except ValueError:
        calib_seconds = config.DEFAULT_CALIB_SECONDS

    # Save uploaded video
    safe_name = secure_filename(file.filename) or "video.mp4"
    temp_prefix = f"upload_{int(time.time())}"
    dest_path = config.UPLOAD_DIR / f"{temp_prefix}_{safe_name}"
    file.save(dest_path)

    job = register_job(
        video_name=file.filename,
        input_path=str(dest_path),
        calib_seconds=calib_seconds
    )

    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
        return jsonify({
            "job_id": job.id,
            "redirect_url": url_for("watch", job_id=job.id)
        })
    return redirect(url_for("watch", job_id=job.id))


@app.route("/stream/<job_id>")
def stream(job_id):
    job = get_job(job_id)
    if not job:
        abort(404, description="Job not found")

    def mjpeg_generator():
        last_emitted_frame = -1
        while True:
            frame_data = None
            with job.frame_condition:
                while True:
                    cur_f = job.latest.get("frame", 0) if job.latest else 0
                    if cur_f != last_emitted_frame and job.current_frame_jpg is not None:
                        frame_data = job.current_frame_jpg
                        last_emitted_frame = cur_f
                        break
                    if job.status in ("done", "error"):
                        frame_data = job.current_frame_jpg
                        break
                    # Wait for next frame signal
                    job.frame_condition.wait(timeout=0.4)

            if frame_data:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")

            if job.status in ("done", "error"):
                break

    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats/<job_id>")
def stats(job_id):
    job = get_job(job_id)
    if not job:
        abort(404, description="Job not found")

    # Send compact chart data points
    chart_points = [
        {
            "time": s["time"],
            "score": round(s["score"], 4),
            "threshold": round(s["threshold"], 4),
            "alert": s["alert"]
        }
        for s in job.stats_history
    ]

    return jsonify({
        "id": job.id,
        "video_name": job.video_name,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "latest": job.latest,
        "alert_events": job.alert_events,
        "summary": job.summary,
        "error_message": job.error_message,
        "chart_points": chart_points,
        "has_result_file": (job.status == "done" and os.path.exists(job.output_path))
    })


@app.route("/result/<job_id>")
def result(job_id):
    job = get_job(job_id)
    if not job:
        abort(404, description="Job not found")
    if job.status != "done" or not os.path.exists(job.output_path):
        abort(400, description="Video processing not finished or result file unavailable")
    
    return send_file(
        job.output_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=f"iadl_cde_{job.id}_annotated.mp4"
    )


@app.route("/stats/<job_id>/csv")
def stats_csv(job_id):
    job = get_job(job_id)
    if not job:
        abort(404, description="Job not found")
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["frame", "time_sec", "density", "density_conf", "score", "threshold", "alert", "boxes_count"])
    for s in job.stats_history:
        writer.writerow([
            s.get("frame"),
            s.get("time"),
            s.get("density"),
            round(s.get("density_conf", 0.0), 4),
            round(s.get("score", 0.0), 5),
            round(s.get("threshold", 0.0), 5),
            1 if s.get("alert") else 0,
            s.get("boxes", 0)
        ])
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"iadl_cde_{job.id}_stats.csv"
    )


# ----------------------------------------------------------------- Camera Fine-Tuning API
@app.route("/camera/finetune", methods=["POST"])
def camera_finetune():
    cur_status = get_finetune_status()
    if cur_status["status"] == "running":
        return jsonify({"error": "Fine-tuning is already in progress"}), 400

    if "clip" not in request.files:
        return jsonify({"error": "No clip file provided"}), 400

    file = request.files["clip"]
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid video file format for camera clip"}), 400

    safe_name = secure_filename(file.filename) or "clip.mp4"
    dest_path = config.UPLOAD_DIR / f"finetune_{int(time.time())}_{safe_name}"
    file.save(dest_path)

    try:
        seconds = int(request.form.get("seconds", 40))
        epochs = int(request.form.get("epochs", 15))
    except ValueError:
        seconds = 40
        epochs = 15

    run_camera_finetune(str(dest_path), seconds=seconds, epochs=epochs)
    return jsonify({"message": "Camera fine-tuning started in background", "status": "running"})


@app.route("/camera/status")
def camera_status():
    st = get_finetune_status()
    info = get_active_model_info()
    st.update(info)
    return jsonify(st)


@app.route("/camera/switch", methods=["POST"])
def camera_switch():
    data = request.get_json(silent=True) or request.form
    target = data.get("model", "default")
    if target not in ("default", "camera"):
        return jsonify({"error": "Invalid model selection"}), 400
    try:
        switch_engine_model(target)
        info = get_active_model_info()
        return jsonify({
            "message": f"Successfully switched to {target} model",
            "active_model": info["active_model"],
            "config": info["config"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File exceeds maximum upload size (500 MB)"}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
