"""Crowd Density Monitoring web app (IADL-CDE): upload a CCTV mp4 -> live annotated stream + density + alerts.
Run:  python app.py   then open http://127.0.0.1:5000"""
import os, time, uuid, threading
import cv2
from flask import Flask, request, redirect, render_template_string, Response, jsonify, abort
from iadl_core import Engine, VideoAnalyzer

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("IADL_WEIGHTS", os.path.join(BASE, "weights"))   # density.pt, predictor.pt, config.json
UPLOADS = os.path.join(BASE, "uploads")
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024        # 1 GB upload limit
engine = Engine.from_dir(WEIGHTS)                             # loaded once at start-up
print("models loaded on", engine.dev)
JOBS = {}                                                     # job id -> state
LOCK = threading.Lock()                                       # one video analysed at a time

INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>Crowd Density Monitoring</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#222}
.card{border:1px solid #ddd;border-radius:10px;padding:20px}label{display:block;margin:14px 0 4px;font-weight:600}
button{margin-top:18px;padding:10px 20px;border:0;border-radius:8px;background:#1a56db;color:#fff;font-size:16px;cursor:pointer}
small{color:#666}</style></head><body>
<h1>Crowd Density Monitoring</h1>
<p>Upload a CCTV video (mp4). The system classifies crowd density (sparse / medium / dense) and flags anomalies
(e.g. bicycles, skateboards, vehicles) in real time.</p>
<div class="card"><form action="/upload" method="post" enctype="multipart/form-data">
<label>Video file</label><input type="file" name="video" accept="video/*" required>
<label>Calibration (seconds)</label><input type="number" name="calib" value="10" min="0" max="120">
<small>The first N seconds are assumed to show only normal activity and are used to set the anomaly thresholds for this camera.
Use 0 to use the thresholds learned on UCSD Ped2.</small><br>
<button type="submit">Analyse video</button></form></div></body></html>"""

WATCH = """<!doctype html><html><head><meta charset="utf-8"><title>Analysis</title>
<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
.row{display:flex;gap:20px;flex-wrap:wrap}.video{flex:2;min-width:320px}.side{flex:1;min-width:260px}
img{width:100%;border-radius:10px;background:#000}.box{border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:14px}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;color:#fff;font-weight:700}
.ok{background:#16a34a}.bad{background:#dc2626}table{width:100%;border-collapse:collapse}td,th{padding:4px 6px;border-bottom:1px solid #eee;text-align:left;font-size:14px}</style></head><body>
<h2>Live analysis</h2><div class="row"><div class="video"><img src="/stream/__JID__" alt="stream"></div>
<div class="side"><div class="box"><div>Status: <span id="st" class="badge ok">NORMAL</span></div>
<p>Crowd density: <b id="den">-</b> <span id="conf"></span></p><p>Time: <span id="tm">0</span> s</p>
<p>Anomaly score: <span id="sc">0</span> (threshold <span id="th">0</span>)</p><p id="done"></p></div>
<div class="box"><b>Alert log</b><table><thead><tr><th>Time (s)</th><th>Density</th></tr></thead><tbody id="log"></tbody></table></div>
<a href="/">&larr; Analyse another video</a></div></div>
<script>
async function tick(){
  try{
    const r = await fetch('/stats/__JID__'); const s = await r.json();
    if(s.latest){const l=s.latest;
      document.getElementById('den').textContent=l.density; document.getElementById('conf').textContent='('+Math.round(l.density_conf*100)+'%)';
      document.getElementById('tm').textContent=l.time; document.getElementById('sc').textContent=l.score.toFixed(4);
      document.getElementById('th').textContent=l.threshold.toFixed(4);
      const st=document.getElementById('st'); st.textContent=l.alert?'ANOMALY':'NORMAL'; st.className='badge '+(l.alert?'bad':'ok');}
    document.getElementById('log').innerHTML = s.alerts.map(a=>'<tr><td>'+a.time+'</td><td>'+a.density+'</td></tr>').join('');
    if(s.done) document.getElementById('done').textContent='Analysis finished.';
  }catch(e){}
}
setInterval(tick, 1000); tick();
</script></body></html>"""


@app.route("/")
def index():
    return INDEX


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return redirect("/")
    jid = uuid.uuid4().hex[:10]
    path = os.path.join(UPLOADS, jid + "_" + os.path.basename(f.filename).replace(" ", "_"))
    f.save(path)
    try:
        calib = max(0, min(120, int(request.form.get("calib", 10))))
    except ValueError:
        calib = 10
    JOBS[jid] = {"path": path, "calib": calib, "latest": None, "alerts": [], "in_alert": False, "done": False}
    return redirect(f"/watch/{jid}")


@app.route("/watch/<jid>")
def watch(jid):
    if jid not in JOBS:
        abort(404)
    return render_template_string(WATCH.replace("__JID__", jid))


def generate(job):
    analyzer = VideoAnalyzer(engine, calib_seconds=job["calib"])
    with LOCK:
        try:
            for vis, st in analyzer.run(job["path"]):
                t0 = time.time()
                job["latest"] = st
                if st["alert"] and not job["in_alert"]:
                    job["alerts"].append({"time": st["time"], "density": st["density"]})
                job["in_alert"] = st["alert"]
                ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(max(0.0, 1.0 / st["out_fps"] - (time.time() - t0)))     # real-time pacing
        except Exception as e:                                                       # e.g. calibration clip too short
            job["latest"] = None
            job["alerts"].append({"time": "-", "density": f"error: {e}"})
        finally:
            job["done"] = True


@app.route("/stream/<jid>")
def stream(jid):
    job = JOBS.get(jid) or abort(404)
    return Response(generate(job), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats/<jid>")
def stats(jid):
    job = JOBS.get(jid) or abort(404)
    return jsonify(latest=job["latest"], alerts=job["alerts"][-30:], done=job["done"])


@app.route("/health")
def health():
    return jsonify(status="ok", device=engine.dev)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
