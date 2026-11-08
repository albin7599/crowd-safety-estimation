"""iadl_core.py - shared by the Kaggle notebook (IADL-CDE.1) and the web app.
Density: ResNet34 + self-modulation + attention (SMCNN-v2).  Anomaly: U-Net next-frame predictor."""
import os, json, random, subprocess
from collections import deque
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision

CLASSES = ["sparse", "medium", "dense"]
N_IN, PRED_H, PRED_W, SHT_SIZE = 4, 192, 288, 384        # 4 input frames -> predict the 5th
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DENSITY_COLORS = [(0, 200, 0), (0, 165, 255), (0, 0, 255)]   # BGR


# ----------------------------------------------------------------- models
class Modulation(nn.Module):
    """Self-modulation: scene-dependent scale/shift computed from the feature map itself."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(c, c // 2), nn.ReLU(), nn.Linear(c // 2, 2 * c))
    def forward(self, x):
        h = F.relu(self.conv(x))
        g, b = self.fc(h.mean((2, 3))).chunk(2, 1)
        return F.relu(h * (1 + g[:, :, None, None]) + b[:, :, None, None])


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ch = nn.Sequential(nn.Linear(c, c // 4), nn.ReLU(), nn.Linear(c // 4, c), nn.Sigmoid())
        self.sp = nn.Sequential(nn.Conv2d(c, 1, 7, padding=3), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ch(x.mean((2, 3)))[:, :, None, None]
        return x * self.sp(x)


class DensityNet(nn.Module):
    """Pretrained ResNet34 backbone -> 2 x Modulation -> Attention (+skip) -> class logits + log-count."""
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.mod = nn.Sequential(Modulation(512), Modulation(512))
        self.att = Attention(512)
        self.drop = nn.Dropout(0.3)
        self.cls = nn.Linear(512, 3)
        self.reg = nn.Linear(512, 1)
    def forward(self, x):
        f = self.backbone(x)
        h = self.att(self.mod(f)) + f
        v = self.drop(h.mean((2, 3)))
        return self.cls(v), self.reg(v).squeeze(1)


def _cb(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    """Next-frame predictor: 4 grayscale frames in [-1,1] -> 5th frame."""
    def __init__(self, cin=N_IN, base=48):
        super().__init__()
        self.e1, self.e2 = _cb(cin, base), _cb(base, base * 2)
        self.e3, self.e4 = _cb(base * 2, base * 4), _cb(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3, self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), _cb(base * 8, base * 4)
        self.u2, self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), _cb(base * 4, base * 2)
        self.u1, self.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), _cb(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return torch.tanh(self.out(d1))


def grad_loss(p, g):
    dxp, dxg = (p[..., 1:] - p[..., :-1]).abs(), (g[..., 1:] - g[..., :-1]).abs()
    dyp, dyg = (p[..., 1:, :] - p[..., :-1, :]).abs(), (g[..., 1:, :] - g[..., :-1, :]).abs()
    return (dxp - dxg).abs().mean() + (dyp - dyg).abs().mean()


def pred_loss(p, g):                       # intensity (L2) + gradient (L1) loss
    return F.mse_loss(p, g) + grad_loss(p, g)


# ----------------------------------------------------------------- helpers
def gray_small(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (PRED_W, PRED_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def frame_score(e):
    """Anomaly score of a frame = max of the 16x16 box-averaged error map."""
    return float(cv2.blur(e, (16, 16)).max())


def fps_step(path, target_fps=10.0):
    """UCSD runs at ~10 fps; subsample faster CCTV videos so motion between frames is comparable."""
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; cap.release()
    return fps, max(1, int(round(fps / target_fps)))


def read_gray_frames(path, seconds=None, target_fps=10.0):
    fps, step = fps_step(path, target_fps)
    cap = cv2.VideoCapture(path); limit = int(seconds * fps) if seconds else 10 ** 9
    out, i = [], 0
    while i < limit:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(gray_small(fr))
        i += 1
    cap.release()
    return np.stack(out) if out else np.zeros((0, PRED_H, PRED_W), np.float32)


def _label(img, text, org, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (org[0] - 4, org[1] - h - 6), (org[0] + w + 4, org[1] + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------- engine
class Engine:
    def __init__(self, density, pred, cfg, device):
        self.dev, self.cfg = device, cfg
        self.density, self.pred = density.to(device).eval(), pred.to(device).eval()
        self.mean = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)

    @classmethod
    def from_dir(cls, d, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(os.path.join(d, "config.json")))
        den = DensityNet(pretrained=False)
        den.load_state_dict(torch.load(os.path.join(d, "density.pt"), map_location=device))
        pr = UNet()
        pr.load_state_dict(torch.load(os.path.join(d, cfg.get("predictor_file", "predictor.pt")), map_location=device))
        return cls(den, pr, cfg, device)

    @torch.no_grad()
    def density_of(self, img):
        """img: BGR or gray uint8 frame -> (class index, probabilities). Flip test-time augmentation."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (SHT_SIZE, SHT_SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).to(self.dev).permute(2, 0, 1)[None].float() / 255
        x = (x - self.mean) / self.std
        l1, r1 = self.density(x); l2, r2 = self.density(torch.flip(x, [3]))
        p = ((l1.softmax(1) + l2.softmax(1)) / 2)[0].cpu().numpy()
        th = self.cfg.get("class_thresholds")
        if th and len(th) >= 2:
            pred_logc = float(((r1 + r2) / 2).reshape(-1)[0].cpu())
            pred_count = max(0.0, float(np.expm1(pred_logc)))
            if pred_count < th[0]:
                c = 0
            elif pred_count <= th[1]:
                c = 1
            else:
                c = 2
            old_c = int(p.argmax())
            if old_c != c:
                p[c], p[old_c] = p[old_c], p[c]
            return c, p
        return int(p.argmax()), p

    @torch.no_grad()
    def err_map(self, buf):
        """buf: N_IN+1 small gray frames -> blurred squared prediction error of the last frame."""
        a = torch.from_numpy(np.stack(buf)).to(self.dev) * 2 - 1
        p = self.pred(a[:N_IN][None])
        e = ((p[0, 0] - a[N_IN]) ** 2).cpu().numpy()
        return cv2.GaussianBlur(e, (7, 7), 0)

    @torch.no_grad()
    def video_errors(self, vid, bs=32):
        """vid: (T,H,W) float [0,1] -> error maps (T-N_IN,H,W) for frames N_IN..T-1."""
        x = torch.from_numpy(vid).to(self.dev) * 2 - 1
        E = []
        for i in range(0, len(vid) - N_IN, bs):
            ts = range(i, min(i + bs, len(vid) - N_IN))
            xb = torch.stack([x[t:t + N_IN] for t in ts]); gt = torch.stack([x[t + N_IN] for t in ts])
            E.append(((self.pred(xb)[:, 0] - gt) ** 2).cpu().numpy())
        return np.stack([cv2.GaussianBlur(e, (7, 7), 0) for e in np.concatenate(E)])

    def calibrate(self, path, seconds=10, target_fps=10.0):
        """Thresholds from the first `seconds` of a video that is ASSUMED to contain only normal activity."""
        vid = read_gray_frames(path, seconds, target_fps)
        if len(vid) < N_IN + 10:
            raise ValueError("Need more normal footage for calibration (try a longer clip or smaller calibration time).")
        E = self.video_errors(vid)
        pix = float(np.percentile(E[::2], 99.9))
        fs = np.array([frame_score(e) for e in E])
        return pix, float(np.percentile(fs, 99) * 1.1)


def finetune_camera(engine, path, seconds=40, epochs=15, lr=5e-5, target_fps=10.0, batch=8):
    """Adapt the predictor to one fixed camera using footage that contains only normal activity."""
    vid = read_gray_frames(path, seconds, target_fps)
    ntr = int(len(vid) * 0.8)
    tr, va = vid[:ntr], vid[ntr - N_IN:]
    if len(tr) < N_IN + 10 or len(va) < N_IN + 5:
        raise ValueError("Not enough footage; use a longer normal clip.")
    net, dev = engine.pred, engine.dev
    net.train(); opt = torch.optim.Adam(net.parameters(), lr=lr)
    x = torch.from_numpy(tr).to(dev) * 2 - 1
    idx = list(range(len(tr) - N_IN))
    for ep in range(epochs):
        random.shuffle(idx)
        for i in range(0, len(idx) - batch + 1, batch):
            ts = idx[i:i + batch]
            xb = torch.stack([x[t:t + N_IN] for t in ts]); gt = torch.stack([x[t + N_IN] for t in ts]).unsqueeze(1)
            loss = pred_loss(net(xb), gt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    E = engine.video_errors(va)
    pix = float(np.percentile(E[::2], 99.9))
    fs = np.array([frame_score(e) for e in E])
    frame = float(np.percentile(fs, 99) * 1.1)
    engine.cfg = dict(engine.cfg, pix_thr=pix, frame_thr=frame)
    return pix, frame


# ----------------------------------------------------------------- video analysis
class VideoAnalyzer:
    def __init__(self, engine, calib_seconds=0, target_fps=10.0, density_every=15,
                 smooth=5, min_area=30, max_width=960):
        self.eng, self.calib_seconds, self.target_fps = engine, calib_seconds, target_fps
        self.density_every, self.smooth, self.min_area, self.max_width = density_every, smooth, min_area, max_width

    def run(self, path):
        """Generator: yields (annotated BGR frame, stats dict) for every processed frame."""
        eng, cfg = self.eng, self.eng.cfg
        pix_thr, frame_thr = cfg["pix_thr"], cfg["frame_thr"]
        if self.calib_seconds > 0:
            pix_thr, frame_thr = eng.calibrate(path, self.calib_seconds, self.target_fps)
        fps, step = fps_step(path, self.target_fps)
        out_fps = fps / step
        cap = cv2.VideoCapture(path)
        buf, scores = deque(maxlen=N_IN + 1), deque(maxlen=self.smooth)
        dens, k, n = (0, np.array([1.0, 0.0, 0.0])), 0, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            if (n - 1) % step:
                continue
            H0, W0 = fr.shape[:2]
            sc = min(1.0, self.max_width / W0)
            vis = cv2.resize(fr, (int(W0 * sc), int(H0 * sc))) if sc < 1 else fr.copy()
            buf.append(gray_small(fr))
            if k % self.density_every == 0:
                dens = eng.density_of(fr)
            k += 1
            c, probs = dens
            fac = cfg["density_factor"][str(c)]           # stricter thresholds in denser scenes
            score, alert, boxes = 0.0, False, []
            if len(buf) == N_IN + 1:
                e = eng.err_map(list(buf))
                scores.append(frame_score(e)); score = float(np.median(scores))
                alert = score > frame_thr * fac
                if alert:
                    m = (e > pix_thr * fac).astype(np.uint8)
                    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
                    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    sx, sy = vis.shape[1] / PRED_W, vis.shape[0] / PRED_H
                    for ct in cs:
                        x, y, w, h = cv2.boundingRect(ct)
                        if w * h >= self.min_area:
                            boxes.append((int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)))
                    for b in boxes:
                        cv2.rectangle(vis, b[:2], b[2:], (0, 0, 255), 2)
            _label(vis, f"Density: {CLASSES[c]} ({probs[c]:.0%})", (10, 25), DENSITY_COLORS[c])
            if alert:
                _label(vis, "ANOMALY DETECTED", (10, 55), (0, 0, 255))
            yield vis, {"frame": n, "time": round((n - 1) / fps, 2), "density": CLASSES[c],
                        "density_conf": float(probs[c]), "score": score, "threshold": float(frame_thr * fac),
                        "alert": bool(alert), "boxes": len(boxes), "out_fps": out_fps}
        cap.release()


def process_to_file(analyzer, path, out_path, max_seconds=None):
    """Annotate a whole video and write an H.264 mp4 (ffmpeg if available). Returns the per-frame stats."""
    tmp, writer, stats = out_path + ".tmp.mp4", None, []
    for vis, st in analyzer.run(path):
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), st["out_fps"], (w, h))
        writer.write(vis); stats.append(st)
        if max_seconds and st["time"] > max_seconds:
            break
    if writer is not None:
        writer.release()
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", out_path], check=True)
            os.remove(tmp)
        except Exception:
            os.replace(tmp, out_path)
    return stats
