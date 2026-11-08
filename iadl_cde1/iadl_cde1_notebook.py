# =====================================================================================
# IADL-CDE.1  -  Kaggle notebook  (density classification + anomaly localization + video test)
# Paste each "# %%" block into its own cell, in order.
# Settings: Accelerator = GPU T4 x2 (or P100), Internet = ON (needed once to download ResNet34 weights),
#           Persistence = Files only (so /kaggle/working survives a session restart).
# Inputs:   your ShanghaiTech dataset + your UCSD Anomaly dataset (+ your own mp4 test videos later).
# Upgrades vs the first notebook: pretrained backbone, proper train/val/TEST split, next-frame U-Net
# for anomalies (motion-aware), thresholds from HELD-OUT normal videos, density-aware thresholds.
# =====================================================================================

# %% [1] Imports + seed
import os, sys, glob, json, random, time, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, cohen_kappa_score,
                             confusion_matrix, roc_auc_score)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEV == "cuda"
OUT = "/kaggle/working"
sys.path.insert(0, OUT)
print("device:", DEV)

# %% [2] Look at the input folders (fix the globs in cells 5 and 8 if your folder names differ)
for root, dirs, files in os.walk("/kaggle/input"):
    if root.count(os.sep) <= 5:
        print(root, "| dirs:", dirs[:4], "| files:", files[:3])

# %% [3] Write the shared core module (models + engine + video analyzer) to /kaggle/working/iadl_core.py
%%writefile /kaggle/working/iadl_core.py
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
        l1, _ = self.density(x); l2, _ = self.density(torch.flip(x, [3]))
        p = ((l1.softmax(1) + l2.softmax(1)) / 2)[0].cpu().numpy()
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


# %% [4] Config + import the shared core (models, engine, video analyzer)
# (If you edit the writefile cell above, re-run it and then restart the session or use importlib.reload.)
from iadl_core import *
T_SPARSE, T_MEDIUM = 100, 400            # head-count thresholds: <100 sparse, 100-400 medium, >400 dense
DEN_EPOCHS, DEN_BATCH, DEN_LR = 25, 32, 3e-4
PRED_EPOCHS, PRED_BATCH, PRED_LR = 40, 16, 2e-4
VAL_VIDS = 3                             # last 3 UCSD training videos are held out to set thresholds

# %% [5] ShanghaiTech: images + head counts -> density labels, train/val/test split (70/15/15)
imgs = sorted(glob.glob("/kaggle/input/**/IMG_*.jpg", recursive=True))
print("images found:", len(imgs))

def head_count(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    d = os.path.dirname(os.path.dirname(img_path))
    mats = glob.glob(os.path.join(d, "ground[-_]truth", f"GT_{base}.mat")) + \
           glob.glob(f"/kaggle/input/**/GT_{base}.mat", recursive=True)
    if mats:
        m = loadmat(mats[0]); return int(m["image_info"][0, 0][0, 0][0].shape[0])
    import h5py                                   # fallback: density-map .h5 files
    hs = glob.glob(os.path.join(d, "ground[-_]truth*", f"{base}.h5")) + glob.glob(f"/kaggle/input/**/{base}.h5", recursive=True)
    with h5py.File(hs[0], "r") as f:
        return int(round(np.array(f["density"]).sum()))

counts = np.array([head_count(p) for p in imgs])
labels = np.digitize(counts, [T_SPARSE, T_MEDIUM])
print("class distribution:", {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)})
X = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (SHT_SIZE, SHT_SIZE),
                         interpolation=cv2.INTER_AREA) for p in imgs])            # N,H,W,3 uint8
logc = np.log1p(counts).astype(np.float32)
idx = np.arange(len(X))
tr, tmp = train_test_split(idx, test_size=0.3, stratify=labels, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=SEED)
print("train/val/test:", len(tr), len(va), len(te), "| RAM for images: %.0f MB" % (X.nbytes / 1e6))

# %% [6] Train the density network (SMCNN-v2). Checkpoint chosen on VAL; TEST is only used in cell 7.
mean_t = torch.tensor(IMNET_MEAN, device=DEV).view(1, 3, 1, 1); std_t = torch.tensor(IMNET_STD, device=DEV).view(1, 3, 1, 1)
def prep(xb): return (xb.permute(0, 3, 1, 2).float() / 255 - mean_t) / std_t

@torch.no_grad()
def predict_density(net, Xs, bs=32):
    net.eval(); P = []
    for i in range(0, len(Xs), bs):
        xb = prep(torch.from_numpy(Xs[i:i + bs]).to(DEV))
        with torch.autocast("cuda", enabled=USE_AMP):
            l1, _ = net(xb); l2, _ = net(torch.flip(xb, [3]))
        P.append(((l1.float().softmax(1) + l2.float().softmax(1)) / 2).cpu().numpy())
    return np.concatenate(P)

den = DensityNet(pretrained=True).to(DEV)
head = [p for n, p in den.named_parameters() if not n.startswith("backbone")]
opt = torch.optim.AdamW([{"params": den.backbone.parameters(), "lr": DEN_LR / 3},
                         {"params": head, "lr": DEN_LR}], weight_decay=1e-4)
spe = int(np.ceil(len(tr) / DEN_BATCH))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[DEN_LR / 3, DEN_LR], total_steps=DEN_EPOCHS * spe)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
cw = torch.tensor(len(tr) / (3 * np.bincount(labels[tr], minlength=3).clip(1)), dtype=torch.float32, device=DEV)
best = 0
for ep in range(DEN_EPOCHS):
    den.train(); perm = np.random.permutation(tr); tl = 0
    for i in range(0, len(perm), DEN_BATCH):
        b = perm[i:i + DEN_BATCH]
        xb = prep(torch.from_numpy(X[b]).to(DEV))
        flip = torch.rand(len(b), device=DEV) < 0.5
        xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)                      # horizontal flip
        xb = xb * (1 + 0.2 * (torch.rand(len(b), 1, 1, 1, device=DEV) - 0.5))          # intensity jitter
        yb = torch.from_numpy(labels[b]).long().to(DEV); rb = torch.from_numpy(logc[b]).to(DEV)
        with torch.autocast("cuda", enabled=USE_AMP):
            lg, rg = den(xb)
        loss = F.cross_entropy(lg.float(), yb, weight=cw) + 0.5 * F.smooth_l1_loss(rg.float(), rb)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(b)
    pv = predict_density(den, X[va]).argmax(1); acc = accuracy_score(labels[va], pv)
    if acc > best:
        best = acc; torch.save(den.state_dict(), f"{OUT}/density.pt")
    print(f"ep{ep+1:02d} loss {tl/len(tr):.4f} val_acc {acc:.4f} (best {best:.4f})")

# %% [7] Final density evaluation on the untouched TEST split
den.load_state_dict(torch.load(f"{OUT}/density.pt")); den.eval()
pt = predict_density(den, X[te]).argmax(1)
p_, r_, f_, _ = precision_recall_fscore_support(labels[te], pt, average="macro", zero_division=0)
print(f"TEST acc {accuracy_score(labels[te], pt):.4f} | macro P {p_:.4f} R {r_:.4f} F1 {f_:.4f} | kappa {cohen_kappa_score(labels[te], pt):.4f}")
cm = confusion_matrix(labels[te], pt); print(cm)
plt.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3): plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xticks(range(3), CLASSES); plt.yticks(range(3), CLASSES); plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Density confusion matrix (test)"); plt.savefig(f"{OUT}/density_confusion.png", dpi=200); plt.show()

# %% [8] UCSD Ped2: normal training videos + test videos with pixel masks
ped = glob.glob("/kaggle/input/**/UCSDped2", recursive=True)[0]
def read_gray(f): return cv2.resize(cv2.imread(f, 0), (PRED_W, PRED_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
train_vids = [np.stack([read_gray(f) for f in sorted(glob.glob(os.path.join(d, "*.tif")))])
              for d in sorted(glob.glob(f"{ped}/Train/Train*")) if os.path.isdir(d)]
test_vids, test_masks = [], []
for d in sorted(glob.glob(f"{ped}/Test/Test[0-9][0-9][0-9]")):
    gt = d + "_gt"
    if not os.path.isdir(gt): continue
    mks = sorted(glob.glob(os.path.join(gt, "*.bmp")))
    frs = sorted(glob.glob(os.path.join(d, "*.tif")))[:len(mks)]
    test_vids.append(np.stack([read_gray(f) for f in frs]))
    test_masks.append(np.stack([cv2.resize(cv2.imread(m, 0), (PRED_W, PRED_H), interpolation=cv2.INTER_NEAREST) > 0 for m in mks]))
print("train videos:", len(train_vids), [len(v) for v in train_vids])
print("test videos:", len(test_vids), [len(v) for v in test_vids])

# %% [9] Train the next-frame U-Net on NORMAL frames only (3 held-out videos kept for thresholds)
G = [torch.from_numpy((v * 255).round().astype(np.uint8)).to(DEV) for v in train_vids[:-VAL_VIDS]]
wins = [(vi, t) for vi, g in enumerate(G) for t in range(len(g) - N_IN)]
unet = UNet().to(DEV)
opt = torch.optim.Adam(unet.parameters(), lr=PRED_LR)
steps = len(wins) // PRED_BATCH
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRED_EPOCHS * steps)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
print("training windows:", len(wins), "| steps/epoch:", steps)
for ep in range(PRED_EPOCHS):
    unet.train(); random.shuffle(wins); tl = 0
    for i in range(0, steps * PRED_BATCH, PRED_BATCH):
        clip = torch.stack([G[v][t:t + N_IN + 1] for v, t in wins[i:i + PRED_BATCH]]).float() / 127.5 - 1   # B,5,H,W
        xb, gt = clip[:, :N_IN], clip[:, N_IN:]
        if random.random() < 0.5: xb, gt = xb.flip(3), gt.flip(3)
        with torch.autocast("cuda", enabled=USE_AMP):
            p = unet(xb)
        loss = pred_loss(p.float(), gt)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item()
    if ep % 5 == 0 or ep == PRED_EPOCHS - 1: print(f"ep{ep+1:02d} loss {tl/steps:.5f}")
torch.save(unet.state_dict(), f"{OUT}/predictor.pt")

# %% [10] Thresholds from HELD-OUT normal videos, then evaluation on the Ped2 test set
unet.eval()
cfg = {"class_thresholds": [T_SPARSE, T_MEDIUM], "density_factor": {"0": 1.0, "1": 1.15, "2": 1.3},   # factors: heuristic
       "pix_thr": 0.0, "frame_thr": 0.0, "predictor_file": "predictor.pt"}
eng = Engine(den, unet, cfg, DEV)
VE = [eng.video_errors(v) for v in train_vids[-VAL_VIDS:]]
Eall = np.concatenate([e[::3] for e in VE])
pix_thr = float(np.percentile(Eall, 99.9))
fs_val = np.concatenate([[frame_score(e) for e in E] for E in VE])
frame_thr = float(np.percentile(fs_val, 99) * 1.1)
eng.cfg.update(pix_thr=pix_thr, frame_thr=frame_thr)
print(f"thresholds from held-out normal videos: pixel {pix_thr:.5f} | frame {frame_thr:.5f}")

def video_density_factor(v):
    cs = [eng.density_of((v[i] * 255).astype(np.uint8))[0] for i in range(0, len(v), 15)]
    return eng.cfg["density_factor"][str(int(np.bincount(cs).argmax()))]

Y, PM, PS, VID, Es, Ms, Fac = [], [], [], [], [], [], []
for vi, (v, m) in enumerate(zip(test_vids, test_masks)):
    E = eng.video_errors(v)
    Y.append(m[N_IN:].any((1, 2)))
    PM.append(np.array([frame_score(e) for e in E]))
    PS.append(-10 * np.log10(4.0 / (E.mean((1, 2)) + 1e-12)))          # negative PSNR = anomaly score
    VID.append(np.full(len(E), vi)); Es.append(E); Ms.append(m[N_IN:]); Fac.append(video_density_factor(v))
Yc, PMc, PSc, Vc = map(np.concatenate, (Y, PM, PS, VID))
Fc = np.concatenate([np.full(len(y), f) for y, f in zip(Y, Fac)])

def vnorm(s, vid):
    out = s.astype(np.float64).copy()
    for v in np.unique(vid):
        m = vid == v; lo, hi = out[m].min(), out[m].max(); out[m] = (out[m] - lo) / (hi - lo + 1e-9)
    return out

print("\n--- frame level ---")
for name, s in [("patch-max error", PMc), ("PSNR", PSc)]:
    print(f"{name}: AUC raw {roc_auc_score(Yc, s):.4f} | AUC per-video-normalized {roc_auc_score(Yc, vnorm(s, Vc)):.4f}")
pf = PMc > frame_thr * Fc
p1, r1, f1, _ = precision_recall_fscore_support(Yc, pf, average="binary", zero_division=0)
print(f"fixed threshold: acc {accuracy_score(Yc, pf):.4f} P {p1:.4f} R {r1:.4f} F1 {f1:.4f} kappa {cohen_kappa_score(Yc, pf):.4f}")

def pixel_eval(use_density, thr0=None):
    tp = fp = fn = tn = 0
    for E, m, f in zip(Es, Ms, Fac):
        pr = E > (thr0 or pix_thr) * (f if use_density else 1.0)
        tp += int((pr & m).sum()); fp += int((pr & ~m).sum()); fn += int((~pr & m).sum()); tn += int((~pr & ~m).sum())
    return tp / (tp + fp + fn + 1e-9), tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
print("\n--- pixel level (IoU, precision, recall) ---")
print("without density factor:", [round(x, 4) for x in pixel_eval(False)])
print("with density factor   :", [round(x, 4) for x in pixel_eval(True)])
print("diagnosis only (thresholds from held-out normal data):")
for q in [99.0, 99.5, 99.9, 99.99]:
    print(f"  pct {q}:", [round(x, 4) for x in pixel_eval(False, float(np.percentile(Eall, q)))])

# %% [11] Save deployment files (download density.pt, predictor.pt, config.json for the web app)
json.dump(eng.cfg, open(f"{OUT}/config.json", "w"), indent=2)
print(os.listdir(OUT)); print(json.dumps(eng.cfg, indent=2))

# %% [12] Visualize: frame | error heat-map with ground truth outline | detected boxes
sel = [(vi, t) for vi in range(len(Ms)) for t in range(len(Ms[vi])) if Ms[vi][t].any()]
pick = sel[::max(1, len(sel) // 4)][:4]
fig, ax = plt.subplots(len(pick), 3, figsize=(12, 3.2 * len(pick)))
for r, (vi, t) in enumerate(pick):
    fr, E, m = test_vids[vi][t + N_IN], Es[vi][t], Ms[vi][t]
    ax[r, 0].imshow(fr, cmap="gray"); ax[r, 1].imshow(E, cmap="jet"); ax[r, 1].contour(m, colors="white", linewidths=0.8)
    ax[r, 2].imshow(fr, cmap="gray")
    cs, _ = cv2.findContours(cv2.dilate((E > pix_thr).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 30: ax[r, 2].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.5))
    for a, tt in zip(ax[r], ["Frame", "Prediction error (white = ground truth)", "Detections"]): a.set_title(tt); a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/anomaly_localization.png", dpi=200); plt.show()

# %% [13] TEST ON YOUR OWN CCTV mp4 VIDEOS
# Upload videos: Add Input -> Upload -> New Dataset (drag the mp4 files in).  They appear in /kaggle/input/<name>/
# NOTE: the predictor was trained on UCSD Ped2 (fixed camera, walkway, 10 fps). For another camera, give it a few
# seconds of NORMAL footage via calib_seconds (below) and ideally fine-tune on that camera (cell 14).
videos = sorted(glob.glob("/kaggle/input/**/*.mp4", recursive=True))
print(videos)
eng = Engine.from_dir(OUT)                                   # loads density.pt, predictor.pt, config.json
analyzer = VideoAnalyzer(eng, calib_seconds=10)              # first 10 s of the video assumed normal; 0 = use saved thresholds
stats = process_to_file(analyzer, videos[0], f"{OUT}/result_{os.path.basename(videos[0])}")
al = np.array([s["alert"] for s in stats]); print("frames:", len(stats), "| alert frames: %.1f%%" % (100 * al.mean()))
print("density counts:", {c: sum(s["density"] == c for s in stats) for c in CLASSES})
t = [s["time"] for s in stats]
plt.plot(t, [s["score"] for s in stats], label="anomaly score"); plt.plot(t, [s["threshold"] for s in stats], "--", label="threshold")
plt.xlabel("time (s)"); plt.legend(); plt.savefig(f"{OUT}/video_score_timeline.png", dpi=200); plt.show()
# Download result_*.mp4 from the Output tab (browsers play it if ffmpeg re-encoded it to H.264).

# %% [14] OPTIONAL: adapt to ONE camera with a clip that contains only normal activity (30-60 s is plenty)
cam_video = videos[0]                                        # <- a normal-only clip from the same camera
eng = Engine.from_dir(OUT)
print("fine-tuned thresholds (pixel, frame):", finetune_camera(eng, cam_video, seconds=40, epochs=15))
torch.save(eng.pred.state_dict(), f"{OUT}/predictor_cam.pt")
json.dump(dict(eng.cfg, predictor_file="predictor_cam.pt"), open(f"{OUT}/config_cam.json", "w"), indent=2)
# To use it: copy predictor_cam.pt next to density.pt and rename config_cam.json -> config.json.
