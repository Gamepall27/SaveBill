"""Train the small receipt segmentation network used by app/scanner.py.

The network sees a photo letterboxed to 256x256 (BGR, uint8 values as float) and predicts,
for every pixel, the probability that it belongs to the receipt (the full outline, also
where fingers cover it). It is trained on

- real photos: the labelled CORD v2 training split (tools/fetch_cord.py --train), with
  random crops, rotations, colour and blur changes, and
- composites: rectified CORD training receipts placed on real background photos
  (tools/fetch_backgrounds.py) with random size (2-85 % of the photo), rotation,
  perspective, curl, torn edges, shadows, lighting, fingers, bright distractors, cards and
  tags that are not receipts, near-white surfaces, partial cut-off by the frame, noise and
  JPEG artefacts.

None of the CORD test/validation photos used by tools/eval_detection.py are used here.
The result is exported to ONNX (app/models/receipt_seg.onnx) and run with cv2.dnn.

Needs PyTorch + onnx (training only, not needed by the app):
    pip install torch onnx onnxscript opencv-python-headless numpy

Usage:
    python tools/train_segmenter.py prepare <cord-train-dir> <background-dir> <cache-dir>
    python tools/train_segmenter.py train <cache-dir> --out app/models/receipt_seg.onnx [--iters 20000]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
PAD = 114  # letterbox fill value (the app uses the same)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Background search terms kept out of training, so that make_small_set.py --split eval
# composes its evaluation photos on surfaces the network has never seen.
EVAL_BACKGROUNDS = ("lace_tablecloth", "granite_countertop", "car_seat_upholstery", "cafe_table",
                    "jacket_fabric", "laminate_flooring", "newspaper_page")


def is_eval_background(name: str) -> bool:
    return name.startswith(EVAL_BACKGROUNDS)


# ----------------------------------------------------------------------------- prepare

def prepare(cord: Path, bgdir: Path, cache: Path) -> None:
    from eval_detection import load_labelled
    from app import scanner

    for sub in ("real", "tex", "bg"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    real, tex = [], []
    for it in json.loads((cord / "manifest.json").read_text()):
        if not it.get("roi"):
            continue
        img, roi = load_labelled(cord / "img" / it["name"], it["roi"])
        q = np.array(roi, np.float32)
        h, w = img.shape[:2]
        s = min(1.0, 768 / max(h, w))
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(cache / "real" / it["name"]), small, [cv2.IMWRITE_JPEG_QUALITY, 93])
        real.append({"name": it["name"], "roi": (q * s).round(2).tolist()})
        # Fully visible receipts become textures for the composites.
        if (q[:, 0] > 2).all() and (q[:, 1] > 2).all() and (q[:, 0] < w - 3).all() and (q[:, 1] < h - 3).all():
            t = scanner.warp(img, q)
            ts = min(1.0, 900 / max(t.shape[:2]))
            t = cv2.resize(t, (round(t.shape[1] * ts), round(t.shape[0] * ts)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(cache / "tex" / it["name"]), t, [cv2.IMWRITE_JPEG_QUALITY, 93])
            tex.append(it["name"])
    bgs = []
    for p in sorted(bgdir.glob("*.jpg")):
        if is_eval_background(p.name):
            continue
        b = cv2.imread(str(p))
        if b is None or min(b.shape[:2]) < 300:
            continue
        s = min(1.0, 1024 / max(b.shape[:2]))
        b = cv2.resize(b, (round(b.shape[1] * s), round(b.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(cache / "bg" / p.name), b, [cv2.IMWRITE_JPEG_QUALITY, 93])
        bgs.append(p.name)
    (cache / "index.json").write_text(json.dumps({"real": real, "tex": tex, "bg": bgs}))
    print(f"{len(real)} real photos, {len(tex)} textures, {len(bgs)} backgrounds -> {cache}")


# ----------------------------------------------------------------------------- samples

def letterbox(img: np.ndarray, size: int = SIZE):
    """Scale the long side to `size` and pad. Returns (canvas, scale, (x0, y0))."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    out = np.full((size, size) + img.shape[2:], PAD if img.ndim == 3 else 0, img.dtype)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out, s, (x0, y0)


def photometric(img: np.ndarray, rng) -> np.ndarray:
    f = img.astype(np.float32)
    f *= rng.uniform(0.6, 1.25)
    f += rng.uniform(-25, 25)
    f *= rng.uniform(0.85, 1.15, 3)  # white balance
    if rng.random() < 0.3:  # desaturate / oversaturate
        g = f.mean(axis=2, keepdims=True)
        f = g + (f - g) * rng.uniform(0.3, 1.6)
    if rng.random() < 0.5:  # uneven light
        lf = cv2.resize(rng.uniform(0.65, 1.15, (3, 3)).astype(np.float32), (f.shape[1], f.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
        f *= lf[:, :, None]
    f = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 2.0))
    if rng.random() < 0.5:
        f = np.clip(f.astype(np.float32) + rng.normal(0, rng.uniform(2, 8), f.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        f = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return f


def finger(canvas: np.ndarray, quad: np.ndarray, rng) -> None:
    """Paint a skin-coloured finger over a receipt edge (the label keeps the full receipt)."""
    i = rng.integers(4)
    a, b = quad[i], quad[(i + 1) % 4]
    p = a + (b - a) * rng.uniform(0.15, 0.85)
    c = quad.mean(axis=0)
    d = (p - c) / (np.linalg.norm(p - c) + 1e-6)
    length = np.linalg.norm(b - a) * rng.uniform(0.25, 0.6)
    width = length * rng.uniform(0.3, 0.45)
    center = p + d * length * rng.uniform(0.0, 0.35)
    ang = math.degrees(math.atan2(d[1], d[0]))
    skin = np.array([rng.uniform(70, 170), rng.uniform(100, 190), rng.uniform(150, 235)])
    cv2.ellipse(canvas, (int(center[0]), int(center[1])), (int(length), int(width / 2)), ang, 0, 360,
                skin.tolist(), -1, cv2.LINE_AA)


def distractor(canvas: np.ndarray, rng) -> None:
    """Bright objects that are not receipts: plates, light spots, rounded shapes."""
    h, w = canvas.shape[:2]
    c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
    v = float(rng.uniform(200, 255))
    col = (v - rng.uniform(0, 20), v - rng.uniform(0, 20), v - rng.uniform(0, 20))
    r = int(min(h, w) * rng.uniform(0.05, 0.3))
    if rng.random() < 0.5:
        cv2.circle(canvas, c, r, col, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, c, (r, int(r * rng.uniform(0.3, 0.9))), float(rng.uniform(0, 180)), 0, 360, col, -1,
                    cv2.LINE_AA)


def card(canvas: np.ndarray, rng, tex: np.ndarray) -> None:
    """A flat object that is NOT a receipt: tag, card, fabric patch, sticky note (drawn under the receipt)."""
    H, W = canvas.shape[:2]
    area = rng.uniform(0.01, 0.2) * W * H
    asp = rng.uniform(1.0, 2.5)
    rh = math.sqrt(area * asp)
    rw = area / rh
    ang = math.radians(rng.uniform(-180, 180))
    rect = np.array([[-rw / 2, -rh / 2], [rw / 2, -rh / 2], [rw / 2, rh / 2], [-rw / 2, rh / 2]], np.float32)
    rect += rng.uniform(-0.08, 0.08, rect.shape).astype(np.float32) * np.array([rw, rh], np.float32)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], np.float32)
    quad = rect @ rot.T + np.array([rng.uniform(0, W), rng.uniform(0, H)], np.float32)
    kind = rng.random()
    if kind < 0.4:  # crop of another photo (fabric, wood, print ...)
        th, tw = tex.shape[:2]
        cw, ch = int(tw * rng.uniform(0.2, 0.6)), int(th * rng.uniform(0.2, 0.6))
        x0, y0 = rng.integers(0, tw - cw + 1), rng.integers(0, th - ch + 1)
        patch = tex[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    else:  # plain card: kraft, coloured, pastel
        col = (rng.uniform(60, 140), rng.uniform(110, 180), rng.uniform(150, 220)) if kind < 0.7 else             tuple(rng.uniform(40, 230, 3))
        patch = np.empty((80, 120, 3), np.float32)
        patch[:] = col
        if rng.random() < 0.4:  # some print on it
            for _ in range(rng.integers(1, 6)):
                y = int(rng.uniform(5, 75))
                cv2.line(patch, (int(rng.uniform(5, 40)), y), (int(rng.uniform(60, 115)), y), (30, 30, 30),
                         int(rng.integers(1, 4)))
    ph, pw = patch.shape[:2]
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]]), quad)
    warped = cv2.warpPerspective(patch, m, (W, H))
    alpha = cv2.warpPerspective(np.ones((ph, pw), np.float32), m, (W, H))[:, :, None]
    shadow = cv2.GaussianBlur(cv2.warpAffine(alpha[:, :, 0], np.float32([[1, 0, 5], [0, 1, 7]]), (W, H)), (0, 0), 6)
    canvas *= 1 - 0.35 * shadow[:, :, None]
    canvas[:] = canvas * (1 - alpha) + warped * alpha


def torn(mask: np.ndarray, rng) -> np.ndarray:
    """Jagged top/bottom edge on the texture mask (torn-off receipts)."""
    h, w = mask.shape
    for top in (True, False):
        if rng.random() < 0.4:
            amp = rng.uniform(2, 0.02 * h + 3)
            xs = np.arange(w)
            prof = (np.abs(((xs / rng.uniform(4, 14)) % 2) - 1) * amp).astype(int)
            for x in range(w):
                if top:
                    mask[:prof[x], x] = 0
                else:
                    mask[h - prof[x]:, x] = 0
    return mask


def curl(tex: np.ndarray, mask: np.ndarray, rng):
    """Gentle bending of the paper (sinusoidal displacement)."""
    h, w = tex.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp = rng.uniform(0, 0.03) * w
    dx = amp * np.sin(yy / h * math.pi * rng.uniform(0.5, 2) + rng.uniform(0, 6.28))
    dy = rng.uniform(0, 0.02) * h * np.sin(xx / w * math.pi * rng.uniform(0.5, 2) + rng.uniform(0, 6.28))
    pad = int(amp) + 4
    tex = cv2.copyMakeBorder(tex, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    yy, xx = np.mgrid[0:h + 2 * pad, 0:w + 2 * pad].astype(np.float32)
    dxp = cv2.copyMakeBorder(dx, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dyp = cv2.copyMakeBorder(dy, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    mx, my = xx - dxp, yy - dyp
    return (cv2.remap(tex, mx, my, cv2.INTER_LINEAR), cv2.remap(mask, mx, my, cv2.INTER_NEAREST), pad)


def composite(tex: np.ndarray, bg: np.ndarray, rng, other: np.ndarray | None = None):
    """Receipt texture on a background photo. Returns (photo, full-outline mask)."""
    portrait = rng.random() < 0.7
    W, H = (480, 640) if portrait else (640, 480)
    bh, bw = bg.shape[:2]
    s = max(W / bw, H / bh) * rng.uniform(1.0, 1.8)
    bgs = cv2.resize(bg, (int(bw * s) + 1, int(bh * s) + 1), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        bgs = cv2.flip(bgs, 1)
    y0 = rng.integers(0, bgs.shape[0] - H + 1)
    x0 = rng.integers(0, bgs.shape[1] - W + 1)
    canvas = bgs[y0:y0 + H, x0:x0 + W].astype(np.float32)
    if rng.random() < 0.15:  # bright, almost white surfaces (white desk, tablecloth)
        a = rng.uniform(0.4, 0.85)
        canvas = canvas * (1 - a) + a * rng.uniform(205, 250)
    for _ in range(rng.integers(1, 3) if other is not None and rng.random() < 0.4 else 0):
        card(canvas, rng, other)
    for _ in range(rng.integers(0, 3) if rng.random() < 0.35 else 0):
        distractor(canvas, rng)

    th, tw = tex.shape[:2]
    if rng.random() < 0.3:  # shorter / longer receipts
        cut = rng.uniform(0.35, 1.0)
        tex = tex[: max(20, int(th * cut))] if rng.random() < 0.5 else tex[th - max(20, int(th * cut)):]
        th = tex.shape[0]
    tmask = torn(np.full((th, tw), 255, np.uint8), rng)
    pad = 0
    if rng.random() < 0.4:
        tex, tmask, pad = curl(tex, tmask, rng)

    frac = math.exp(rng.uniform(math.log(0.02), math.log(0.85)))
    area = frac * W * H
    rh = math.sqrt(area * th / tw)
    rw = area / rh
    ang = rng.uniform(-35, 35) if rng.random() < 0.8 else rng.uniform(-180, 180)
    ang = math.radians(ang)
    rect = np.array([[-rw / 2, -rh / 2], [rw / 2, -rh / 2], [rw / 2, rh / 2], [-rw / 2, rh / 2]], np.float32)
    rect += rng.uniform(-0.1, 0.1, rect.shape).astype(np.float32) * np.array([rw, rh], np.float32)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], np.float32)
    rect = rect @ rot.T
    ext = np.abs(rect).max(axis=0)
    if rng.random() < 0.25:  # partly outside the frame
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    else:
        cx = rng.uniform(ext[0], W - ext[0]) if W > 2 * ext[0] else W / 2 + rng.uniform(-0.1, 0.1) * W
        cy = rng.uniform(ext[1], H - ext[1]) if H > 2 * ext[1] else H / 2 + rng.uniform(-0.1, 0.1) * H
    quad = rect + np.array([cx, cy], np.float32)

    sh, sw = tex.shape[:2]
    src = np.array([[pad, pad], [sw - 1 - pad, pad], [sw - 1 - pad, sh - 1 - pad], [pad, sh - 1 - pad]],
                   np.float32)
    m = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(tex, m, (W, H), flags=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.warpPerspective(tmask, m, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[:, :, None]

    if rng.random() < 0.8:  # drop shadow
        off = rng.uniform(-8, 12, 2)
        shadow = cv2.warpAffine(alpha[:, :, 0], np.float32([[1, 0, off[0]], [0, 1, off[1]]]), (W, H))
        shadow = cv2.GaussianBlur(shadow, (0, 0), rng.uniform(3, 12))[:, :, None]
        canvas *= 1 - rng.uniform(0.15, 0.55) * shadow
    if rng.random() < 0.2:  # paper slightly darker / tinted than the texture
        warped *= rng.uniform(0.75, 1.0, 3)
    out = canvas * (1 - alpha) + warped * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Label: the receipt outline where there is paper (fingers painted later stay inside).
    label = np.zeros((H, W), np.uint8)
    cv2.fillPoly(label, [np.round(quad).astype(np.int32)], 1)
    label &= (alpha[:, :, 0] > 0.5).astype(np.uint8)
    if rng.random() < 0.3:
        for _ in range(rng.integers(1, 3)):
            finger(out, quad, rng)
    return out, label


def real_sample(img: np.ndarray, roi: np.ndarray, rng):
    h, w = img.shape[:2]
    # random crop (may cut the receipt, like close-up photos)
    if rng.random() < 0.7:
        cw, ch = w * rng.uniform(0.6, 1.0), h * rng.uniform(0.6, 1.0)
        x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
    else:
        cw, ch, x0, y0 = w, h, 0.0, 0.0
    # zoom out: receipt smaller in the frame (pad with replicated border)
    if rng.random() < 0.25:
        f = rng.uniform(1.1, 1.8)
        nw, nh = cw * f, ch * f
        x0, y0 = x0 - rng.uniform(0, nw - cw), y0 - rng.uniform(0, nh - ch)
        cw, ch = nw, nh
    ang = rng.uniform(-12, 12) if rng.random() < 0.6 else 0.0
    k = rng.integers(4) if rng.random() < 0.15 else 0
    out_w, out_h = (480, 640) if ch >= cw else (640, 480)
    # affine: crop -> rotate around centre -> output size
    src = np.float32([[x0, y0], [x0 + cw, y0], [x0, y0 + ch]])
    dst = np.float32([[0, 0], [out_w, 0], [0, out_h]])
    m = cv2.getAffineTransform(src, dst)
    r = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), ang, 1.0)
    m = (np.vstack([r, [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]]))[:2]
    border = cv2.BORDER_REFLECT_101 if rng.random() < 0.5 else cv2.BORDER_CONSTANT
    photo = cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_AREA, borderMode=border,
                           borderValue=tuple(int(v) for v in rng.integers(0, 120, 3)))
    q = cv2.transform(roi.reshape(-1, 1, 2), m).reshape(4, 2)
    label = np.zeros((out_h, out_w), np.uint8)
    cv2.fillPoly(label, [np.round(q).astype(np.int32)], 1)
    if k:
        photo, label = np.rot90(photo, k).copy(), np.rot90(label, k).copy()
    return photo, label


class Samples:
    def __init__(self, cache: Path, real_share: float = 0.4, seed: int = 0):
        self.cache = Path(cache)
        idx = json.loads((self.cache / "index.json").read_text())
        self.real = idx["real"]
        self.tex = idx["tex"]
        self.bg = idx["bg"]
        self.real_share = real_share
        self.seed = seed

    def get(self, rng):
        if rng.random() < self.real_share:
            it = self.real[rng.integers(len(self.real))]
            img = cv2.imread(str(self.cache / "real" / it["name"]))
            photo, label = real_sample(img, np.array(it["roi"], np.float32), rng)
        else:
            tex = cv2.imread(str(self.cache / "tex" / self.tex[rng.integers(len(self.tex))]))
            bg = cv2.imread(str(self.cache / "bg" / self.bg[rng.integers(len(self.bg))]))
            other = cv2.imread(str(self.cache / "bg" / self.bg[rng.integers(len(self.bg))]))
            photo, label = composite(tex, bg, rng, other)
        if rng.random() < 0.5:
            photo, label = cv2.flip(photo, 1), cv2.flip(label, 1)
        photo = photometric(photo, rng)
        x, _, _ = letterbox(photo)
        y, _, _ = letterbox(label)
        return x, y


# ----------------------------------------------------------------------------- model

def build_model():
    import torch
    from torch import nn

    def cbr(cin, cout, k=3, s=1, groups=1):
        return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))

    def dw(cin, cout, s=1):
        return nn.Sequential(cbr(cin, cin, 3, s, groups=cin), cbr(cin, cout, 1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = cbr(3, 16, 3, 2)                               # 128
            self.e2 = nn.Sequential(dw(16, 32, 2), dw(32, 32))       # 64
            self.e3 = nn.Sequential(dw(32, 48, 2), dw(48, 48))       # 32
            self.e4 = nn.Sequential(dw(48, 80, 2), dw(80, 80))       # 16
            self.e5 = nn.Sequential(dw(80, 128, 2), dw(128, 128), dw(128, 128))  # 8
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d4 = dw(128 + 80, 80)
            self.d3 = dw(80 + 48, 48)
            self.d2 = dw(48 + 32, 32)
            self.d1 = dw(32 + 16, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            x = (x - 127.5) / 64.0
            e1 = self.e1(x)
            e2 = self.e2(e1)
            e3 = self.e3(e2)
            e4 = self.e4(e3)
            e5 = self.e5(e4)
            d = self.d4(torch.cat([self.up(e5), e4], 1))
            d = self.d3(torch.cat([self.up(d), e3], 1))
            d = self.d2(torch.cat([self.up(d), e2], 1))
            d = self.d1(torch.cat([self.up(d), e1], 1))
            return torch.sigmoid(self.up(self.head(d)))

    return Net()


# ----------------------------------------------------------------------------- train

def _worker_batches(cache, seed, batch, q):
    rng = np.random.default_rng(seed)
    n = 0
    while True:
        if n % 200 == 0:  # pick up a re-run of "prepare" (e.g. more backgrounds) while training
            try:
                s = Samples(cache)
            except (OSError, ValueError):
                pass
        xs, ys = zip(*(s.get(rng) for _ in range(batch)))
        q.put((np.stack(xs), np.stack(ys)))
        n += 1


def train(cache: Path, out: Path, iters: int, batch: int, workers: int, lr: float, resume: str | None):
    import multiprocessing as mp

    import torch
    from torch import nn

    torch.set_num_threads(max(1, (mp.cpu_count() - workers)))
    model = build_model()
    if resume:
        model.load_state_dict(torch.load(resume))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    bce = nn.BCELoss()

    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=workers * 3)
    procs = [ctx.Process(target=_worker_batches, args=(str(cache), 1000 + i, batch, q), daemon=True)
             for i in range(workers)]
    for p in procs:
        p.start()

    # fixed validation batches (different seed)
    vrng = np.random.default_rng(424242)
    vs = Samples(cache)
    val = [vs.get(vrng) for _ in range(96)]
    vx = torch.from_numpy(np.stack([v[0] for v in val])).permute(0, 3, 1, 2).float()
    vy = torch.from_numpy(np.stack([v[1] for v in val])).float()[:, None]

    ckpt = out.with_suffix(".pt")
    t0, run = time.time(), 0.0
    best = 1e9
    for it in range(1, iters + 1):
        xb, yb = q.get()
        x = torch.from_numpy(xb).permute(0, 3, 1, 2).float()
        y = torch.from_numpy(yb).float()[:, None]
        model.train()
        p = model(x)
        inter = (p * y).sum((1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
        loss = bce(p, y) + dice.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 250 == 0 or it == iters:
            model.eval()
            with torch.no_grad():
                pv = torch.cat([model(vx[i:i + 32]) for i in range(0, len(vx), 32)])
                pb = (pv > 0.5).float()
                iou = ((pb * vy).sum((1, 2, 3)) / ((pb + vy).clamp(max=1).sum((1, 2, 3)) + 1e-6)).mean().item()
                vloss = nn.functional.binary_cross_entropy(pv, vy).item()
            print(f"it {it:6d}  loss {run:.4f}  val bce {vloss:.4f}  val IoU {iou:.4f}  "
                  f"{(time.time() - t0) / it:.2f} s/it", flush=True)
            torch.save(model.state_dict(), ckpt)
            if vloss < best:
                best = vloss
                torch.save(model.state_dict(), ckpt.with_name(ckpt.stem + "_best.pt"))
    for p in procs:
        p.terminate()
    export(model, out)


def export(model, out: Path):
    import torch

    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, SIZE, SIZE)
    torch.onnx.export(model, dummy, str(out), input_names=["image"], output_names=["prob"], opset_version=13,
                      dynamo=False)
    net = cv2.dnn.readNetFromONNX(str(out))
    net.setInput(np.zeros((1, 3, SIZE, SIZE), np.float32))
    assert net.forward().shape == (1, 1, SIZE, SIZE)
    print(f"exported {out} ({out.stat().st_size / 1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("cord")
    p.add_argument("backgrounds")
    p.add_argument("cache")
    t = sub.add_parser("train")
    t.add_argument("cache")
    t.add_argument("--out", default=str(ROOT / "app" / "models" / "receipt_seg.onnx"))
    t.add_argument("--iters", type=int, default=20000)
    t.add_argument("--batch", type=int, default=24)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--resume")
    e = sub.add_parser("export")
    e.add_argument("weights")
    e.add_argument("--out", default=str(ROOT / "app" / "models" / "receipt_seg.onnx"))
    d = sub.add_parser("preview")
    d.add_argument("cache")
    d.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(Path(args.cord), Path(args.backgrounds), Path(args.cache))
    elif args.cmd == "train":
        train(Path(args.cache), Path(args.out), args.iters, args.batch, args.workers, args.lr, args.resume)
    elif args.cmd == "export":
        import torch
        m = build_model()
        m.load_state_dict(torch.load(args.weights))
        export(m, Path(args.out))
    elif args.cmd == "preview":
        rng = np.random.default_rng(5)
        s = Samples(Path(args.cache))
        tiles = []
        for _ in range(24):
            x, y = s.get(rng)
            v = x.copy()
            cnt, _ = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(v, cnt, -1, (0, 255, 0), 2)
            tiles.append(v)
        cv2.imwrite(args.out, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 24, 8)]))


if __name__ == "__main__":
    main()
