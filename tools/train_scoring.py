"""Learn the candidate scorer (logistic regression) from labelled photos.

Every candidate region the detector proposes is described by app.scanner._features and
labelled "good" when it overlaps the true receipt with IoU >= --good. The learned weights
are printed as Python constants for app/scanner.py (SCORER_W, SCORER_B, SCORER_MIN).

Usage:  python tools/train_scoring.py <dataset-dir> [<dataset-dir> ...]
        (datasets in the eval_detection.py layout, e.g. from fetch_cord.py / make_small_set.py)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import scanner  # noqa: E402
from eval_detection import quad_iou  # noqa: E402


def collect(dirs, split=None):
    """Per image: (features [n,k], ious [n], fallback_iou)."""
    images = []
    for d in dirs:
        root = Path(d)
        for it in json.loads((root / "manifest.json").read_text()):
            if not it.get("roi") or (split and not it["name"].startswith(split)):
                continue
            img = scanner.load_image((root / "img" / it["name"]).read_bytes())
            h, w = img.shape[:2]
            scale, small, gray, white, edges = scanner._prepare(img)
            sh, sw = gray.shape
            roi = np.array(it["roi"], np.float32) * scale
            feats, ious = [], []
            for q, c, ex, is_box in scanner._candidates(small, gray, scanner.MIN_AREA):
                feats.append(scanner._features(q, c, white, edges, ex, is_box))
                ious.append(quad_iou(q, roi, sw, sh))
            fallback = quad_iou(scanner.full_frame(sw, sh), roi, sw, sh)
            images.append((np.array(feats).reshape(-1, len(scanner.FEATURE_NAMES)), np.array(ious), fallback))
    return images


def fit(x, y, weights, l2=1e-2, epochs=4000, lr=0.1):
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = (x - mu) / sd
    pos = np.sum(weights * y) / np.sum(weights)
    sw = weights * np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))  # balance classes
    sw = sw / sw.mean()
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        g = (p - y) * sw
        w -= lr * (z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    # fold standardisation into raw-feature weights
    return w / sd, b - np.sum(w * mu / sd)


def select_iou(images, w, b, thr):
    out = []
    for f, ious, fallback in images:
        if len(f) == 0:
            out.append(fallback)
            continue
        p = 1 / (1 + np.exp(-(f @ w + b)))
        k = int(np.argmax(p))
        out.append(ious[k] if p[k] >= thr else fallback)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train", nargs="+")
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--good", type=float, default=0.85)
    args = ap.parse_args()

    # Every dataset counts the same, and within it every image counts the same.
    train, xs, ys, ws = [], [], [], []
    for d in args.train:
        imgs = collect([d])
        train += imgs
        for f, i, _ in imgs:
            if len(f):
                xs.append(f)
                ys.append((i >= args.good).astype(float))
                ws.append(np.full(len(f), 1.0 / (len(f) * len(imgs))))
    x, y, wts = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    print(f"train: {len(train)} images, {len(y)} candidates, {int(y.sum())} good")
    w, b = fit(x, y, wts)
    sets = [collect([d]) for d in args.train]
    best = max(((np.mean([select_iou(s_, w, b, t).mean() for s_ in sets]), t) for t in np.linspace(0.05, 0.95, 19)))
    print(f"train mean IoU (pre-refine): {best[0]:.3f} at threshold {best[1]:.2f}")
    for d in args.eval:
        ev = collect([d])
        print(f"eval {d}: mean IoU (pre-refine) {select_iou(ev, w, b, best[1]).mean():.3f}")
    print()
    for n, v in zip(scanner.FEATURE_NAMES, w):
        print(f"  {n:16s} {v:+.4f}")
    print(f"\nSCORER_W = ({', '.join(f'{v:.4f}' for v in w)})")
    print(f"SCORER_B = {b:.4f}")
    print(f"SCORER_MIN = {best[1]:.2f}")


if __name__ == "__main__":
    main()
