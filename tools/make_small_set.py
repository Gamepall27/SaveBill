"""Build a labelled test set of *small* receipts (3-25 % of the photo).

Real receipts are cut out of the labelled CORD photos (see fetch_cord.py) and placed with
random size, rotation and perspective onto real background photos, with a soft shadow,
uneven lighting, sensor noise and JPEG compression. The exact corners are written to a
manifest that tools/eval_detection.py understands.

Usage:  python tools/make_small_set.py <cord-dir> <background-dir> <out-dir> [--n 150] [--seed 7]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import scanner  # noqa: E402


def receipt_textures(cord: Path) -> list[np.ndarray]:
    out = []
    for it in json.loads((cord / "manifest.json").read_text()):
        roi = it.get("roi")
        if not roi:
            continue
        img = cv2.imread(str(cord / "img" / it["name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        q = np.array(roi, np.float32)
        # Only receipts that are completely visible give clean edges.
        if (q[:, 0] < 2).any() or (q[:, 1] < 2).any() or (q[:, 0] > w - 3).any() or (q[:, 1] > h - 3).any():
            continue
        out.append(scanner.warp(img, q))
    return out


def lighting(h, w, rng):
    """Smooth multiplicative illumination field (0.75 .. 1.1)."""
    small = rng.uniform(0.75, 1.1, (3, 3)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[:, :, None]


def compose(tex, bg, rng, frac_range):
    portrait = rng.random() < 0.75
    W, H = (1200, 1600) if portrait else (1600, 1200)
    # random crop of the background, scaled to the frame
    bh, bw = bg.shape[:2]
    s = max(W / bw, H / bh) * rng.uniform(1.0, 1.6)
    bgs = cv2.resize(bg, (int(bw * s) + 1, int(bh * s) + 1), interpolation=cv2.INTER_AREA)
    y0 = rng.integers(0, bgs.shape[0] - H + 1)
    x0 = rng.integers(0, bgs.shape[1] - W + 1)
    canvas = bgs[y0:y0 + H, x0:x0 + W].astype(np.float32)

    th, tw = tex.shape[:2]
    frac = rng.uniform(*frac_range)
    area = frac * W * H
    rh = np.sqrt(area * th / tw)
    rw = area / rh
    angle = np.deg2rad(rng.uniform(-35, 35))
    rect = np.array([[-rw / 2, -rh / 2], [rw / 2, -rh / 2], [rw / 2, rh / 2], [-rw / 2, rh / 2]], np.float32)
    rect += rng.uniform(-0.08, 0.08, rect.shape).astype(np.float32) * np.array([rw, rh], np.float32)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    rect = rect @ rot.T
    margin = np.abs(rect).max(axis=0) + 10
    cx = rng.uniform(margin[0], W - margin[0]) if W > 2 * margin[0] else W / 2
    cy = rng.uniform(margin[1], H - margin[1]) if H > 2 * margin[1] else H / 2
    quad = rect + np.array([cx, cy], np.float32)

    src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(tex, m, (W, H), flags=cv2.INTER_AREA).astype(np.float32)
    mask = cv2.warpPerspective(np.ones((th, tw), np.float32), m, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[:, :, None]

    # soft drop shadow
    off = rng.uniform(4, 14, 2)
    shadow = cv2.warpAffine(mask[:, :, 0], np.float32([[1, 0, off[0]], [0, 1, off[1]]]), (W, H))
    shadow = cv2.GaussianBlur(shadow, (0, 0), rng.uniform(6, 14))[:, :, None]
    canvas *= 1 - 0.45 * shadow

    out = canvas * (1 - mask) + warped * mask
    out *= lighting(H, W, rng)
    out += rng.normal(0, rng.uniform(2, 6), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 92))])
    return buf.tobytes(), quad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cord")
    ap.add_argument("backgrounds")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min", type=float, default=0.03)
    ap.add_argument("--max", type=float, default=0.25)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    textures = receipt_textures(Path(args.cord))
    bgs = [b for b in (cv2.imread(str(p)) for p in sorted(Path(args.backgrounds).glob("*.jpg"))) if b is not None]
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(args.n):
        data, quad = compose(textures[rng.integers(len(textures))], bgs[rng.integers(len(bgs))], rng,
                             (args.min, args.max))
        name = f"small_{i:03d}.jpg"
        (out / "img" / name).write_bytes(data)
        manifest.append({"name": name, "roi": np.round(quad, 1).tolist()})
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"{len(textures)} receipts x {len(bgs)} backgrounds -> {args.n} images in {out}")


if __name__ == "__main__":
    main()
