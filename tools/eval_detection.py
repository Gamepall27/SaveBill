"""Evaluate receipt corner detection against labelled photos.

Dataset layout: <dir>/manifest.json  ([{"name", "roi": [[x,y]x4] | null}])
                <dir>/img/<name>

Usage:  python tools/eval_detection.py <dataset-dir> [--overlays <out-dir>] [--all]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import scanner  # noqa: E402


def quad_iou(a, b, w, h) -> float:
    """IoU of two quadrilaterals, both clipped to the image frame."""
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(np.asarray(a)).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b)).astype(np.int32)], 1)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union else 0.0


def overlay(img, pred, gt, path):
    out = img.copy()
    if gt is not None:
        cv2.polylines(out, [np.round(np.asarray(gt)).astype(np.int32)], True, (0, 200, 0), 3)
    cv2.polylines(out, [np.round(np.asarray(pred)).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(path), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--overlays")
    ap.add_argument("--all", action="store_true", help="also process unlabelled images (overlays only)")
    args = ap.parse_args()

    root = Path(args.dataset)
    items = json.loads((root / "manifest.json").read_text())
    if args.overlays:
        Path(args.overlays).mkdir(parents=True, exist_ok=True)

    results, t0 = [], time.time()
    for it in items:
        if it.get("roi") is None and not args.all:
            continue
        p = root / "img" / it["name"]
        if not p.exists():
            continue
        img = scanner.load_image(p.read_bytes())
        h, w = img.shape[:2]
        pred, found = scanner.detect_corners(img)
        iou = None
        if it.get("roi") is not None:
            iou = quad_iou(pred, it["roi"], w, h)
            results.append((iou, found, it["name"]))
        if args.overlays:
            tag = "nogt" if iou is None else f"{iou:.2f}"
            overlay(img, pred, it.get("roi"), Path(args.overlays) / f"{tag}_{it['name']}")

    dt = time.time() - t0
    if not results:
        print("no labelled images")
        return
    ious = np.array([r[0] for r in results])
    print(f"images: {len(results)}  time/img: {dt / max(len(results), 1) * 1000:.0f} ms")
    print(f"mean IoU: {ious.mean():.3f}  median: {np.median(ious):.3f}")
    for t in (0.95, 0.9, 0.8):
        print(f"  IoU >= {t}: {(ious >= t).mean() * 100:5.1f} %")
    print(f"  found flag: {sum(r[1] for r in results)}/{len(results)}")
    print("worst:")
    for iou, found, name in sorted(results)[:10]:
        print(f"  {iou:.3f}  found={found}  {name}")


if __name__ == "__main__":
    main()
