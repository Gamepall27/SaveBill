"""Evaluate receipt corner detection against labelled photos.

Dataset layout: <dir>/manifest.json  ([{"name", "roi": [[x,y]x4] | null}])
                <dir>/img/<name>

Usage:  python tools/eval_detection.py <dataset-dir> [--overlays <out-dir>] [--all] [--labels <json>]

For CORD use --labels tools/cord_labels_verified.json: the original CORD outlines are often
loose (several percent of the photo next to the paper edge); these were re-checked by hand
corner by corner.
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


def load_labelled(path: Path, roi):
    """Load a photo like the app does and scale its label to the (possibly downscaled) image."""
    data = path.read_bytes()
    img = scanner.load_image(data)
    if roi is None:
        return img, None
    import io
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as im:
        ow, oh = ImageOps.exif_transpose(im).size
    h, w = img.shape[:2]
    return img, (np.asarray(roi, np.float32) * np.array([w / ow, h / oh], np.float32)).tolist()


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
    ap.add_argument("--classic", action="store_true", help="without the segmentation network")
    ap.add_argument("--model", help="use this ONNX model instead of app/models/receipt_seg.onnx")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override scanner constants, e.g. SEG_MIN_IOU=0.6")
    ap.add_argument("--labels", help="JSON with verified corners ({labels: {name: {roi}}}) overriding the manifest")
    args = ap.parse_args()
    if args.model:
        scanner.SEG_PATH = Path(args.model)
    if args.classic:
        scanner._seg_state.update(loaded=True, net=None)
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(scanner, k, eval(v))  # noqa: S307 (local tool)

    root = Path(args.dataset)
    items = json.loads((root / "manifest.json").read_text())
    if args.labels:
        verified = json.loads(Path(args.labels).read_text())["labels"]
        items = [dict(it, roi=verified[it["name"]]["roi"]) if it["name"] in verified else dict(it, roi=None)
                 for it in items]
    if args.overlays:
        Path(args.overlays).mkdir(parents=True, exist_ok=True)

    results, t0 = [], time.time()
    for it in items:
        if it.get("roi") is None and not args.all:
            continue
        p = root / "img" / it["name"]
        if not p.exists():
            continue
        img, roi = load_labelled(p, it.get("roi"))
        h, w = img.shape[:2]
        pred, found = scanner.detect_corners(img)
        iou = None
        if roi is not None:
            iou = quad_iou(pred, roi, w, h)
            results.append((iou, found, it["name"]))
        if args.overlays:
            tag = "nogt" if iou is None else f"{iou:.2f}"
            overlay(img, pred, roi, Path(args.overlays) / f"{tag}_{it['name']}")

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
