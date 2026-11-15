"""Write language-neutral reference data for ports of the receipt logic (e.g. Kotlin/Swift).

The Python code in app/ is the reference. This tool records what it computes, so that a
port can be checked against the same inputs (see tests/conformance/README.md):

  suggest_cases.json      OCR text(s) -> title, date, category, total, alternatives,
                          confidence, ranked candidates and the classification of every line
  segmentation/           network input (256x256 PNG) -> probability map (16-bit PNG)
  detection_synth/        synthetic photos (JPEG) -> detected corners
  detection_cord.json     CORD photos (download with tools/fetch_cord.py) -> corners + IoU

tests/test_conformance.py fails when app/ and these files disagree; regenerate on purpose:
    python tools/export_conformance.py [--cord <cord-dir>]
"""
import argparse
import ast
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from app import amount, ocr, scanner  # noqa: E402
from tests import synth  # noqa: E402
from tests.test_scanner import CASES  # noqa: E402

OUT = ROOT / "tests" / "conformance"
TODAY = date(2026, 11, 13)  # fixed "today" for the date plausibility check
SEG_SAMPLES = {  # CORD photos (CC BY 4.0) used as network inputs
    "cord_test_000": "test_000.jpg", "cord_test_043": "test_043.jpg", "cord_val_016": "val_016.jpg",
}


# ---------------------------------------------------------------- text -> suggestions

def _texts_from_tests() -> list[tuple[str, list[str]]]:
    """All receipt-like string literals of tests/test_ocr.py plus the real OCR fixtures."""
    cases, seen = [], set()
    tree = ast.parse((ROOT / "tests" / "test_ocr.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            t = node.value
            if t in seen or len(t) < 8 or not any(c.isdigit() for c in t):
                continue
            seen.add(t)
            cases.append((f"test_ocr.py:{node.lineno}", [t]))
    for p in sorted((ROOT / "tests" / "fixtures" / "ocr").glob("*.txt")):
        cases.append((f"fixtures/ocr/{p.name}", [p.read_text(encoding="utf-8")]))
    # several OCR passes of the same receipt are evaluated together
    cases.append(("multi-pass", ["Brot 2,49\nSumme ?\nEC-Karte\n", "Brot 2,49\nSumme 7,48\nEC-Karte 7,48\n"]))
    fx = ROOT / "tests" / "fixtures" / "ocr"
    for a in sorted(fx.glob("*a.txt")):
        b = a.with_name(a.stem[:-1] + "b.txt")
        if b.exists():
            cases.append((f"multi-pass fixtures/ocr/{a.name}+{b.name}",
                          [a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8")]))
    return cases


def suggest_case(cid: str, source: str, texts: list[str]) -> dict:
    first = texts[0]
    return {
        "id": cid, "source": source, "texts": texts, "today": TODAY.isoformat(),
        "expected": {
            "title": ocr.suggest_title(first),
            "date": ocr.suggest_date(first, today=TODAY),
            "category": ocr.suggest_category(first),
            "amount": amount.suggest(texts),
            "alternatives": amount.alternatives(texts),
            "confidence": round(amount.confidence(texts), 6),
            "ranked": [[a, round(s, 6)] for a, s in amount.ranked(texts)[:8]],
            "lines": [{"raw": r["raw"], "amounts": r["amounts"], "kind": r["kind"], "orphan": r["orphan"],
                       "item": r["item"], "tabular": r["tabular"], "borrowed": r["borrowed"]}
                      for r in amount._lines(first)],
        },
    }


# ---------------------------------------------------------------- segmentation network

def letterbox(img: np.ndarray) -> np.ndarray:
    """Exactly the network input of scanner.receipt_probability (BGR uint8, 256x256)."""
    h, w = img.shape[:2]
    s = scanner.SEG_SIZE / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.full((scanner.SEG_SIZE, scanner.SEG_SIZE, 3), scanner.SEG_PAD, np.uint8)
    x0, y0 = (scanner.SEG_SIZE - nw) // 2, (scanner.SEG_SIZE - nh) // 2
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas


def net_output(canvas: np.ndarray) -> np.ndarray:
    net = scanner._seg_net()
    net.setInput(canvas.transpose(2, 0, 1)[None].astype(np.float32))
    return net.forward()[0, 0]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cord", help="CORD test/validation dir (tools/fetch_cord.py) for detection_cord.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [suggest_case(f"s{i:03d}", src, texts) for i, (src, texts) in enumerate(_texts_from_tests())]
    (OUT / "suggest_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"suggest_cases.json: {len(cases)} cases")

    seg = OUT / "segmentation"
    seg.mkdir(exist_ok=True)
    inputs = {"synth_perspective": synth.photo(CASES[2]), "synth_small": synth.photo(CASES[3])}
    if args.cord:
        for name, file in SEG_SAMPLES.items():
            inputs[name] = scanner.load_image((Path(args.cord) / "img" / file).read_bytes())
    for name, img in inputs.items():
        canvas = letterbox(img)
        cv2.imwrite(str(seg / f"{name}_input.png"), canvas)
        prob = net_output(cv2.imread(str(seg / f"{name}_input.png")))
        cv2.imwrite(str(seg / f"{name}_prob.png"), np.round(np.clip(prob, 0, 1) * 65535).astype(np.uint16))
    print(f"segmentation/: {len(inputs)} inputs")

    syn = OUT / "detection_synth"
    syn.mkdir(exist_ok=True)
    expected = []
    for i, truth in enumerate(CASES):
        path = syn / f"case{i}.jpg"
        cv2.imwrite(str(path), synth.photo(truth), [cv2.IMWRITE_JPEG_QUALITY, 80])
        img = scanner.load_image(path.read_bytes())
        corners, found = scanner.detect_corners(img)
        expected.append({"file": path.name, "width": img.shape[1], "height": img.shape[0],
                         "truth": truth, "corners": corners, "found": found})
    blank = syn / "blank.jpg"
    cv2.imwrite(str(blank), np.full((600, 400, 3), 128, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    corners, found = scanner.detect_corners(scanner.load_image(blank.read_bytes()))
    expected.append({"file": blank.name, "width": 400, "height": 600, "truth": None, "corners": corners,
                     "found": found})
    (syn / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    print(f"detection_synth/: {len(expected)} photos")

    if args.cord:
        from eval_detection import load_labelled, quad_iou
        verified = json.loads((ROOT / "tools" / "cord_labels_verified.json").read_text())["labels"]
        rows = []
        for name, lab in sorted(verified.items()):
            img, roi = load_labelled(Path(args.cord) / "img" / name, lab["roi"])
            h, w = img.shape[:2]
            corners, found = scanner.detect_corners(img)
            rows.append({"name": name, "width": w, "height": h, "label": np.round(roi, 1).tolist(),
                         "corners": corners, "found": found, "iou": round(quad_iou(corners, roi, w, h), 4)})
        ious = [r["iou"] for r in rows]
        (OUT / "detection_cord.json").write_text(json.dumps(
            {"mean_iou": round(float(np.mean(ious)), 4), "images": rows}, indent=1), encoding="utf-8")
        print(f"detection_cord.json: {len(rows)} photos, mean IoU {np.mean(ious):.3f}")


if __name__ == "__main__":
    main()
