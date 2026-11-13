"""Learn the receipt-total scorer (logistic regression over app.amount.FEATURE_NAMES).

Datasets: <bons>/<set>/manifest.json ([{"name", "total"}]) and OCR texts in
<bons>/<set>/variants/<variant>/<stem>.txt (tools/ocr_variants.py, run in the container).
Every amount found in a receipt's OCR text is a candidate, labelled "good" if it equals the
printed total. Each dataset counts the same, each receipt within it counts the same.

Usage:
  python tools/train_amount.py <bons> --train sroie_train nadel synth --eval sroie_test nadel_eval fresh real
                               [--primary norm1400_p4] [--secondary otsu1400_p6]
Prints accuracy of the hand-made and the learned scorer and the constants for app/amount.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import amount, ocr  # noqa: E402
from train_scoring import fit  # noqa: E402


def load(bons: Path, name: str, primary: str, secondary: str | None):
    root = bons / name
    rows = []
    for it in json.loads((root / "manifest.json").read_text()):
        stem = Path(it["name"]).stem
        p = root / "variants" / primary / f"{stem}.txt"
        if not p.exists():
            continue
        texts = [p.read_text(encoding="utf-8")]
        if secondary and amount.confidence(texts[0]) < ocr.CONFIDENT:
            q = root / "variants" / secondary / f"{stem}.txt"
            if q.exists():
                texts.append(q.read_text(encoding="utf-8"))
        rows.append((texts, round(float(it["total"]), 2)))
    return rows


def accuracy(rows, use_learned: bool, w=None, b=0.0):
    saved = amount.SCORER_W, amount.SCORER_B
    amount.SCORER_W, amount.SCORER_B = (tuple(w), b) if use_learned else (None, 0.0)
    try:
        return float(np.mean([amount.suggest(t) == tot for t, tot in rows])) if rows else float("nan")
    finally:
        amount.SCORER_W, amount.SCORER_B = saved


def reachable(rows):
    return float(np.mean([tot in amount.evidence(t) for t, tot in rows])) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bons")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--primary", default="norm1400_p4")
    ap.add_argument("--secondary", default="otsu1400_p6")
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()
    bons = Path(args.bons)

    xs, ys, ws = [], [], []
    train = {}
    for name in args.train:
        rows = load(bons, name, args.primary, args.secondary)
        train[name] = rows
        for texts, tot in rows:
            ev = amount.evidence(texts)
            if not ev:
                continue
            for a, e in ev.items():
                xs.append(amount.features(e))
                ys.append(float(a == tot))
                ws.append(1.0 / (len(ev) * len(rows)))
    x, y, wt = np.array(xs), np.array(ys), np.array(ws)
    w, b = fit(x, y, wt, l2=args.l2)

    print(f"{'set':14s} {'n':>4s} {'reach':>6s} {'hand':>6s} {'learned':>8s}")
    for name in list(args.train) + list(args.eval):
        rows = train.get(name) or load(bons, name, args.primary, args.secondary)
        tag = "train" if name in train else "eval"
        print(f"{name:14s} {len(rows):4d} {reachable(rows):6.2f} {accuracy(rows, False):6.2f} "
              f"{accuracy(rows, True, w, b):8.2f}  ({tag})")
    print()
    for n_, v in zip(amount.FEATURE_NAMES, w):
        print(f"  {n_:14s} {v:+.3f}")
    print(f"\nSCORER_W = ({', '.join(f'{v:.4f}' for v in w)})")
    print(f"SCORER_B = {b:.4f}")


if __name__ == "__main__":
    main()
