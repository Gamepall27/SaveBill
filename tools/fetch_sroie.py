"""Download real scanned receipts with their printed total (ICDAR 2019 SROIE, via Hugging Face).

Usage:  python tools/fetch_sroie.py <out-dir> [--split train|test] [--n 100] [--offset 0]
Writes <out>/img/*.jpg and <out>/manifest.json ([{"name", "total"}]) like make_german_bons.py.
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows?dataset=jsdnrs/ICDAR2019-SROIE&config=default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    rows, off = [], args.offset
    while len(rows) < args.n:
        n = min(100, args.n - len(rows))
        with urllib.request.urlopen(f"{API}&split={args.split}&offset={off}&length={n}", timeout=60) as r:
            batch = json.load(r).get("rows", [])
        if not batch:
            break
        rows += batch
        off += len(batch)

    manifest = []
    for r in rows:
        total = (r["row"]["entities"].get("total") or "").replace(",", "").strip()
        try:
            value = round(float(total), 2)
        except ValueError:
            continue  # unlabeled / malformed
        manifest.append({"name": f"sroie_{args.split}_{r['row_idx']:03d}.jpg", "total": value,
                         "url": r["row"]["image"]["src"]})

    def fetch(item):
        target = out / "img" / item["name"]
        if not target.exists():
            with urllib.request.urlopen(item["url"], timeout=60) as r:
                target.write_bytes(r.read())

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, manifest))
    for m in manifest:
        m.pop("url")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} receipts with total -> {out}")


if __name__ == "__main__":
    main()
