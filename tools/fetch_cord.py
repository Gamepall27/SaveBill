"""Download real receipt photos with labelled receipt corners for evaluating detection.

Source: CORD v2 (naver-clova-ix/cord-v2 on Hugging Face, CC BY 4.0), test + validation
splits. Only images with an annotated receipt outline ("roi") are labelled; the others are
kept for visual checks (`eval_detection.py --all`).

Usage:  python tools/fetch_cord.py <out-dir>
Then:   python tools/eval_detection.py <out-dir> [--overlays <dir>]
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix/cord-v2&config=default"


def rows(split: str):
    out, offset = [], 0
    while True:
        with urllib.request.urlopen(f"{API}&split={split}&offset={offset}&length=100", timeout=60) as r:
            data = json.load(r)
        batch = data.get("rows", [])
        out += batch
        if len(batch) < 100:
            return out
        offset += 100


def main():
    out = Path(sys.argv[1])
    (out / "img").mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, tag in (("test", "test"), ("validation", "val")):
        for r in rows(split):
            gt = json.loads(r["row"]["ground_truth"])
            roi = gt.get("roi")
            manifest.append({
                "name": f"{tag}_{r['row_idx']:03d}.jpg",
                "url": r["row"]["image"]["src"],
                "roi": [[roi[f"x{i}"], roi[f"y{i}"]] for i in (1, 2, 3, 4)] if roi else None,
            })

    def fetch(item):
        target = out / "img" / item["name"]
        if not target.exists():
            with urllib.request.urlopen(item["url"], timeout=60) as r:
                target.write_bytes(r.read())

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, manifest))
    for m in manifest:
        m.pop("url")
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"{len(manifest)} images, {sum(1 for m in manifest if m['roi'])} with corner labels -> {out}")


if __name__ == "__main__":
    main()
