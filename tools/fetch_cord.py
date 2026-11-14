"""Download real receipt photos with labelled receipt corners.

Source: CORD v2 (naver-clova-ix/cord-v2 on Hugging Face, CC BY 4.0). Only images with an
annotated receipt outline ("roi") are labelled; the others are kept for visual checks
(`eval_detection.py --all`).

- default:  test + validation split (200 photos, 77 labelled) for evaluating detection
- --train:  the 800 training photos (for tools/train_segmenter.py). The datasets server
            does not serve this split, so the parquet files are downloaded (~1.8 GB,
            needs pyarrow).

Usage:  python tools/fetch_cord.py <out-dir> [--train]
Then:   python tools/eval_detection.py <out-dir> [--overlays <dir>]
"""
import io
import json
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix/cord-v2&config=default"
REPO = "https://huggingface.co/datasets/naver-clova-ix/cord-v2"


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


def roi_of(ground_truth: str):
    roi = json.loads(ground_truth).get("roi")
    return [[roi[f"x{i}"], roi[f"y{i}"]] for i in (1, 2, 3, 4)] if roi else None


def fetch_eval(out: Path) -> list:
    manifest = []
    for split, tag in (("test", "test"), ("validation", "val")):
        for r in rows(split):
            manifest.append({"name": f"{tag}_{r['row_idx']:03d}.jpg", "url": r["row"]["image"]["src"],
                             "roi": roi_of(r["row"]["ground_truth"])})

    def fetch(item):
        target = out / "img" / item["name"]
        if not target.exists():
            with urllib.request.urlopen(item["url"], timeout=60) as r:
                target.write_bytes(r.read())

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, manifest))
    for m in manifest:
        m.pop("url")
    return manifest


def fetch_train(out: Path) -> list:
    import pyarrow.parquet as pq

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/tree/main/data",
                                timeout=60) as r:
        files = sorted(f["path"] for f in json.load(r) if f["path"].split("/")[-1].startswith("train-"))
    manifest, idx = [], 0
    for path in files:
        with tempfile.TemporaryFile() as tmp:
            with urllib.request.urlopen(f"{REPO}/resolve/main/{path}", timeout=600) as r:
                while chunk := r.read(1 << 20):
                    tmp.write(chunk)
            tmp.seek(0)
            table = pq.read_table(io.BytesIO(tmp.read()), columns=["image", "ground_truth"]).to_pylist()
        for row in table:
            name = f"train_{idx:03d}.jpg"
            (out / "img" / name).write_bytes(row["image"]["bytes"])
            manifest.append({"name": name, "roi": roi_of(row["ground_truth"])})
            idx += 1
        print(f"{path}: {len(table)} rows", flush=True)
    return manifest


def main():
    out = Path(sys.argv[1])
    (out / "img").mkdir(parents=True, exist_ok=True)
    manifest = fetch_train(out) if "--train" in sys.argv[2:] else fetch_eval(out)
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"{len(manifest)} images, {sum(1 for m in manifest if m['roi'])} with corner labels -> {out}")


if __name__ == "__main__":
    main()
