"""End-to-end check of the receipt pipeline (detect -> crop -> OCR -> suggested total).

Needs Tesseract, so it is meant to run inside the container. To test code that is not
deployed yet, copy the app package to /tmp/newapp in the container first:
    tar c app | docker exec -i savebill sh -c "rm -rf /tmp/newapp; mkdir /tmp/newapp; tar x -C /tmp/newapp"
    docker exec -i savebill python - /tmp/bench/real < tools/ocr_dump.py

<root>/img/*.jpg (+ optional manifest.json with "total") -> <root>/txt/<name>.txt and a score.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tmp/newapp" if os.path.isdir("/tmp/newapp/app") else "/app")
from app import ocr, scanner  # noqa: E402

root = Path(sys.argv[1])
manifest = root / "manifest.json"
totals = {m["name"]: m.get("total") for m in json.loads(manifest.read_text())} if manifest.exists() else {}
(root / "txt").mkdir(exist_ok=True)
ok = n = 0
t_all = time.time()
for p in sorted((root / "img").glob("*.jpg")):
    t = time.time()
    img = scanner.load_image(p.read_bytes())
    corners, _ = scanner.detect_corners(img)
    text, sugg = ocr.analyze(scanner.warp(img, corners))
    (root / "txt" / (p.stem + ".txt")).write_text(text, encoding="utf-8")
    want = totals.get(p.name)
    if want is not None:
        n += 1
        ok += sugg["amount"] == want
    print(f"{'OK ' if sugg['amount'] == want else 'XX '}{p.name}: want={want} got={sugg['amount']} "
          f"{time.time() - t:.1f}s", flush=True)
if n:
    print(f"\n{ok}/{n} totals correct, {(time.time() - t_all) / max(n, 1):.1f}s per receipt")
