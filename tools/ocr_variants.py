"""Compare OCR preprocessing variants on a folder of photos (run inside the container).

    docker exec -i savebill python - /tmp/bench/real < tools/ocr_variants.py
Writes <root>/variants/<variant>/<name>.txt
"""
import sys
import time
from pathlib import Path

import cv2
import pytesseract

sys.path.insert(0, "/app")
from app import scanner  # noqa: E402


def scale_to(gray, width):
    h, w = gray.shape
    f = width / w
    return cv2.resize(gray, (width, int(h * f)), interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)


def otsu(g):
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive(g, block=31, c=15):
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)


def bg_norm(g):
    """Divide by a blurred background estimate: removes shading, keeps strokes."""
    bg = cv2.medianBlur(cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))), 31)
    return cv2.normalize(cv2.divide(g, bg, scale=255), None, 0, 255, cv2.NORM_MINMAX)


VARIANTS = {
    "cur": lambda warped, enh: (scale_to(cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY), 1000), "--psm 4"),
    "raw1400_p4": lambda w, e: (scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400), "--psm 4"),
    "raw1400_p6": lambda w, e: (scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400), "--psm 6"),
    "otsu1400_p4": lambda w, e: (otsu(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400)), "--psm 4"),
    "norm1400_p4": lambda w, e: (bg_norm(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400)), "--psm 4"),
    "norm1400_p6": lambda w, e: (bg_norm(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400)), "--psm 6"),
    "adapt1400_p4": lambda w, e: (adaptive(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400)), "--psm 4"),
    "otsu1400_p6": lambda w, e: (otsu(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400)), "--psm 6"),
    "otsu1400_p6s": lambda w, e: (otsu(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1400)),
                                  "--psm 6 -c preserve_interword_spaces=1"),
    "otsu1800_p6": lambda w, e: (otsu(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 1800)), "--psm 6"),
    "norm2000_p4": lambda w, e: (bg_norm(scale_to(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 2000)), "--psm 4"),
}

root = Path(sys.argv[1])
only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
for p in sorted((root / "img").glob("*.jpg")):
    img = scanner.load_image(p.read_bytes())
    corners, _ = scanner.detect_corners(img)
    warped = scanner.warp(img, corners)
    enh = scanner.enhance(warped)
    for name in only:
        t = time.time()
        g, cfg = VARIANTS[name](warped, enh)
        try:
            text = pytesseract.image_to_string(g, lang="deu+eng", config=cfg, timeout=120)
        except Exception as e:  # noqa: BLE001
            text = ""
            print("  fail", name, e)
        d = root / "variants" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / (p.stem + ".txt")).write_text(text, encoding="utf-8")
        print(f"{p.name} {name}: {len(text)} chars {time.time() - t:.1f}s", flush=True)
