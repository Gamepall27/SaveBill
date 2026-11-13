"""Compare OCR preprocessing variants on a folder of photos (run inside the container).

    docker exec -i savebill python - /tmp/bench/real < tools/ocr_variants.py
Writes <root>/variants/<variant>/<name>.txt
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
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


def ink_channel(bgr):
    """The colour channel in which print and paper separate best (Otsu between-class variance)."""
    b, g, r = cv2.split(bgr)
    cands = {"gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "b": b, "g": g, "r": r, "min": cv2.min(cv2.min(b, g), r)}
    best, best_score = None, -1.0
    for name, ch in cands.items():
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel()
        p = hist / hist.sum()
        levels = np.arange(256)
        w0 = np.cumsum(p)
        mu = np.cumsum(p * levels)
        mu_t = mu[-1]
        var_t = float(np.sum(p * (levels - mu_t) ** 2)) + 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            between = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
        score = float(np.nanmax(between)) / var_t
        if score > best_score:
            best, best_score = ch, score
    return best


def text_height_scale(gray, target=30.0):
    """Scale so that capital letters are ~target px high (estimated from connected components)."""
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2)) if gray.shape[1] > 1600 else gray
    f0 = gray.shape[1] / small.shape[1]
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    hs = [st[3] for st in stats[1:] if 4 <= st[3] <= 120 and st[2] <= 3 * st[3] and st[4] >= 6]
    if len(hs) < 20:
        return scale_to(gray, 1400)
    h = float(np.percentile(hs, 75)) * f0
    f = float(np.clip(target / h, 0.4, 4.0))
    w = int(np.clip(gray.shape[1] * f, 600, 3000))
    return scale_to(gray, w)


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
    "inknorm1400_p4": lambda w, e: (bg_norm(scale_to(ink_channel(w), 1400)), "--psm 4"),
    "inkotsu1400_p6": lambda w, e: (otsu(scale_to(ink_channel(w), 1400)), "--psm 6"),
    "normH_p4": lambda w, e: (bg_norm(text_height_scale(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY))), "--psm 4"),
    "otsuH_p6": lambda w, e: (otsu(text_height_scale(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY))), "--psm 6"),
    "inknormH_p4": lambda w, e: (bg_norm(text_height_scale(ink_channel(w))), "--psm 4"),
    "inkotsuH_p6": lambda w, e: (otsu(text_height_scale(ink_channel(w))), "--psm 6"),
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
