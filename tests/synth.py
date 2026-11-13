"""Synthetic receipt photos for tests and local demos."""
import cv2
import numpy as np

LINES = [
    "REWE Markt GmbH",
    "Hauptstrasse 12",
    "12345 Musterstadt",
    "",
    "Milch 1,5%          1,19",
    "Brot                2,49",
    "Kaffee             6,99",
    "Bananen            1,85",
    "------------------------",
    "SUMME EUR          12,52",
    "Geg. BAR           20,00",
    "Rueckgeld           7,48",
    "",
    "Datum: 13.11.2026  14:32",
]


def receipt_paper(w: int = 420, h: int = 900) -> np.ndarray:
    paper = np.full((h, w, 3), 245, np.uint8)
    y = 50
    for line in LINES:
        cv2.putText(paper, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
        y += 42
    return paper


def photo(corners_dst, size=(1200, 1600), seed: int = 1) -> np.ndarray:
    """Place the receipt on a darker textured surface with the given destination corners."""
    rng = np.random.default_rng(seed)
    w, h = size
    bg = np.full((h, w, 3), (95, 80, 70), np.uint8)
    noise = rng.normal(0, 12, (h, w, 3))
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    paper = receipt_paper()
    ph, pw = paper.shape[:2]
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    m = cv2.getPerspectiveTransform(src, np.float32(corners_dst))
    warped = cv2.warpPerspective(paper, m, (w, h))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (w, h))
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out
