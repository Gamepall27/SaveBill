"""Receipt detection and perspective correction."""
import io
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config

DETECT_SIDE = 800  # detection runs on a downscaled copy


def load_image(data: bytes, max_side: int = config.MAX_SIDE) -> np.ndarray:
    """Decode upload bytes, apply EXIF orientation, downscale, return BGR array."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        arr = np.asarray(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def order_corners(pts) -> np.ndarray:
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def full_frame(w: int, h: int) -> list[list[float]]:
    return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]


def _candidate_masks(gray: np.ndarray) -> list[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Receipts are usually brighter than the surface they lie on.
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    big = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, big, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
    return [bright, edges]


def _quad_from_contour(c: np.ndarray) -> tuple[np.ndarray, bool]:
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), True
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32), False


def detect_corners(img: np.ndarray) -> tuple[list[list[float]], bool]:
    """Find the receipt outline. Returns (corners TL,TR,BR,BL in image coords, found)."""
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_SIDE / max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    frame_area = sh * sw

    best = None  # (score, quad)
    for mask in _candidate_masks(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cv2.convexHull(c))
            if area < 0.08 * frame_area:
                break
            if area > 0.97 * frame_area:
                continue  # that's just the photo border
            quad, exact = _quad_from_contour(c)
            quad_area = cv2.contourArea(quad)
            # Prefer exact quadrilaterals, then larger ones.
            score = quad_area * (1.0 if exact else 0.7)
            if best is None or score > best[0]:
                best = (score, quad)

    if best is None:
        return full_frame(w, h), False

    quad = order_corners(best[1]) / scale
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in quad], True


def warp(img: np.ndarray, corners) -> np.ndarray:
    pts = order_corners(corners)
    tl, tr, br, bl = pts
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    width, height = max(width, 10), max(height, 10)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhance(img: np.ndarray) -> np.ndarray:
    """Gentle contrast boost + sharpening; keeps colour."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def crop_receipt(img: np.ndarray, corners) -> np.ndarray:
    return enhance(warp(img, corners))


def thumbnail(img: np.ndarray, width: int = 400) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def save_jpeg(img: np.ndarray, path, quality: int = 88) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img
