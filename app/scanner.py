"""Receipt detection and perspective correction."""
import io
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config

DETECT_SIDE = 800  # detection runs on a downscaled copy
MIN_SCORE = 0.05  # below this the photo itself is assumed to be the receipt
THRESH_STEPS = ()  # extra binarisation levels between Otsu and white
SUPPORT_DIFF = 18  # min. whiteness step across a receipt edge
TOP_CONTOURS = 4
REFINE = True
CLOSE_BORDER = True
CONTRAST_WEIGHT = 1.0
MIN_AREA = 0.015  # smallest candidate (fraction of the photo)
MAX_AREA = 0.9  # larger candidates are treated as 'whole photo'
AREA_EXP = 0.1  # preference for larger candidates
TEXT_WEIGHT = 1.0  # weight of the print-density cue
TEXT_DENSITY = 0.04  # edge density at which a region counts as fully 'printed'
LOCAL_WINDOWS = (0.25, 0.5)  # neighbourhood sizes for the local-contrast masks
LOCAL_OFFSET = 12
LOCAL_MIN_WHITE = 120
ZOOM_BELOW = 0.2  # candidates smaller than this get a zoomed second pass
ZOOM_MARGIN = 0.3
ASPECT_PRIOR = True
REFINE_TOP = 6  # candidates that get edge-snapped before the final pick
RING = 0.4  # contrast ring width relative to candidate size (0 = whole photo)
BOX_HYPOTHESIS = 0.9  # weight of the min-area-rectangle alternative (0 = off)
REFINE_REACH = 0.025  # search distance for edge snapping (fraction of short side)


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
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coords, starting at the left-most angle
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def full_frame(w: int, h: int) -> list[list[float]]:
    return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]


def _whiteness(small: np.ndarray) -> np.ndarray:
    """High for bright, unsaturated pixels (paper), low for coloured/dark surfaces and skin."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    return np.clip(v - s, 0, 255).astype(np.uint8)


def _candidate_masks(small: np.ndarray, gray: np.ndarray) -> list[np.ndarray]:
    h, w = gray.shape
    k = max(3, int(round(min(h, w) / 60)) | 1)  # text-filling kernel, scales with image
    kernel = np.ones((5, 5), np.uint8)
    big = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    masks = []

    white = cv2.GaussianBlur(_whiteness(small), (7, 7), 0)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    def clean(m):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, big, iterations=2)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=2)

    for src in (white, blur):
        t, m = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(clean(m))
        # Stricter levels separate white paper from light tables/hands.
        for extra in THRESH_STEPS:
            level = t + extra * (255 - t)
            if level < 250:
                masks.append(clean(np.where(src > level, 255, 0).astype(np.uint8)))

    if LOCAL_WINDOWS:
        # Paper that is whiter than its own surroundings (small receipts on busy/light backgrounds).
        for frac in LOCAL_WINDOWS:
            win = int(min(h, w) * frac) | 1
            local = cv2.blur(white, (win, win))
            m = np.where(white.astype(np.int16) - local > LOCAL_OFFSET, 255, 0).astype(np.uint8)
            m &= np.where(white > LOCAL_MIN_WHITE, 255, 0).astype(np.uint8)
            masks.append(clean(m))

    for src in (blur, white):
        edges = cv2.Canny(src, 20, 60)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, big, iterations=1)
        if CLOSE_BORDER:
            # Regions enclosed by edges and/or the photo border (receipt cut off by the frame).
            framed = edges.copy()
            cv2.rectangle(framed, (0, 0), (w - 1, h - 1), 255, 1)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(255 - framed, connectivity=4)
            for lab in np.argsort(-stats[:, cv2.CC_STAT_AREA])[:4]:
                if stats[lab, cv2.CC_STAT_AREA] < 0.03 * h * w:
                    break
                region = np.where(labels == lab, 255, 0).astype(np.uint8)
                region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, big, iterations=2)
                cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                solid = np.zeros_like(region)
                cv2.drawContours(solid, cs, -1, 255, cv2.FILLED)
                masks.append(solid)
        # Fill closed outlines so that edge contours become solid regions.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(edges)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        masks.append(filled)
    return masks


def _quad_from_contour(c: np.ndarray) -> tuple[np.ndarray, bool]:
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), True
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32), False


def _edge_support(quad: np.ndarray, white: np.ndarray) -> float:
    """Fraction of the outline where the inside is clearly whiter than the outside.

    Sides lying on the photo border (receipt cut off by the frame) count as neutral.
    """
    h, w = white.shape
    d = max(3.0, 0.015 * min(h, w))
    c = quad.mean(axis=0)
    per_side = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        t = np.linspace(0.1, 0.9, 24)[:, None]
        pts = a + (b - a) * t
        n = np.array([b[1] - a[1], a[0] - b[0]], dtype=np.float32)
        n /= np.linalg.norm(n) + 1e-6
        if np.dot(c - (a + b) / 2, n) > 0:
            n = -n  # make n point outwards
        inner = pts - n * d
        outer = pts + n * d
        inside_ok = (outer[:, 0] >= 0) & (outer[:, 0] < w) & (outer[:, 1] >= 0) & (outer[:, 1] < h)
        if inside_ok.mean() < 0.5:
            per_side.append(0.5)  # side on the photo border
            continue
        inner = np.clip(np.round(inner).astype(int), 0, [w - 1, h - 1])
        outer = np.clip(np.round(outer).astype(int), 0, [w - 1, h - 1])
        diff = white[inner[:, 1], inner[:, 0]].astype(int) - white[outer[:, 1], outer[:, 0]].astype(int)
        per_side.append(float(((diff > SUPPORT_DIFF) & inside_ok).sum() / max(inside_ok.sum(), 1)))
    return float(np.mean(per_side)) if min(per_side) > 0.15 else float(np.mean(per_side)) * 0.5


def _text_factor(quad: np.ndarray, edges: np.ndarray) -> float:
    """Receipts carry print: share of edge pixels in the inner part of the quad (saturating)."""
    c = quad.mean(axis=0)
    inner = c + (quad - c) * 0.8
    m = np.zeros(edges.shape, np.uint8)
    cv2.fillPoly(m, [np.round(inner).astype(np.int32)], 1)
    vals = edges[m == 1]
    if vals.size == 0:
        return 0.0
    density = float((vals > 0).mean())
    return max(0.05, min(1.0, density / TEXT_DENSITY))


def _score(quad: np.ndarray, contour: np.ndarray, white: np.ndarray, frame_area: float,
           edges: np.ndarray | None = None) -> float:
    quad_area = cv2.contourArea(quad)
    if quad_area <= 0:
        return 0.0
    frac = quad_area / frame_area
    fill = min(1.0, cv2.contourArea(contour) / quad_area)  # how rectangular the blob is
    score = _edge_support(quad, white) * frac ** AREA_EXP * fill ** 2
    if TEXT_WEIGHT and edges is not None:
        score *= _text_factor(quad, edges) ** TEXT_WEIGHT
    if CONTRAST_WEIGHT:
        m = np.zeros(white.shape, np.uint8)
        cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 1)
        if RING:
            # Compare with the immediate surroundings only (a ring around the candidate).
            c = quad.mean(axis=0)
            ring = np.zeros_like(m)
            cv2.fillPoly(ring, [np.round(c + (quad - c) * (1 + RING)).astype(np.int32)], 1)
            outside_mask = (ring == 1) & (m == 0)
        else:
            outside_mask = m == 0
        inside, outside = white[m == 1], white[outside_mask]
        if inside.size and outside.size > 0.03 * white.size * (0.2 if RING else 1):
            contrast = (float(np.median(inside)) - float(np.median(outside))) / 255
        else:
            contrast = 0.15  # (almost) the whole photo: neutral
        score *= max(contrast, 0.02) ** CONTRAST_WEIGHT
    return float(score)


def _intersect(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, np.array([x2 - x1, y2 - y1], dtype=np.float64))[0]
    return np.array([x1 + vx1 * t, y1 + vy1 * t], dtype=np.float32)


def _refine(quad: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Snap each side to the strongest nearby paper edge and re-intersect the sides."""
    h, w = white.shape
    g = cv2.GaussianBlur(white, (5, 5), 0).astype(np.float32)
    reach = max(4.0, REFINE_REACH * min(h, w))
    offsets = np.linspace(-reach, reach, int(2 * reach) + 1)
    c = quad.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        n = np.array([b[1] - a[1], a[0] - b[0]], dtype=np.float32)
        n /= np.linalg.norm(n) + 1e-6
        if np.dot(c - (a + b) / 2, n) > 0:
            n = -n
        pts = []
        for t in np.linspace(0.08, 0.92, 40):
            p = a + (b - a) * t
            probe = p[None, :] + offsets[:, None] * n[None, :]
            xi = np.round(probe[:, 0]).astype(int)
            yi = np.round(probe[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if ok.sum() < len(offsets) * 0.6:
                continue
            prof = np.where(ok, g[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], np.nan)
            step = prof[:-2] - prof[2:]  # inside minus outside
            if np.all(np.isnan(step)):
                continue
            k = int(np.nanargmax(step))
            if step[k] > SUPPORT_DIFF:
                pts.append(probe[k + 1])
        if len(pts) < 8:
            lines.append(None)
            continue
        lines.append(cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    out = quad.copy()
    for i in range(4):  # corner i is between side i-1 and side i
        l_prev, l_cur = lines[i - 1], lines[i]
        if l_prev is None or l_cur is None:
            continue
        q = _intersect(l_prev, l_cur)
        if q is not None and np.linalg.norm(q - quad[i]) < 3 * reach:
            out[i] = q
    return out


def _hull_of(q: np.ndarray) -> np.ndarray:
    return np.round(q).astype(np.int32).reshape(-1, 1, 2)


def _aspect_factor(q: np.ndarray) -> float:
    """Receipts are elongated: long/short side ratio roughly 1.2 .. 6."""
    if not ASPECT_PRIOR:
        return 1.0
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    r = max(a, b) / max(min(a, b), 1e-6)
    if r < 1.15:
        return 0.6
    if r > 7:
        return 0.6
    return 1.0


FEATURE_NAMES = ("support", "fill", "log_frac", "ring_contrast", "global_contrast", "inside_white",
                 "inside_std", "print_density", "log_aspect", "log_aspect_sq", "exact", "is_box", "border_sides")
# Logistic candidate scorer, learned with tools/train_scoring.py (None = hand-made score).
SCORER_W = (3.3091, 1.4681, 1.2539, 1.2425, -0.0816, 1.9035, -13.5606, -5.2261, 0.5020, -0.3615, -0.1874, 0.7880, -3.2258)
SCORER_B = -0.3161
SCORER_MIN = 0.25  # min. probability for a candidate, else the whole photo is used
SCORER_ALONE_MIN = 0.6  # probability needed when the hand-made score found nothing
HYBRID_BIG = 0.3  # hand-made result at least this large (fraction of photo) is kept


def _border_sides(q: np.ndarray, w: int, h: int) -> int:
    tol = 0.01 * min(w, h) + 1
    n = 0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        if (a[0] < tol and b[0] < tol) or (a[1] < tol and b[1] < tol)                 or (a[0] > w - 1 - tol and b[0] > w - 1 - tol) or (a[1] > h - 1 - tol and b[1] > h - 1 - tol):
            n += 1
    return n


def _features(q: np.ndarray, c: np.ndarray, white: np.ndarray, edges: np.ndarray, exact: bool,
              is_box: bool) -> np.ndarray:
    h, w = white.shape
    qa = max(cv2.contourArea(q), 1.0)
    frac = qa / (w * h)
    fill = min(1.0, cv2.contourArea(c) / qa)
    ctr = q.mean(axis=0)
    inside = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inside, [np.round(q).astype(np.int32)], 1)
    ring = np.zeros_like(inside)
    cv2.fillPoly(ring, [np.round(ctr + (q - ctr) * 1.4).astype(np.int32)], 1)
    core = np.zeros_like(inside)
    cv2.fillPoly(core, [np.round(ctr + (q - ctr) * 0.8).astype(np.int32)], 1)
    win = white[inside == 1]
    wring = white[(ring == 1) & (inside == 0)]
    wout = white[inside == 0]
    in_med = float(np.median(win)) if win.size else 0.0
    ring_med = float(np.median(wring)) if wring.size > 20 else in_med - 30
    out_med = float(np.median(wout)) if wout.size > 0.02 * w * h else in_med - 30
    density = float((edges[core == 1] > 0).mean()) if core.any() else 0.0
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    log_r = float(np.log(max(a, b) / max(min(a, b), 1e-6)))
    return np.array([
        _edge_support(q, white), fill, np.log(frac), (in_med - ring_med) / 255, (in_med - out_med) / 255,
        in_med / 255, float(np.std(win)) / 255 if win.size else 0.0, min(density, 0.3), log_r, log_r ** 2,
        float(exact), float(is_box), _border_sides(q, w, h) / 4,
    ], dtype=np.float64)


def _candidates(small: np.ndarray, gray: np.ndarray, min_area: float):
    """Yield (quad, contour, exact, is_box) for all candidate regions of a downscaled photo."""
    sh, sw = gray.shape
    frame_area = float(sh * sw)
    for mask in _candidate_masks(small, gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:TOP_CONTOURS]:
            if cv2.contourArea(c) < min_area * frame_area:
                break
            quad, exact = _quad_from_contour(c)
            hyps = [(quad, exact, False)]
            if BOX_HYPOTHESIS:
                box = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(c))).astype(np.float32)
                box[:, 0] = np.clip(box[:, 0], 0, sw - 1)
                box[:, 1] = np.clip(box[:, 1], 0, sh - 1)
                hyps.append((box, False, True))
            for q, ex, is_box in hyps:
                if cv2.contourArea(q) > MAX_AREA * frame_area:
                    continue  # the whole photo: no information, that's the fallback anyway
                yield order_corners(q), c, ex, is_box


def _prepare(img: np.ndarray):
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_SIDE / max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return scale, small, gray, _whiteness(small), cv2.Canny(gray, 60, 160)


def _learned_score(f: np.ndarray) -> float:
    z = float(np.dot(SCORER_W, f) + SCORER_B)
    return 1.0 / (1.0 + np.exp(-z))


def _detect_learned(img: np.ndarray, min_area: float) -> tuple[np.ndarray | None, float]:
    """Detection pass that picks the candidate with the highest learned probability."""
    h, w = img.shape[:2]
    scale, small, gray, white, text_edges = _prepare(img)
    best = None
    for q, c, ex, is_box in _candidates(small, gray, min_area):
        sc = _learned_score(_features(q, c, white, text_edges, ex, is_box))
        if best is None or sc > best[0]:
            best = (sc, q)
    if best is None or best[0] < SCORER_MIN:
        return None, 0.0
    quad = best[1]
    if REFINE:
        refined = _refine(quad, white)
        if _edge_support(refined, white) >= _edge_support(quad, white) - 0.02:
            quad = refined
    quad = quad / scale
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad, max(best[0], MIN_SCORE)


def _detect_once(img: np.ndarray, min_area: float) -> tuple[np.ndarray | None, float]:
    """One detection pass. Returns (quad in img coords or None, score)."""
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_SIDE / max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    white = _whiteness(small)
    text_edges = cv2.Canny(gray, 60, 160)
    sh, sw = gray.shape
    frame_area = float(sh * sw)

    cands = []  # (score, quad, contour)
    for mask in _candidate_masks(small, gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:TOP_CONTOURS]:
            if cv2.contourArea(c) < min_area * frame_area:
                break
            quad, exact = _quad_from_contour(c)
            hyps = [(quad, 1.0 if exact else 0.85)]
            if BOX_HYPOTHESIS:
                box = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(c))).astype(np.float32)
                box[:, 0] = np.clip(box[:, 0], 0, sw - 1)
                box[:, 1] = np.clip(box[:, 1], 0, sh - 1)
                hyps.append((box, BOX_HYPOTHESIS))
            for q, weight in hyps:
                if cv2.contourArea(q) > MAX_AREA * frame_area:
                    continue  # the whole photo: no information, that's the fallback anyway
                q = order_corners(q)
                score = _score(q, c, white, frame_area, text_edges) * weight * _aspect_factor(q)
                cands.append((score, q, c, weight))

    if not cands:
        return None, 0.0
    cands.sort(key=lambda t: -t[0])
    best = cands[0][:2]
    if REFINE:
        # Snap the strongest candidates to their edges, then pick again.
        rescored = []
        for score, q, c, weight in cands[:REFINE_TOP]:
            r = _refine(q, white)
            if _edge_support(r, white) < _edge_support(q, white) - 0.02:
                r = q
            r_score = _score(r, _hull_of(r), white, frame_area, text_edges) * weight * _aspect_factor(r)
            # keep the contour-based fill of the original blob
            fill_ratio = min(1.0, cv2.contourArea(c) / max(cv2.contourArea(r), 1.0))
            r_score *= max(fill_ratio, 0.5) ** 2 / 1.0
            rescored.append((max(score, r_score), r if r_score >= score else q))
        best = max(rescored, key=lambda t: t[0])
    score, quad = best
    quad = quad / scale
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad, score


def _zoom(img: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Re-detect inside a crop around a small candidate, at higher resolution."""
    h, w = img.shape[:2]
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    mx, my = (x1 - x0) * ZOOM_MARGIN, (y1 - y0) * ZOOM_MARGIN
    cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    cx1, cy1 = int(min(w, x1 + mx + 1)), int(min(h, y1 + my + 1))
    crop = img[cy0:cy1, cx0:cx1]
    if crop.shape[0] < 40 or crop.shape[1] < 40:
        return None
    zq, score = _detect_once(crop, 0.05)
    if zq is None or score < MIN_SCORE:
        return None
    crop_area = crop.shape[0] * crop.shape[1]
    if cv2.contourArea(zq) > 0.92 * crop_area:
        return None  # only found the crop itself
    return zq + np.array([cx0, cy0], np.float32)


def detect_corners(img: np.ndarray) -> tuple[list[list[float]], bool]:
    """Find the receipt outline. Returns (corners TL,TR,BR,BL in image coords, found)."""
    h, w = img.shape[:2]
    quad, score = _detect_once(img, MIN_AREA)
    if quad is None or score < MIN_SCORE:
        quad = None
    if SCORER_W is not None and (quad is None or cv2.contourArea(quad) < HYBRID_BIG * w * h):
        # Small or no receipt found by the hand-made score: let the learned scorer decide.
        lq, prob = _detect_learned(img, MIN_AREA)
        if lq is not None and (quad is not None or prob >= SCORER_ALONE_MIN):
            quad = lq
    if quad is None:
        return full_frame(w, h), False
    if ZOOM_BELOW and cv2.contourArea(quad) < ZOOM_BELOW * w * h:
        zq = _zoom(img, quad)
        if zq is not None:
            quad = order_corners(zq)
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


def save_png(img: np.ndarray, path) -> None:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img
