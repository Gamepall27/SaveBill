import cv2
import numpy as np
import pytest

from app import scanner
from tests.synth import photo

CASES = [
    # straight
    [[300, 200], [900, 200], [900, 1400], [300, 1400]],
    # rotated / perspective
    [[350, 150], [950, 260], [820, 1450], [180, 1330]],
    # strong perspective
    [[420, 300], [800, 280], [1000, 1500], [200, 1480]],
    # small receipts (~5-10 % of the photo)
    [[700, 900], [930, 940], [880, 1380], [650, 1340]],
    [[150, 200], [380, 180], [400, 620], [170, 640]],
]


@pytest.mark.parametrize("truth", CASES)
def test_detect_corners_finds_receipt(truth):
    img = photo(truth)
    corners, found = scanner.detect_corners(img)
    assert found
    diag = np.hypot(*img.shape[:2])
    err = np.linalg.norm(np.array(corners) - scanner.order_corners(truth), axis=1)
    assert err.max() < 0.03 * diag, (corners, truth)


def test_detect_corners_fallback_on_blank_image():
    img = np.full((600, 400, 3), 128, np.uint8)
    corners, found = scanner.detect_corners(img)
    assert not found
    assert corners == scanner.full_frame(400, 600)


def test_order_corners_any_input_order():
    pts = [[10, 100], [10, 10], [100, 100], [100, 10]]
    ordered = scanner.order_corners(pts).tolist()
    assert ordered == [[10, 10], [100, 10], [100, 100], [10, 100]]


def test_warp_produces_upright_portrait():
    truth = CASES[1]
    img = photo(truth)
    out = scanner.crop_receipt(img, truth)
    h, w = out.shape[:2]
    assert h > w * 1.5
    # Cropped receipt is mostly bright paper.
    assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean() > 180


def test_load_image_applies_exif_and_downscale(tmp_path):
    from PIL import Image
    im = Image.new("RGB", (4000, 2000), "white")
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° CW on display
    import io
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    arr = scanner.load_image(buf.getvalue(), max_side=1000)
    h, w = arr.shape[:2]
    assert (w, h) == (500, 1000)
