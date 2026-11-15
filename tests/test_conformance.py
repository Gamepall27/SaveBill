"""The reference data in tests/conformance must match what app/ computes.

These files are the yardstick for ports (Kotlin/Swift, see tests/conformance/README.md).
If a change to app/ is intended, regenerate them: python tools/export_conformance.py --cord <dir>
"""
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import amount, ocr, scanner

DIR = Path(__file__).parent / "conformance"
CASES = json.loads((DIR / "suggest_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_suggestions_match_reference(case):
    texts, exp = case["texts"], case["expected"]
    today = date.fromisoformat(case["today"])
    assert ocr.suggest_title(texts[0]) == exp["title"]
    assert ocr.suggest_date(texts[0], today=today) == exp["date"]
    assert ocr.suggest_category(texts[0]) == exp["category"]
    assert amount.suggest(texts) == exp["amount"]
    assert amount.alternatives(texts) == exp["alternatives"]
    assert round(amount.confidence(texts), 6) == exp["confidence"]
    assert [[a, round(s, 6)] for a, s in amount.ranked(texts)[:8]] == exp["ranked"]
    lines = [{k: r[k] for k in ("raw", "amounts", "kind", "orphan", "item", "tabular", "borrowed")}
             for r in amount._lines(texts[0])]
    assert lines == exp["lines"]


@pytest.mark.parametrize("name", sorted(p.name[:-10] for p in (DIR / "segmentation").glob("*_input.png")))
def test_network_output_matches_reference(name):
    canvas = cv2.imread(str(DIR / "segmentation" / f"{name}_input.png"))
    net = scanner._seg_net()
    net.setInput(canvas.transpose(2, 0, 1)[None].astype(np.float32))
    prob = net.forward()[0, 0]
    ref = cv2.imread(str(DIR / "segmentation" / f"{name}_prob.png"), cv2.IMREAD_UNCHANGED) / 65535.0
    assert np.abs(prob - ref).max() < 1e-3


EXPECTED = json.loads((DIR / "detection_synth" / "expected.json").read_text())


@pytest.mark.parametrize("item", EXPECTED, ids=[e["file"] for e in EXPECTED])
def test_detection_matches_reference(item):
    img = scanner.load_image((DIR / "detection_synth" / item["file"]).read_bytes())
    corners, found = scanner.detect_corners(img)
    assert found == item["found"]
    assert np.abs(np.array(corners) - np.array(item["corners"])).max() < 1.0
