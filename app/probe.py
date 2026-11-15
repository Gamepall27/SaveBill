"""Trial run: Tesseract on the phone (tesseract.js) against Tesseract on the Pi.

Temporary (plan step 2, docs/PLAN-berechnung-auf-dem-geraet.md). The probe page
/ocr-probe.html fetches, for every stored receipt, exactly the images the Pi OCR uses
(un-enhanced crop -> normalized / binarized), recognises them in the browser with the
Pi's own language files, and reports times and agreement. Only numbers are logged.
"""
import difflib
import json
import logging
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from . import amount, config, db, ocr, scanner

log = logging.getLogger("savebill")
router = APIRouter(prefix="/api/probe")
TESSDATA = Path("/usr/share/tesseract-ocr/5/tessdata")
PASSES = {1: (4, ocr.normalized), 2: (6, ocr.binarized)}  # as in ocr.analyze
_pi_cache: dict[tuple[str, int], tuple[str, int]] = {}


def _crop(rec_id: str):
    rec = db.get(rec_id)
    original = config.ORIGINALS_DIR / f"{rec_id}.jpg"
    if not rec or not rec.get("corners") or not original.exists():
        raise HTTPException(404, "Nicht gefunden")
    return rec, scanner.warp(scanner.read_image(original), rec["corners"])


@router.get("/receipts")
def receipts():
    return [r["id"] for r in db.search() if (config.ORIGINALS_DIR / f"{r['id']}.jpg").exists()]


@router.get("/tessdata/{name}")
def tessdata(name: str):
    if name not in ("deu.traineddata", "eng.traineddata"):
        raise HTTPException(404, "Nicht gefunden")
    return FileResponse(TESSDATA / name, headers={"Cache-Control": "max-age=86400"})


@router.get("/{rec_id}/pass{n}.png")
def prepared(rec_id: str, n: int):
    if n not in PASSES:
        raise HTTPException(404, "Nicht gefunden")
    _, crop = _crop(rec_id)
    ok, buf = cv2.imencode(".png", PASSES[n][1](crop))
    return Response(buf.tobytes(), media_type="image/png")


def _pi(rec_id: str, n: int, crop) -> tuple[str, int]:
    if (rec_id, n) not in _pi_cache:
        psm, prep = PASSES[n]
        page = prep(crop)
        t = time.perf_counter()
        text = ocr.run_ocr(crop, psm, page)
        _pi_cache[(rec_id, n)] = (text, round((time.perf_counter() - t) * 1000))
    return _pi_cache[(rec_id, n)]


class PhoneResult(BaseModel):
    index: int
    texts: dict[int, str]  # pass -> text recognised on the phone
    ms: dict[int, int]  # pass -> recognition time on the phone
    load_ms: int | None = None
    ua: str | None = None


@router.post("/{rec_id}/compare")
def compare(rec_id: str, body: PhoneResult):
    rec, crop = _crop(rec_id)
    out = {"index": body.index, "passes": {}}
    pi_texts, phone_texts = [], []
    for n in sorted(body.texts):
        if n not in PASSES:
            continue
        pi_text, pi_ms = _pi(rec_id, n, crop)
        phone = body.texts[n]
        pi_texts.append(pi_text)
        phone_texts.append(phone)
        out["passes"][n] = {"phone_ms": body.ms.get(n), "pi_ms": pi_ms,
                            "text_similarity": round(difflib.SequenceMatcher(None, pi_text, phone).ratio(), 3)}
    saved = None if rec.get("amount") is None else round(rec["amount"] * 100)
    for name, texts in (("pi", pi_texts), ("phone", phone_texts)):
        a = amount.suggest(texts) if texts else None
        out[f"{name}_amount_ok"] = None if saved is None else (a is not None and round(a * 100) == saved)
    out["same_amount"] = (amount.suggest(pi_texts) if pi_texts else None) == \
        (amount.suggest(phone_texts) if phone_texts else None)
    out["load_ms"] = body.load_ms
    log.info("ocr-probe %s", json.dumps(out | {"ua": (body.ua or "")[:160]}, separators=(",", ":")))
    return out
