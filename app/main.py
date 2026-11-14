import json
import logging
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date as date_cls
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import config, db, export, ocr, scanner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("savebill")


_ID = re.compile(r"^[0-9a-f]{32}$")


@asynccontextmanager
async def _lifespan(_app):
    db.init()
    _cleanup_scans()
    yield


app = FastAPI(title="SaveBill", lifespan=_lifespan)


# ------------------------------------------------------------------ helpers

def _check_id(value: str) -> str:
    if not _ID.match(value):
        raise HTTPException(404, "Nicht gefunden")
    return value


def _scan_paths(scan_id: str) -> dict:
    d = config.SCANS_DIR
    return {
        "image": d / f"{scan_id}.png",  # lossless: re-encoding can flip OCR on faint print
        "crop": d / f"{scan_id}_crop.jpg",
        "meta": d / f"{scan_id}.json",
    }


def _cleanup_scans() -> None:
    cutoff = time.time() - config.SCAN_TTL_SECONDS
    for p in config.SCANS_DIR.glob("*"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _no_cache(path, **kw) -> FileResponse:
    if not path.exists():
        raise HTTPException(404, "Nicht gefunden")
    return FileResponse(path, headers={"Cache-Control": "no-cache"}, **kw)


def _attachment(filename: str) -> dict:
    return {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}


Corners = list[list[float]]


class CornersIn(BaseModel):
    corners: Corners

    @field_validator("corners")
    @classmethod
    def _four(cls, v):
        if len(v) != 4 or any(len(p) != 2 for p in v):
            raise ValueError("4 Eckpunkte erwartet")
        return v


def _valid_date(v: str) -> str:
    date_cls.fromisoformat(v)
    return v


class ReceiptIn(BaseModel):
    scan_id: str
    title: str = Field(min_length=1, max_length=200)
    date: str
    amount: float | None = None
    category: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=5000)

    @field_validator("date")
    @classmethod
    def _date(cls, v):
        return _valid_date(v)


class ReceiptPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    date: str | None = None
    amount: float | None = None
    category: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=5000)

    @field_validator("date")
    @classmethod
    def _date(cls, v):
        return None if v is None else _valid_date(v)


def _with_urls(rec: dict) -> dict:
    rid = rec["id"]
    v = rec["updated_at"].replace(":", "").replace("-", "").replace("+", "")
    rec = {k: val for k, val in rec.items() if k != "ocr_text"}
    rec["image_url"] = f"/api/receipts/{rid}/image?v={v}"
    rec["thumb_url"] = f"/api/receipts/{rid}/thumb?v={v}"
    rec["filename"] = export.base_name(rec)
    return rec


# ------------------------------------------------------------------ scans

@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/scans")
def create_scan(file: UploadFile = File(...)):
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Datei zu groß")
    try:
        img = scanner.load_image(data)
    except Exception:
        raise HTTPException(400, "Bild konnte nicht gelesen werden")
    _cleanup_scans()
    scan_id = uuid.uuid4().hex
    paths = _scan_paths(scan_id)
    scanner.save_png(img, paths["image"])
    corners, found = scanner.detect_corners(img)
    h, w = img.shape[:2]
    return {
        "scan_id": scan_id,
        "width": w,
        "height": h,
        "corners": corners,
        "detected": found,
        "image_url": f"/api/scans/{scan_id}/image",
    }


@app.get("/api/scans/{scan_id}/image")
def scan_image(scan_id: str):
    return _no_cache(_scan_paths(_check_id(scan_id))["image"])


@app.get("/api/scans/{scan_id}/crop")
def scan_crop(scan_id: str):
    return _no_cache(_scan_paths(_check_id(scan_id))["crop"])


@app.post("/api/scans/{scan_id}/crop")
def crop_scan(scan_id: str, body: CornersIn):
    paths = _scan_paths(_check_id(scan_id))
    if not paths["image"].exists():
        raise HTTPException(404, "Scan abgelaufen – bitte neu aufnehmen")
    img = scanner.read_image(paths["image"])
    warped = scanner.warp(img, body.corners)
    scanner.save_jpeg(scanner.enhance(warped), paths["crop"])
    text, suggestions = ocr.analyze(warped)  # OCR works best on the un-enhanced crop
    paths["meta"].write_text(json.dumps({"corners": body.corners, "ocr_text": text}), encoding="utf-8")
    return {
        "crop_url": f"/api/scans/{scan_id}/crop?t={int(time.time() * 1000)}",
        "suggestions": suggestions,
        "ocr_text": text,
    }


# ------------------------------------------------------------------ receipts

@app.post("/api/receipts")
def create_receipt(body: ReceiptIn):
    scan_id = _check_id(body.scan_id)
    paths = _scan_paths(scan_id)
    if not paths["crop"].exists():
        raise HTTPException(400, "Bitte zuerst zuschneiden")
    meta = json.loads(paths["meta"].read_text(encoding="utf-8")) if paths["meta"].exists() else {}

    rec_id = uuid.uuid4().hex
    image = config.IMAGES_DIR / f"{rec_id}.jpg"
    shutil.move(paths["crop"], image)
    scanner.save_jpeg(scanner.read_image(paths["image"]), config.ORIGINALS_DIR / f"{rec_id}.jpg", quality=92)
    paths["image"].unlink(missing_ok=True)
    scanner.save_jpeg(scanner.thumbnail(scanner.read_image(image)), config.THUMBS_DIR / f"{rec_id}.jpg", 80)
    paths["meta"].unlink(missing_ok=True)

    rec = db.insert(
        rec_id,
        title=body.title.strip(),
        date=body.date,
        amount=body.amount,
        category=(body.category or "").strip() or None,
        note=(body.note or "").strip() or None,
        ocr_text=meta.get("ocr_text"),
        corners=meta.get("corners"),
    )
    return _with_urls(rec)


def _filtered(q, date_from, date_to, category):
    return db.search(q=q or None, date_from=date_from or None, date_to=date_to or None,
                     category=category or None)


@app.get("/api/receipts")
def list_receipts(q: str | None = None, date_from: str | None = Query(None, alias="from"),
                  date_to: str | None = Query(None, alias="to"), category: str | None = None):
    items = _filtered(q, date_from, date_to, category)
    total = sum(r["amount"] for r in items if r["amount"] is not None)
    return {"items": [_with_urls(r) for r in items], "count": len(items), "total": round(total, 2)}


def _get_or_404(rec_id: str) -> dict:
    rec = db.get(_check_id(rec_id))
    if not rec:
        raise HTTPException(404, "Nicht gefunden")
    return rec


@app.get("/api/receipts/{rec_id}")
def get_receipt(rec_id: str):
    rec = _get_or_404(rec_id)
    out = _with_urls(rec)
    out["ocr_text"] = rec.get("ocr_text")
    out["original_url"] = f"/api/receipts/{rec_id}/original"
    return out


@app.patch("/api/receipts/{rec_id}")
def patch_receipt(rec_id: str, body: ReceiptPatch):
    _get_or_404(rec_id)
    changes = body.model_dump(exclude_unset=True)
    for k in ("title", "category", "note"):
        if isinstance(changes.get(k), str):
            changes[k] = changes[k].strip()
    return _with_urls(db.update(rec_id, changes))


@app.delete("/api/receipts/{rec_id}")
def delete_receipt(rec_id: str):
    _get_or_404(rec_id)
    db.delete(rec_id)
    for d in (config.IMAGES_DIR, config.THUMBS_DIR, config.ORIGINALS_DIR):
        (d / f"{rec_id}.jpg").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/receipts/{rec_id}/recrop")
def recrop_receipt(rec_id: str, body: CornersIn):
    _get_or_404(rec_id)
    original = config.ORIGINALS_DIR / f"{rec_id}.jpg"
    if not original.exists():
        raise HTTPException(404, "Originalbild fehlt")
    cropped = scanner.crop_receipt(scanner.read_image(original), body.corners)
    scanner.save_jpeg(cropped, config.IMAGES_DIR / f"{rec_id}.jpg")
    scanner.save_jpeg(scanner.thumbnail(cropped), config.THUMBS_DIR / f"{rec_id}.jpg", 80)
    return _with_urls(db.update(rec_id, {"corners": body.corners}))


@app.get("/api/receipts/{rec_id}/original")
def receipt_original(rec_id: str):
    return _no_cache(config.ORIGINALS_DIR / f"{_check_id(rec_id)}.jpg")


@app.get("/api/receipts/{rec_id}/thumb")
def receipt_thumb(rec_id: str):
    p = config.THUMBS_DIR / f"{_check_id(rec_id)}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/receipts/{rec_id}/image")
def receipt_image(rec_id: str, download: bool = False):
    rec = _get_or_404(rec_id)
    p = config.IMAGES_DIR / f"{rec_id}.jpg"
    if not p.exists():
        raise HTTPException(404)
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    if download:
        headers.update(_attachment(export.base_name(rec) + ".jpg"))
    return FileResponse(p, media_type="image/jpeg", headers=headers)


@app.get("/api/receipts/{rec_id}/pdf")
def receipt_pdf(rec_id: str):
    rec = _get_or_404(rec_id)
    return Response(export.pdf_bytes([rec]), media_type="application/pdf",
                    headers=_attachment(export.base_name(rec) + ".pdf"))


@app.get("/api/categories")
def list_categories():
    return db.categories()


# ------------------------------------------------------------------ export

def _export_name(date_from, date_to, ext):
    span = "_".join(x for x in (date_from, date_to) if x) or date_cls.today().isoformat()
    return f"Quittungen_{span}.{ext}"


@app.get("/api/export.zip")
def export_zip(q: str | None = None, date_from: str | None = Query(None, alias="from"),
               date_to: str | None = Query(None, alias="to"), category: str | None = None):
    items = _filtered(q, date_from, date_to, category)
    if not items:
        raise HTTPException(404, "Keine Quittungen für diesen Filter")
    return Response(export.zip_bytes(items), media_type="application/zip",
                    headers=_attachment(_export_name(date_from, date_to, "zip")))


@app.get("/api/export.pdf")
def export_pdf(q: str | None = None, date_from: str | None = Query(None, alias="from"),
               date_to: str | None = Query(None, alias="to"), category: str | None = None):
    items = _filtered(q, date_from, date_to, category)
    if not items:
        raise HTTPException(404, "Keine Quittungen für diesen Filter")
    return Response(export.pdf_bytes(items), media_type="application/pdf",
                    headers=_attachment(_export_name(date_from, date_to, "pdf")))


@app.exception_handler(ValueError)
def _value_error(_, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
