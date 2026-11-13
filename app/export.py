"""Export helpers: download file names, PDF and ZIP."""
import csv
import io
import re
import unicodedata
import zipfile

import img2pdf

from . import config

A4 = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
_LAYOUT = img2pdf.get_layout_fun(A4, border=(img2pdf.mm_to_pt(10), img2pdf.mm_to_pt(10)),
                                 fit=img2pdf.FitMode.into)


def slug(text: str, max_len: int = 40) -> str:
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = text.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text[:max_len] or "Quittung"


def format_amount(amount) -> str:
    return "" if amount is None else f"{amount:.2f}".replace(".", ",")


def base_name(rec: dict) -> str:
    parts = [rec["date"], slug(rec["title"])]
    if rec.get("amount") is not None:
        parts.append(format_amount(rec["amount"]).replace(",", "-"))
    return "_".join(parts)


def image_path(rec_id: str):
    return config.IMAGES_DIR / f"{rec_id}.jpg"


def pdf_bytes(records: list[dict]) -> bytes:
    paths = [str(image_path(r["id"])) for r in records if image_path(r["id"]).exists()]
    if not paths:
        raise ValueError("Keine Bilder vorhanden")
    return img2pdf.convert(paths, layout_fun=_LAYOUT)


def zip_bytes(records: list[dict]) -> bytes:
    buf = io.BytesIO()
    used: set[str] = set()
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, delimiter=";")
    writer.writerow(["Datum", "Titel", "Betrag", "Kategorie", "Notiz", "Datei"])
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rec in records:
            p = image_path(rec["id"])
            if not p.exists():
                continue
            name = base_name(rec)
            candidate, n = f"{name}.jpg", 2
            while candidate in used:
                candidate, n = f"{name}_{n}.jpg", n + 1
            used.add(candidate)
            zf.write(p, candidate)
            writer.writerow([rec["date"], rec["title"], format_amount(rec.get("amount")),
                             rec.get("category") or "", rec.get("note") or "", candidate])
        # utf-8-sig so Excel opens umlauts correctly
        zf.writestr("quittungen.csv", csv_buf.getvalue().encode("utf-8-sig"))
    return buf.getvalue()
