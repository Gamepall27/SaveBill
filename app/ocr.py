"""OCR via Tesseract and heuristics that suggest title, date, amount and category."""
import logging
import re
from datetime import date

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

_lang_cache: str | None = None


def _lang() -> str:
    global _lang_cache
    if _lang_cache is None:
        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception:
            available = set()
        langs = [l for l in ("deu", "eng") if l in available]
        _lang_cache = "+".join(langs) or "eng"
    return _lang_cache


def run_ocr(img: np.ndarray) -> str:
    """Return recognised text, or "" if Tesseract is unavailable or fails."""
    if pytesseract is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1000:
        f = 1000 / w
        gray = cv2.resize(gray, (1000, int(h * f)), interpolation=cv2.INTER_CUBIC)
    elif w > 1800:
        f = 1800 / w
        gray = cv2.resize(gray, (1800, int(h * f)), interpolation=cv2.INTER_AREA)
    try:
        return pytesseract.image_to_string(gray, lang=_lang(), config="--psm 4", timeout=90)
    except Exception as e:  # tesseract missing, timeout, ...
        log.warning("OCR failed: %s", e)
        return ""


# ---------------------------------------------------------------- date

_DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})\s?[./-]\s?(\d{1,2})\s?[./-]\s?(\d{4}|\d{2})(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")


def suggest_date(text: str, today: date | None = None) -> str | None:
    today = today or date.today()
    candidates = []
    for m in _DATE_ISO.finditer(text):
        candidates.append((m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _DATE_DMY.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(y) + 2000 if len(y) == 2 else int(y)
        candidates.append((m.start(), y, mo, d))
    for _, y, mo, d in sorted(candidates):
        try:
            dt = date(y, mo, d)
        except ValueError:
            continue
        if date(2000, 1, 1) <= dt <= today.replace(year=today.year + 1):
            return dt.isoformat()
    return None


# ---------------------------------------------------------------- amount

_AMOUNT = re.compile(r"(?<![\d,.])-?(\d{1,3}(?:[.\s]\d{3})+|\d+)\s?[,.]\s?(\d{2})(?!\d)")
_TOTAL_KW = re.compile(r"\b(summe|gesamt\w*|total|zu\s*zahlen|zahlbetrag|endbetrag|end\s*summe|betrag)\b", re.I)
_PAY_KW = re.compile(r"\b(bar|ec|karte|kartenzahlung|girocard|visa|mastercard|maestro)\b", re.I)
_EXCLUDE_KW = re.compile(r"(mwst|ust|steuer|netto|brutto\s*\d|r[uü]e?ckgeld|zur[uü]ck|gegeben|geg\.|rabatt|pfand|%)", re.I)


def _parse_amounts(line: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(line):
        whole = re.sub(r"[.\s]", "", m.group(1))
        try:
            out.append(float(f"{whole}.{m.group(2)}"))
        except ValueError:
            pass
    return out


def suggest_amount(text: str) -> float | None:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for kw in (_TOTAL_KW, _PAY_KW):
        for line in lines:
            if kw.search(line) and not _EXCLUDE_KW.search(line):
                amounts = [a for a in _parse_amounts(line) if 0 < a < 100000]
                if amounts:
                    return amounts[-1]
    pool = [a for l in lines if not _EXCLUDE_KW.search(l) for a in _parse_amounts(l) if 0 < a < 100000]
    return max(pool) if pool else None


# ---------------------------------------------------------------- title

_TITLE_SKIP = re.compile(
    r"(str\.|stra(ss|ß)e|\bweg\b|platz|\btel\b|telefon|fax|www\.|http|@|\bust|steuer|\bplz\b|\d{5}"
    r"|datum|uhrzeit|kasse|\bbon\b|beleg|filiale|quittung|rechnung|kunden?beleg|willkommen|danke)",
    re.I,
)


def suggest_title(text: str) -> str | None:
    for raw in [l.strip() for l in text.splitlines() if l.strip()][:8]:
        line = re.sub(r"[^\w&+'.\- ]", " ", raw)
        line = re.sub(r"\s+", " ", line).strip(" .-")
        letters = sum(c.isalpha() for c in line)
        if letters < 3 or letters / max(len(line), 1) < 0.6:
            continue
        if _TITLE_SKIP.search(line):
            continue
        # Drop legal-form suffixes for a nicer title.
        line = re.sub(r"\b(gmbh|ohg|kg|ag|e\.?\s?k\.?|co\.?|& co\.?)\b.*$", "", line, flags=re.I).strip(" .-&")
        if len(line) < 2:
            continue
        if line.isupper() and len(line) > 4:
            line = line.title()
        return line[:60]
    return None


# ---------------------------------------------------------------- category

_CATEGORIES = [
    ("Lebensmittel", r"rewe|edeka|aldi|lidl|netto|penny|kaufland|norma|tegut|globus|real\b|marktkauf|supermarkt|b[äa]ckerei"),
    ("Drogerie", r"\bdm\b|drogerie|rossmann|m[üu]ller"),
    ("Tanken", r"aral|shell|esso|\bjet\b|tankstelle|\bomv\b|agip|avia|diesel|super e10|benzin"),
    ("Baumarkt", r"\bobi\b|bauhaus|hornbach|toom|hagebau|baumarkt|globus baumarkt"),
    ("Restaurant", r"restaurant|pizzeria|gastst[äa]tte|bistro|caf[ée]|imbiss|mcdonald|burger king|kfc|subway|trinkgeld|bewirtung"),
    ("Elektronik", r"media\s?markt|saturn|conrad|cyberport|notebooksbilliger|expert"),
    ("Kleidung", r"\bh&m\b|zara|primark|c&a|deichmann|kik\b|tk maxx"),
    ("Apotheke", r"apotheke|pharma"),
    ("Möbel", r"ikea|m[öo]bel|xxxlutz|poco|roller"),
]


def suggest_category(text: str, known: list[str] | None = None) -> str | None:
    low = text.lower()
    for name, pattern in _CATEGORIES:
        if re.search(pattern, low):
            return name
    return None


def suggestions(text: str) -> dict:
    return {
        "title": suggest_title(text),
        "date": suggest_date(text),
        "amount": suggest_amount(text),
        "category": suggest_category(text),
    }
