"""OCR via Tesseract and heuristics that suggest title, date, amount and category."""
import logging
import re
from datetime import date

import cv2
import numpy as np

from . import amount

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


OCR_WIDTH = 1400  # receipts are scaled to this width before OCR
CONFIDENT = 7.0  # amount evidence margin above which no second OCR pass is needed

# Chosen by measuring real (Wikimedia Commons, SROIE, own photos) and rendered receipts end to end
# on the Pi (tools/ocr_variants.py, tools/ocr_dump.py): background-normalised grey + page-layout
# mode 4 first; if the total is uncertain, a second pass in mode 6 (one uniform block, keeps labels
# and prices of a line together) on the best ink channel with dot-matrix dots fused into strokes,
# and the evidence of both passes is merged.


def _scaled_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    f = OCR_WIDTH / w
    return cv2.resize(gray, (OCR_WIDTH, max(1, int(h * f))), interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)


def normalized(img: np.ndarray) -> np.ndarray:
    """Divide by a blurred background estimate: removes shading and tinted paper, keeps print."""
    g = _scaled_gray(img)
    bg = cv2.medianBlur(cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))), 31)
    return cv2.normalize(cv2.divide(g, bg, scale=255), None, 0, 255, cv2.NORM_MINMAX)


def ink_channel(img: np.ndarray) -> np.ndarray:
    """The colour channel in which print and paper separate best (Otsu between-class variance).
    Violet dot-matrix ink, for example, has far more contrast in the green channel than in grey."""
    if img.ndim == 2:
        return img
    b, g, r = cv2.split(img)
    best, best_score = None, -1.0
    levels = np.arange(256)
    for ch in (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), b, g, r, cv2.min(cv2.min(b, g), r)):
        p = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel()
        p /= p.sum()
        w0, mu = np.cumsum(p), np.cumsum(p * levels)
        var_t = float(np.sum(p * (levels - mu[-1]) ** 2)) + 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
        score = float(np.nanmax(between)) / var_t
        if score > best_score:
            best, best_score = ch, score
    return best


def binarized(img: np.ndarray) -> np.ndarray:
    """Second pass: best ink channel, needle dots fused into strokes, Otsu."""
    g = _scaled_gray(ink_channel(img))
    g = cv2.erode(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # grow dark dots
    g = cv2.GaussianBlur(cv2.GaussianBlur(g, (3, 3), 0), (3, 3), 0)  # (as measured: two light blurs)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def run_ocr(img: np.ndarray, psm: int = 4, prepared: np.ndarray | None = None) -> str:
    """Return recognised text of a cropped (not contrast-enhanced) receipt, or "" on failure."""
    if pytesseract is None:
        return ""
    try:
        page = prepared if prepared is not None else normalized(img)
        return pytesseract.image_to_string(page, lang=_lang(), config=f"--psm {psm}", timeout=90)
    except Exception as e:  # tesseract missing, timeout, ...
        log.warning("OCR failed: %s", e)
        return ""


def analyze(img: np.ndarray) -> tuple[str, dict]:
    """OCR a cropped receipt and suggest title/date/amount/category."""
    if pytesseract is None:
        return "", suggestions("") | {"amount_alternatives": []}
    text = run_ocr(img, 4, normalized(img))
    texts = [text]
    if amount_confidence(text) < CONFIDENT:
        second = run_ocr(img, 6, binarized(img))
        if second:
            texts.append(second)
    sugg = suggestions(text)
    sugg["amount"] = suggest_amount(texts)
    sugg["amount_alternatives"] = amount_alternatives(texts)
    return text, sugg


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


# ---------------------------------------------------------------- amount (see app/amount.py)


def suggest_amount(text: str | list[str]) -> float | None:
    return amount.suggest(text)


def amount_alternatives(text: str | list[str]) -> list[float]:
    return amount.alternatives(text)


def amount_confidence(text: str | list[str]) -> float:
    return amount.confidence(text)


def amount_candidates(text: str | list[str]) -> dict[float, float]:
    return dict(amount.ranked(text))


_parse_amounts = amount.parse_amounts


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
