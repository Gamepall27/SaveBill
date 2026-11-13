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


OCR_WIDTH = 1400  # receipts are scaled to this width before OCR
CONFIDENT = 6.0  # amount evidence margin above which no second OCR pass is needed

# Chosen by measuring real (Wikimedia Commons) and rendered German receipts end to end on the
# Pi (tools/ocr_variants.py, tools/ocr_dump.py): background-normalised grey + page-layout mode 4
# first; if the total is uncertain, an Otsu-binarised pass in mode 6 (one uniform block, keeps
# labels and prices of a line together) and the evidence of both passes is merged.


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


def binarized(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(_scaled_gray(img), (3, 3), 0)
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
        return "", suggestions("")
    text = run_ocr(img, 4, normalized(img))
    texts = [text]
    if amount_confidence(text) < CONFIDENT:
        second = run_ocr(img, 6, binarized(img))
        if second:
            texts.append(second)
    sugg = suggestions(text)
    sugg["amount"] = suggest_amount(texts)
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


# ---------------------------------------------------------------- amount
#
# German receipts list every article with its price, then the total ("zu zahlen", "Summe",
# ...), the payment (Bar / EC / Karte) with change, and a VAT table that repeats amounts.
# Each line is classified and every hint for the total gets a weight; amounts confirmed by
# independent hints (e.g. "Summe 12,50" and "EC-Karte 12,50", or "Bar 20,00" minus
# "Rückgeld 7,50") win.

# 1.234,56 | 12,34 | 12.34 | 12, 34 (OCR) | 12,34- ; not part of a longer number
_AMOUNT = re.compile(r"(?<![\d,.])(-(?=\d))?(\d{1,3}(?:\.\d{3})+|\d{1,5})\s?[,.]\s?(\d)\s?(\d)(?![\d,.]\d)")
_DATE_TIME = re.compile(r"\b\d{1,2}\s?[./]\s?\d{1,2}\s?[./]\s?\d{2,4}\b|\b\d{1,2}\s?:\s?\d{2}(?:\s?:\s?\d{2})?\b"
                        r"|\b\d{4}-\d{2}-\d{2}\S*")

_KW_FINAL = re.compile(r"zu\s*zahlen|zu\s*zah1en|zahlbetr|gesamt\s*-?betr|endbetrag|endsumme|rechnungsbetrag|gesamtsumme"
                       r"|summe\s*eur|total\s*eur|bon-?summe|zu\s*bezahlen|betrag\s*eur", re.I)
_KW_SUM = re.compile(r"\b(summe|total|gesamt|zahlbetrag)\b", re.I)
_KW_SUBTOTAL = re.compile(r"zw\.?\s*-?\s*summe|zwischen\s*summe|sub\s*total", re.I)
_KW_CARD = re.compile(r"\b(ec|karte|kartenzahlung|kreditkarte|girocard|visa|mastercard|maestro|v\s?pay|amex|"
                      r"debit|kontaktlos|apple\s?pay|google\s?pay|paypal|ec-cash|electronic\s*cash)\b", re.I)
_KW_CASH = re.compile(r"\b(bar|bargeld|gegeben|geg\.?|barzahlung)\b", re.I)
# OCR often mangles the umlaut in "Rückgeld" ("R?ckgeld", "Ruckgeld", "Rueckgeld").
_KW_CHANGE = re.compile(r"r\S{0,2}c?kgeld|ckgeld|r[uü]ckgabe|zur[uü]ck|wechselgeld", re.I)
_KW_TAX = re.compile(r"mwst|mw\.?\s?st|\bust\b|steuer|netto|brutto|umsatz|%|\bsatz\b|enth\.", re.I)
_KW_IGNORE = re.compile(r"rabatt|preisvorteil|ersparnis|gespart|sparen|punkte|payback|deutschlandcard|bonus"
                        r"|pfand\s*r[uü]ck|leergut|kg\s*x|/\s?kg|eur\s*/|\d\s?x\s|\bx\s?\d|st[uü]ck\s*x"
                        r"|trinkgeld|\btip\b|kilometer|liter\s*x|/\s?l\b|tse|signatur|seriennr|transaktion", re.I)


_OCR_DIGIT_FIXES = [
    (re.compile(r"(?<![\w$])\$(?=\d)"), "5"),          # "$1,26" -> "51,26"
    (re.compile(r"(?<=\d)[oO](?=[\d,.])"), "0"),        # "1o,50" -> "10,50"
    (re.compile(r"(?<=[\s(])[oO](?=[,.]\d\d)"), "0"),   # " o,99" -> " 0,99"
    (re.compile(r"(?<=[,.])[oO](?=\d)|(?<=[,.]\d)[oO]\b"), "0"),  # "4,o5", "4,5o"
    (re.compile(r"(?<=\s)[lI|](?=\d[,.]\d\d)"), "1"),  # "l2,50" -> "12,50"
]


def _norm_line(line: str) -> str:
    line = line.replace("€", " ").replace("€", " ")
    for pattern, repl in _OCR_DIGIT_FIXES:
        line = pattern.sub(repl, line)
    return _DATE_TIME.sub(" ", line)


def _parse_amounts(line: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(_norm_line(line)):
        whole = m.group(2).replace(".", "")
        try:
            v = float(f"{whole}.{m.group(3)}{m.group(4)}")
        except ValueError:
            continue
        out.append(-v if m.group(1) else v)
    return out


def _summe_like(line: str) -> bool:
    """Fuzzy 'Summe' for OCR errors like SUMNE, SUNHE, Suane, SUMM."""
    for tok in re.findall(r"[A-Za-zÄÖÜäöü]{4,6}", line):
        t = tok.lower()
        if t[0] == "s" and t[1] in "uü" and (t.endswith("e") or t.startswith("summ")) and len(set(t) & set("mnhra")) >= 1:
            return True
    return False


def amount_candidates(text: str) -> dict[float, float]:
    """Map amount -> weight of evidence that it is the receipt total."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    votes: dict[float, float] = {}
    sources: dict[float, set] = {}

    def vote(amount, weight, kind):
        if amount is None or not (0 < amount < 100000):
            return
        a = round(amount, 2)
        votes[a] = votes.get(a, 0.0) + weight
        sources.setdefault(a, set()).add(kind)

    tax_block_until = -1
    cash, change = [], []
    n = len(lines)
    for i, line in enumerate(lines):
        amounts = _parse_amounts(line)
        low = line.lower()
        is_tax_header = bool(_KW_TAX.search(line)) and len(amounts) == 0
        if is_tax_header:
            tax_block_until = i + 4  # rows below a VAT header belong to the table
        in_tax = i <= tax_block_until
        tabular = len(amounts) >= 3 or (len(amounts) >= 2 and (_KW_TAX.search(line) or in_tax
                                                               or _KW_SUM.search(line) or _KW_FINAL.search(line)))
        if not amounts or _KW_IGNORE.search(line):
            continue
        last = abs(amounts[-1])
        pos = i / max(n - 1, 1)  # totals are rarely in the first fifth
        late = 0.5 if pos > 0.2 else 0.0

        if _KW_CHANGE.search(low):
            change.append(abs(last))
            continue
        if _KW_TAX.search(line) and not _KW_FINAL.search(line):
            continue  # VAT row, "SUMME MWST", "enth. MwSt" ...
        if tabular and not (_KW_FINAL.search(line) and len(amounts) == 1):
            # VAT table rows repeat net/tax/gross amounts; only its sum row's gross value is the total.
            if (_KW_SUM.search(line) or _KW_FINAL.search(line) or _summe_like(line)) and not _KW_TAX.search(line):
                vote(last, 2.0, "taxsum")
            continue
        if _KW_SUBTOTAL.search(line):
            vote(last, 1.5, "sub")
        elif _KW_FINAL.search(line):
            vote(last, 6.0 + late, "final")
        elif _KW_SUM.search(line) or _summe_like(line):
            vote(last, 4.0 + late + 0.5 * pos, "sum")
        elif _KW_CARD.search(line):
            vote(last, 3.0, "card")
        elif _KW_CASH.search(line):
            cash.append(last)
            vote(last, 1.0, "cash")

    # Cash given minus change = total.
    for c in cash:
        for r in change:
            if c - r > 0:
                vote(round(c - r, 2), 3.5, "cash-change")
    # A paid amount that ends like a "Summe" amount: the sum's first digit was misread ("Summe $1,26").
    for a, kinds in list(sources.items()):
        if kinds & {"card", "cash"}:
            for b, bk in list(sources.items()):
                if b != a and bk & {"sum", "final"} and a > b and f"{a:.2f}".endswith(f"{b:.2f}"):
                    vote(a, votes[b], "repaired")
    # Independent confirmation is the strongest signal.
    for a, kinds in sources.items():
        if len(kinds) > 1:
            votes[a] += 2.0 * (len(kinds) - 1)
    return votes


def amount_candidates_multi(texts: list[str]) -> dict[float, float]:
    """Sum the evidence of several OCR passes of the same receipt."""
    out: dict[float, float] = {}
    for t in texts:
        for a, v in amount_candidates(t).items():
            out[a] = out.get(a, 0.0) + v
    return out


def suggest_amount(text: str | list[str]) -> float | None:
    texts = [text] if isinstance(text, str) else list(text)
    votes = amount_candidates_multi(texts)
    text = "\n".join(texts)
    if votes:
        return max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
    # No keyword at all: the largest plausible amount that is no change/VAT/unit price.
    pool = []
    for line in text.splitlines():
        if _KW_CHANGE.search(line) or _KW_TAX.search(line) or _KW_IGNORE.search(line) or _KW_CASH.search(line):
            continue
        pool += [a for a in _parse_amounts(line) if 0 < a < 100000]
    return max(pool) if pool else None


def amount_confidence(text: str | list[str]) -> float:
    votes = amount_candidates_multi([text] if isinstance(text, str) else list(text))
    if not votes:
        return 0.0
    ranked = sorted(votes.values(), reverse=True)
    return ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)


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
