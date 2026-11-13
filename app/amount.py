"""Find the total of a receipt in (noisy) OCR text.

Receipts list every article with its price, then the total ("zu zahlen", "Summe", "TOTAL"),
the payment (Bar / EC / Karte / CASH) with change, and often a VAT table repeating amounts.
Every line is classified; every amount collects evidence (keyword kind, position, agreement
with the article sum or with cash-minus-change, ...). A learned logistic scorer
(tools/train_amount.py) ranks the candidates; the hand-made weights are the fallback.
"""
import re

import numpy as np

# 1.234,56 | 12,34 | 12.34 | 12, 34 | 0:79 (dot-matrix ":" decimal) ; not part of a longer number
_AMOUNT = re.compile(r"(?<![\d,.:])(-(?=\d))?(\d{1,3}(?:\.\d{3})+|\d{1,5})\s?[,.:]\s?(\d)\s?(\d)(?![\d,.:]?\d)")
# Dates and *real* clock times (two-digit hour). "0:79" or "0:65 B" are prices, not times.
_DATE_TIME = re.compile(r"\b\d{1,2}\s?[./]\s?\d{1,2}\s?[./]\s?\d{2,4}\b"
                        r"|\b(?:[01]\d|2[0-3])\s?:\s?[0-5]\d(?:\s?:\s?[0-5]\d)?\b"
                        r"|\b\d{1,2}:\d{2}\s*uhr\b|\b\d{4}-\d{2}-\d{2}\S*", re.I)

KW_FINAL = re.compile(r"zu\s*zahlen|zu\s*zah1en|zahlbetr|gesamt\s*-?betr|endbetrag|endsumme|rechnungsbetrag"
                      r"|gesamtsumme|summe\s*eur|total\s*eur|bon-?summe|zu\s*bezahlen|betrag\s*eur"
                      r"|grand\s*total|amount\s*due|nett?\s*total|total\s*(?:\(?rm\)?|incl|inclusive|amount|sales"
                      r"|payable|due)|rounded\s*total|total\s*:", re.I)
KW_SUM = re.compile(r"\b(summe|total|gesamt|zahlbetrag)\b", re.I)
KW_SUBTOTAL = re.compile(r"zw\.?\s*-?\s*summe|zwischen\s*summe|sub\s*-?\s*total", re.I)
KW_CARD = re.compile(r"\b(ec|karte|kartenzahlung|kreditkarte|girocard|visa|mastercard|master|maestro|v\s?pay|amex|"
                     r"debit|credit|kontaktlos|apple\s?pay|google\s?pay|paypal|ec-cash|electronic\s*cash)\b", re.I)
KW_CASH = re.compile(r"\b(bar|bargeld|gegeben|geg\.?|barzahlung|cash|tendered|tender|paid)\b", re.I)
# OCR often mangles the umlaut in "Rückgeld" ("R?ckgeld", "Ruckgeld", "Rueckgeld").
KW_CHANGE = re.compile(r"r\S{0,2}c?kgeld|ckgeld|r[uü]ckgabe|zur[uü]ck|wechselgeld|\bchange\b|\bbalance\b", re.I)
KW_TAX = re.compile(r"mwst|mw\.?\s?st|\bust\b|steuer|netto|brutto|umsatz|%|\bsatz\b|enth\.|\bgst\b|\bsst\b"
                    r"|\btax\b|excl", re.I)
KW_IGNORE = re.compile(r"rabatt|preisvorteil|ersparnis|gespart|sparen|punkte|payback|deutschlandcard|bonus"
                       r"|pfand\s*r[uü]ck|leergut|kg\s*x|/\s?kg|eur\s*/|\d\s?x\s|\bx\s?\d|st[uü]ck\s*x"
                       r"|trinkgeld|\btip\b|kilometer|liter\s*x|/\s?l\b|tse|signatur|seriennr|transaktion"
                       r"|\bqty\b|quantity|item\s*\(?s\)?|total\s*item|no\.?\s*of|discount|saving|rounding"
                       r"|points?\b|voucher", re.I)

_OCR_DIGIT_FIXES = [
    (re.compile(r"(?<![\w$])\$(?=\d)"), "5"),          # "$1,26" -> "51,26"
    (re.compile(r"(?<=\d)[oO](?=[\d,.])"), "0"),        # "1o,50" -> "10,50"
    (re.compile(r"(?<=[\s(])[oO](?=[,.]\d\d)"), "0"),   # " o,99" -> " 0,99"
    (re.compile(r"(?<=[,.])[oO](?=\d)|(?<=[,.]\d)[oO]\b"), "0"),  # "4,o5", "4,5o"
    (re.compile(r"(?<=\s)[lI|](?=\d[,.]\d\d)"), "1"),  # "l2,50" -> "12,50"
]
_TAX_CODE_END = re.compile(r"\d\s*[,.:]\s*\d\s?\d\s*[*]?\s*[A-D1-4|8]?\s*[*xX]?\s*$")
BANKNOTES = {5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0}


def norm_line(line: str) -> str:
    line = line.replace("€", " ")
    for pattern, repl in _OCR_DIGIT_FIXES:
        line = pattern.sub(repl, line)
    return _DATE_TIME.sub(" ", line)


def parse_amounts(line: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(norm_line(line)):
        whole = m.group(2).replace(".", "")
        try:
            v = float(f"{whole}.{m.group(3)}{m.group(4)}")
        except ValueError:
            continue
        out.append(-v if m.group(1) else v)
    return out


def summe_like(line: str) -> bool:
    """Fuzzy 'Summe' for OCR errors like SUMNE, SUNHE, Suane, SUMM, Summa."""
    for tok in re.findall(r"[A-Za-zÄÖÜäöü]{4,6}", line):
        t = tok.lower()
        if t[0] == "s" and t[1] in "uü" and (t.endswith(("e", "a")) or t.startswith("summ")) \
                and len(set(t) & set("mnhra")) >= 1:
            return True
    return False


def _classify(line: str) -> str | None:
    if KW_IGNORE.search(line):
        return "ignore"
    if KW_CHANGE.search(line):
        return "change"
    if KW_SUBTOTAL.search(line):
        return "sub"
    if KW_FINAL.search(line):
        return "final"
    if KW_TAX.search(line):
        return "tax"
    if KW_SUM.search(line) or summe_like(line):
        return "sum"
    if KW_CARD.search(line):
        return "card"
    if KW_CASH.search(line):
        return "cash"
    return None


def _letters(line: str) -> int:
    return sum(c.isalpha() for c in norm_line(line))


def _lines(text: str) -> list[dict]:
    rows = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        amounts = parse_amounts(raw)
        rows.append({
            "raw": raw, "amounts": amounts, "kind": _classify(raw),
            # a line holding only a price (dot-matrix printers raise prices half a line above their label)
            "orphan": len(amounts) == 1 and _letters(raw) <= 3,
            "item": len(amounts) == 1 and bool(_TAX_CODE_END.search(norm_line(raw))) and _letters(raw) >= 3,
        })
    n = len(rows)
    tax_until = -1
    for i, r in enumerate(rows):
        r["pos"] = i / max(n - 1, 1)
        if r["kind"] == "tax" and not r["amounts"]:
            tax_until = i + 4  # rows below a VAT header belong to the table
        k = len(r["amounts"])
        r["tabular"] = k >= 3 or (k >= 2 and (r["kind"] in ("tax", "sum", "final") or i <= tax_until))
    # Keyword lines without an amount borrow the price printed next to them (prefer the line above).
    for i, r in enumerate(rows):
        r["borrowed"] = False
        if r["kind"] in ("final", "sum", "sub", "card", "cash", "change") and not r["amounts"]:
            for j in (i - 1, i + 1):
                if 0 <= j < n and rows[j]["orphan"] and rows[j]["kind"] is None and not rows[j].get("taken"):
                    r["amounts"] = list(rows[j]["amounts"])
                    r["borrowed"] = True
                    rows[j]["taken"] = True
                    break
    return rows


# ---------------------------------------------------------------- evidence and features

EVIDENCE = ("final", "sum", "sub", "card", "cash", "taxsum", "cash_change", "items", "repaired", "orphan_only")
FEATURE_NAMES = EVIDENCE + ("borrowed", "late_pos", "n_occ", "n_kinds", "is_max", "frac_of_max", "banknote",
                            "near_cash", "in_item_line", "log_amount", "n_texts")


def evidence(texts: list[str]) -> dict[float, dict]:
    """amount -> evidence counters collected over one or more OCR passes of the same receipt."""
    ev: dict[float, dict] = {}

    def add(a, kind, pos=0.0, borrowed=False):
        if a is None or not (0 < a < 100000):
            return
        a = round(a, 2)
        e = ev.setdefault(a, {k: 0.0 for k in EVIDENCE} | {"borrowed": 0.0, "late_pos": 0.0, "n_occ": 0.0,
                                                          "near_cash": 0.0, "in_item_line": 0.0, "texts": set()})
        if kind:
            e[kind] += 1
        e["late_pos"] = max(e["late_pos"], pos)
        e["borrowed"] = max(e["borrowed"], float(borrowed))
        return e

    for t_idx, text in enumerate(texts):
        rows = _lines(text)
        cash, change, item_sum, n_items = [], [], 0.0, 0
        first_total = next((i for i, r in enumerate(rows) if r["kind"] in ("final", "sum")), len(rows))
        for i, r in enumerate(rows):
            for a in r["amounts"]:
                e = add(abs(a), None, r["pos"])
                if e is not None:
                    e["n_occ"] += 1
                    e["texts"].add(t_idx)
                    if r["item"]:
                        e["in_item_line"] += 1
            if r["item"] and i < first_total and r["kind"] is None:
                item_sum += r["amounts"][0]
                n_items += 1
            if not r["amounts"] or r["kind"] in ("ignore", None):
                continue
            last = abs(r["amounts"][-1])
            kind = r["kind"]
            if kind == "change":
                change.append(last)
                continue
            if kind == "tax" or r["tabular"]:
                # VAT rows repeat net/tax/gross; only a table's sum row carries the gross total.
                if r["tabular"] and kind in ("sum", "final") and not KW_TAX.search(r["raw"]):
                    add(last, "taxsum", r["pos"])
                if not (kind == "final" and len(r["amounts"]) == 1):
                    continue
            if kind == "cash":
                cash.append(last)
                for j in (i - 1, i + 1):  # banknote printed next to "Gegeben"
                    if 0 <= j < len(rows):
                        for b in rows[j]["amounts"]:
                            if round(abs(b), 2) in ev:
                                ev[round(abs(b), 2)]["near_cash"] = 1.0
            add(last, kind, r["pos"], r["borrowed"])
        for c in cash:
            for ch in change:
                if c - ch > 0:
                    add(round(c - ch, 2), "cash_change")
        if n_items >= 2:
            # A printed amount equal to the sum of the article prices is very likely the total.
            # (No new amount is invented: OCR drops or misreads articles too often.)
            s = round(item_sum, 2)
            if s in ev:
                ev[s]["items"] += 1
        # "Summe $1,26" + "VISA 51,26": the sum's first digit was misread
        for a in list(ev):
            if ev[a]["card"] or ev[a]["cash"]:
                for b in list(ev):
                    if b != a and (ev[b]["sum"] or ev[b]["final"]) and a > b and f"{a:.2f}".endswith(f"{b:.2f}"):
                        ev[a]["repaired"] += 1
    if ev:
        mx = max(ev)
        for a, e in ev.items():
            e["is_max"] = float(a == mx)
            e["frac_of_max"] = a / mx
            e["banknote"] = float(a in BANKNOTES)
            e["log_amount"] = float(np.log10(a + 1))
            e["n_texts"] = float(len(e.pop("texts")))
            e["n_kinds"] = float(sum(1 for k in EVIDENCE if e[k] > 0))
            e["orphan_only"] = float(e["n_kinds"] == 0)
    return ev


def features(e: dict) -> np.ndarray:
    return np.array([min(float(e[k]), 3.0) for k in FEATURE_NAMES], dtype=np.float64)


# Hand-made weights (fallback and baseline); a learned scorer replaces them when present.
HAND_WEIGHTS = {"final": 6.0, "sum": 4.0, "sub": 1.5, "card": 3.0, "cash": 1.0, "taxsum": 2.0, "cash_change": 3.5,
                "items": 3.0, "repaired": 4.0}
# Learned with tools/train_amount.py (None = hand-made weights).
SCORER_W = None
SCORER_B = 0.0


def _hand_score(e: dict) -> float:
    s = sum(HAND_WEIGHTS[k] * min(e[k], 2) for k in HAND_WEIGHTS)
    if s == 0:
        return 0.0
    s += 0.5 * e["late_pos"] + 2.0 * max(e["n_kinds"] - 1, 0) + 0.5 * e["borrowed"]
    if e["banknote"] and (e["cash"] or e["near_cash"]) and not (e["final"] or e["sum"] or e["card"]):
        return 0.0  # "Gegeben 50,00": money handed over, not the total
    return s


def score(e: dict) -> float:
    if SCORER_W is not None:
        z = float(np.dot(SCORER_W, features(e)) + SCORER_B)
        return 1.0 / (1.0 + np.exp(-z))
    return _hand_score(e)


def ranked(texts: str | list[str]) -> list[tuple[float, float]]:
    """[(amount, score), ...] best first."""
    texts = [texts] if isinstance(texts, str) else list(texts)
    ev = evidence(texts)
    out = [(a, score(e)) for a, e in ev.items()]
    out = [(a, s) for a, s in out if s > 0 or SCORER_W is not None]
    out.sort(key=lambda t: (-t[1], -t[0]))
    if not out or (SCORER_W is None and out[0][1] == 0):
        # no keyword evidence at all: largest plausible amount, never a banknote next to cash/change
        pool = [(a, 0.0) for a, e in ev.items()
                if not (e["banknote"] and (e["cash"] or e["near_cash"])) and not e["sub"]]
        out = sorted(pool, key=lambda t: -t[0])
    return out


def suggest(texts: str | list[str]) -> float | None:
    r = ranked(texts)
    return r[0][0] if r else None


def alternatives(texts: str | list[str], k: int = 3) -> list[float]:
    r = ranked(texts)
    return [a for a, _ in r[1:k + 1]]


def confidence(texts: str | list[str]) -> float:
    """Margin between best and second candidate (hand-made scale; 0 = nothing found)."""
    texts = [texts] if isinstance(texts, str) else list(texts)
    scores = sorted((_hand_score(e) for e in evidence(texts).values()), reverse=True)
    if not scores or scores[0] == 0:
        return 0.0
    return scores[0] - (scores[1] if len(scores) > 1 else 0.0)
