from datetime import date
from pathlib import Path

import pytest

from app import ocr

REWE = """REWE Markt GmbH
Hauptstr. 12
12345 Musterstadt
UID Nr.: DE123456789
EUR
MILCH 1,5% 1,19 B
BROT 2,49 B
KAFFEE 6,99 A
--------------------------
SUMME EUR 12,52
Geg. BAR EUR 20,00
Rückgeld BAR EUR 7,48
Steuer % Netto Steuer Brutto
A= 19,0% 5,87 1,12 6,99
13.11.26 14:32 Bon-Nr.:1234
"""

TANKE = """*** Willkommen ***
ARAL Tankstelle
Musterweg 3
Super E10 45,12 l 1,799 81,17
Gesamtbetrag EUR 81,17
girocard 81,17
Datum: 2026-10-02
"""

NOKW = """Pizzeria Da Mario
Margherita 8,50
Cola 3,20
Tiramisu 5,90
04/09/2026
"""


def test_amount_from_summe_line():
    assert ocr.suggest_amount(REWE) == 12.52


def test_amount_gesamtbetrag():
    assert ocr.suggest_amount(TANKE) == 81.17


def test_amount_fallback_max():
    assert ocr.suggest_amount(NOKW) == 8.50


def test_amount_thousands():
    assert ocr.suggest_amount("SUMME 1.234,56") == 1234.56


def test_date_two_digit_year():
    assert ocr.suggest_date(REWE, today=date(2026, 11, 13)) == "2026-11-13"


def test_date_iso():
    assert ocr.suggest_date(TANKE, today=date(2026, 11, 13)) == "2026-10-02"


def test_date_slashes_and_invalid_skipped():
    assert ocr.suggest_date("99.99.2026 then 04/09/2026", today=date(2026, 11, 13)) == "2026-09-04"


def test_date_none():
    assert ocr.suggest_date("no date here") is None


def test_title():
    assert ocr.suggest_title(REWE) == "REWE Markt"
    assert ocr.suggest_title(TANKE) == "ARAL Tankstelle"
    assert ocr.suggest_title(NOKW) == "Pizzeria Da Mario"


def test_title_uppercase_long_gets_titlecased():
    assert ocr.suggest_title("BAUHAUS HAMBURG\nStr. 1") == "Bauhaus Hamburg"


# ---------------------------------------------------------------- totals on real OCR output
# Tesseract output of real German receipts (Wikimedia Commons photos, OCR on the Pi),
# file name = <shop>_<total in cents>.

FIXTURES = Path(__file__).parent / "fixtures" / "ocr"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.txt")), ids=lambda p: p.stem)
def test_total_from_real_ocr(path):
    cents = int(path.stem.split("_")[1].rstrip("ab"))
    assert ocr.suggest_amount(path.read_text(encoding="utf-8")) == cents / 100


LIDL_CARD = """LIDL
Bananen 1,29 A
Milch 3,5% 0,99 A
2 x 0,99
Duschgel 1,95 B
zu zahlen 5,22
Kreditkarte 5,22
MWST% MWST + Netto = Brutto
A 7% 0,22 3,05 3,27
B 19% 0,31 1,64 1,95
Summe 0,53 4,69 5,22
16.02.18 15:25 Uhr
Kartenzahlung
EUR 5,22
"""


def test_total_prefers_zu_zahlen_and_ignores_vat_table():
    assert ocr.suggest_amount(LIDL_CARD) == 5.22


def test_second_summe_after_discount_wins():
    text = """SUMME [ 3] EUR 34,97
MwSt.-Senkung -0,88
Summe EUR 34,09
Bar Euro EUR 50,09
Rückgeld EUR -16,00
MwSt D 16,00% 29,39 4,70"""
    assert ocr.suggest_amount(text) == 34.09


def test_total_from_cash_minus_change_without_sum_line():
    text = """MIO MATE BANANE 0,69 A
PFAND 0,15 EUR 0,15 A
CLUB MATE 0,89 A
PFAND 0,15 EUR 0,15 A
Geg. BAR EUR 2,00
R?ckgeld BAR EUR 0,12
16. 02. 2018 15:44 Bon-Nr.:445"""
    assert ocr.suggest_amount(text) == 1.88


def test_cash_given_and_vat_sum_are_not_the_total():
    text = """Kaffee 6,99 A
Brot 2,49 A
Summe 9,48
Bar 50,00
Rückgeld 40,52
SUMME MWST 0,62 8,86"""
    assert ocr.suggest_amount(text) == 9.48


def test_ocr_digit_repair_with_card_payment():
    # "5" of the sum misread as "$"; the card line confirms 51,26
    text = "Pizza 12,50\nSumme $1,26\nVISA 51,26\n"
    assert ocr.suggest_amount(text) == 51.26


def test_ignores_dates_quantities_savings_and_weights():
    text = """Tomaten 2,49 A
0,374 kg x 9,90 EUR/kg
Salatbar 3,70 A
3 x 1,29
Wasser 3,87 B
Sie haben 12,50 EUR gespart!
SUMME EUR 10,06
24.12.2025 18:30"""
    assert ocr.suggest_amount(text) == 10.06


def test_multi_pass_evidence_is_merged():
    first = "Brot 2,49\nSumme ?\nEC-Karte\n"
    second = "Brot 2,49\nSumme 7,48\nEC-Karte 7,48\n"
    assert ocr.suggest_amount([first, second]) == 7.48


def test_category():
    assert ocr.suggest_category(REWE) == "Lebensmittel"
    assert ocr.suggest_category(TANKE) == "Tanken"
    assert ocr.suggest_category(NOKW) == "Restaurant"
    assert ocr.suggest_category("foo") is None


# ---------------------------------------------------------------- dot-matrix / orphan / english

# Anonymised pattern of an old dot-matrix Lidl receipt: prices printed half a line above
# their label (OCR puts them on their own line), ":" read as decimal comma, two running sums.
NADEL = """LIDL
KARTOFFELN 2 KG 0:79 B
ERDBEER KONF 0:99 B
EIER 10ER 0,65 B
H-MILCH 3,5 0,85 B
5,32
Summe 8 Pos.
JOGHURT 0,29 B
CROISSANT 0,89 B
9,33
Summe 17 Pos.
50,00
Gegeben BAR EUR
40,67
Rückgeld EUR
MWST B 7,00% 9,33 enth. 0,61
9397/002/02 26.06.02 18:11
"""


def test_dot_matrix_orphan_prices_and_running_sums():
    assert ocr.suggest_amount(NADEL) == 9.33
    alts = ocr.amount_alternatives(NADEL)
    assert 5.32 in alts


def test_colon_decimal_is_price_but_clock_time_is_not():
    assert ocr._parse_amounts("KARTOFFELN 2 KG 0:79 B") == [0.79]
    assert ocr._parse_amounts("26.06.02 18:11") == []
    assert ocr._parse_amounts("Datum 16.02.18 15:25 Uhr") == []


def test_banknote_given_is_never_the_total():
    text = "BROT 2,49 B\nKAFFEE 4,99 B\n50,00\nGegeben BAR EUR\n"
    assert ocr.suggest_amount(text) != 50.0


def test_item_sum_confirms_the_total():
    text = """Brot 2,49 A
Milch 1,19 A
Kaffee 6,99 B
10,67
Bar 20,00
Rückgeld 9,33"""
    assert ocr.suggest_amount(text) == 10.67


def test_english_receipt_total_cash_change():
    text = """KEDAI MAKAN
NASI LEMAK 1 x 4.50 4.50
TEH TARIK 1 x 2.20 2.20
TOTAL QTY: 2
SUB TOTAL 6.70
ROUNDING 0.00
TOTAL (RM) 6.70
CASH 10.00
CHANGE 3.30"""
    assert ocr.suggest_amount(text) == 6.70
