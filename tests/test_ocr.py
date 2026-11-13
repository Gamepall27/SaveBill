from datetime import date

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


def test_category():
    assert ocr.suggest_category(REWE) == "Lebensmittel"
    assert ocr.suggest_category(TANKE) == "Tanken"
    assert ocr.suggest_category(NOKW) == "Restaurant"
    assert ocr.suggest_category("foo") is None
