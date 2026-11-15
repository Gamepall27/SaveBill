# Referenzdaten für Portierungen

Die Python-Fassung in `app/` ist die Referenz. Diese Dateien halten fest, was sie für feste Eingaben berechnet. Eine Portierung (geplant: native App in Kotlin/Swift, siehe `docs/PLAN-berechnung-auf-dem-geraet.md`) muss dieselben Ergebnisse liefern. `tests/test_conformance.py` stellt sicher, dass Python und Referenzdaten übereinstimmen. Ist eine Änderung an `app/` gewollt, werden die Daten neu erzeugt:

```bash
python tools/fetch_cord.py ../cord                                  # nur für detection_cord.json und die CORD-Netzeingaben
python tools/export_conformance.py --cord ../cord
```

## `suggest_cases.json` – Text → Vorschläge (muss exakt übereinstimmen)
43 Fälle: echte Tesseract-Ausgaben deutscher und malaysischer Bons, die Beispiele aus `tests/test_ocr.py` und Fälle mit mehreren OCR-Durchgängen. Pro Fall:

| Feld | Bedeutung | Python |
|---|---|---|
| `texts` | ein oder mehrere OCR-Texte desselben Bons | – |
| `today` | festes „heute“ für die Plausibilität des Datums | – |
| `expected.title` / `date` / `category` | aus dem ersten Text | `ocr.suggest_title`, `suggest_date(today=…)`, `suggest_category` |
| `expected.amount` / `alternatives` | Endbetrag und weitere Kandidaten aus allen Texten | `amount.suggest`, `amount.alternatives` |
| `expected.confidence` | Abstand bester zu zweitbestem Kandidaten (6 Nachkommastellen) | `amount.confidence` |
| `expected.ranked` | bis zu 8 Kandidaten `[Betrag, Punkte]`, beste zuerst | `amount.ranked` |
| `expected.lines` | Zerlegung des ersten Textes je Zeile: Beträge, Art (`final`, `sum`, `sub`, `card`, `cash`, `change`, `tax`, `ignore`, `null`), Flags | `amount._lines` |

`lines` dient der Fehlersuche: Weicht ein Endbetrag ab, zeigt die erste abweichende Zeile die Ursache (meist Unterschiede bei regulären Ausdrücken, `\b` mit Umlauten, Groß-/Kleinschreibung, Rundung). Beträge sind Dezimalzahlen mit zwei Nachkommastellen; Vergleiche auf Cent genau.

## `segmentation/` – Netzeingabe → Wahrscheinlichkeitskarte (Toleranz 1e-3 in Python, 0,02 je Pixel für andere Laufzeiten)
Für jedes Beispiel `<name>_input.png` (256×256, BGR, so wie es das Netz bekommt: lange Seite auf 256 skaliert, Rand mit 114 gefüllt) und `<name>_prob.png` (16 Bit, Wahrscheinlichkeit × 65535). Das Netz bekommt die Pixel als float32 im Format `1×3×256×256` (BGR, Werte 0–255, die Normalisierung steckt im Modell). Modell: `app/models/receipt_seg.onnx`. Nach Umwandlung (Core ML, TFLite, float16) gelten max. 0,02 Abweichung je Pixel und 0,002 im Mittel als bestanden.

Die drei `cord_*`-Beispiele stammen aus CORD v2 (naver-clova-ix/cord-v2, CC BY 4.0).

## `detection_synth/` – Foto → Ecken (Toleranz 1 % der Bilddiagonale)
Synthetische Fotos (`tests/synth.py`) und ein leeres Bild. `expected.json` enthält Bildgröße, die tatsächliche Lage der Quittung (`truth`), die von Python erkannten Ecken (oben links, oben rechts, unten rechts, unten links; in Pixeln des geladenen Bildes) und `found`. Eine Portierung muss `found` exakt treffen und die Ecken mit höchstens 1 % der Bilddiagonale Abweichung.

## `detection_cord.json` – echte Fotos → Ecken (Gesamtwert)
77 CORD-Prüffotos mit den von Hand geprüften Ecken (`label`, aus `tools/cord_labels_verified.json`), den Ecken der Python-Fassung und deren IoU; Mittelwert 0,958. Die Fotos selbst lädt `tools/fetch_cord.py`. Koordinaten beziehen sich auf das wie in der App geladene Bild (lange Seite höchstens 3000 px). Abnahme einer Portierung: mittlere IoU gegen `label` höchstens 0,005 unter dem Python-Wert.

## Weitere Prüfsätze
Die zusammengesetzten Prüfsätze (kleine und gemischt große Quittungen, je 150 Bilder) lassen sich nur mit dem Python-Werkzeug erzeugen. Welche Untergründe dafür von Hand ausgewählt wurden, steht mit Quelle und Lizenz in `tools/eval_backgrounds.json`. Die erzeugten Bilder dienen einer Portierung als Eingabe; Python-Werte: mittlere IoU 0,977 (klein) bzw. 0,960 (gemischt).
