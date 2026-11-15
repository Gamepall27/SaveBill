# Berechnung aufs Handy verlagern – Plan in 3 Schritten

## Kontext
Vom Handy aus dauert ein Scan deutlich länger als die reine Rechenzeit auf dem Pi (Erkennung 0,28 s, Texterkennung ~4 s, bei unsicherem Betrag zwei Durchgänge). Ursachen laut Code:
- `POST /api/scans` lädt das (schon auf 2800 px verkleinerte) JPEG hoch, danach lädt der Eckeneditor das Bild **als verlustfreies PNG** (`/api/scans/{id}/image`, typ. 8–12 MB) wieder herunter.
- `POST /api/scans/{id}/crop` ist ein zweiter Round-Trip mit OCR auf dem Pi.

Ziel: Erkennung, Zuschnitt, Texterkennung und Endbetrag laufen im Handy-Browser; der Pi speichert nur noch (Original + Zuschnitt + Ergebnisse) und bleibt Rückfall. Später folgt eine **native App (Kotlin/Swift)** – der JS-Port ist daher eine Zwischenstufe. Deshalb wird jede Portierung gegen **sprachneutrale Referenzdaten** geprüft, die die Python-Fassung erzeugt; dieselben Daten dienen später als Konformitätstests für Kotlin/Swift.

Entscheidungen (Nutzer): native App später · Pi bleibt Rückfall und Referenz · Upload von Original (verkleinert) + Zuschnitt.

## Vorab – Plan ins Projekt übernehmen und tracken
- Diesen Plan vollständig als `docs/PLAN-berechnung-auf-dem-geraet.md` im Repo ablegen (Branch `develop`), committen und pushen.
- Während der Umsetzung pro Schritt den Status in dieser Datei pflegen (offen / in Arbeit / erledigt mit Datum, Messwerten, Commit) und mit dem jeweiligen Schritt committen.

## Schritt 0 – Messen (½ Tag, vor allem anderen)
- Zeitstempel im Frontend (`static/app.js`: `prepareImage`, `api()`) und Server-Timing-Header in `app/main.py` (`create_scan`, `crop_scan`): Upload, Erkennung, Bild-Download, OCR getrennt.
- Vom Handy über Tailscale (WLAN und Mobilnetz) je 5 Scans; Ergebnis als Ausgangswert festhalten.
- Sofortmaßnahme unabhängig vom Port: Eckeneditor nutzt das lokale Foto (`URL.createObjectURL(blob)`) statt des PNG-Downloads.

### Ergebnis Schritt 0 (15.11.2026, iPhone, Safari, 6 Scans)
| Teilschritt | gemessen |
|---|---|
| Foto verkleinern auf dem Handy | 0,01–0,15 s |
| Upload + Erkennung (Anfrage gesamt) | 0,5–1,7 s, davon Pi 0,14–1,0 s (PNG speichern 0,02–0,48 s, Erkennung 0,12–0,46 s) |
| Bild für den Eckeneditor | **2,4 s** als PNG vom Pi, **0,04 s** lokal |
| Zuschnitt + Texterkennung (Anfrage gesamt) | **3,5–11,5 s**, davon Texterkennung auf dem Pi 3,3–11,3 s |
| Zuschnitt-Vorschau laden | 0,16–0,78 s |

Folgerungen:
- Die Texterkennung ist mit Abstand der größte Posten (Ø ≈ 6 s, bis 11 s bei zwei Durchgängen). Upload und Erkennung zusammen kosten nur 0,5–1,7 s.
- Sofortmaßnahme umgesetzt: Der Eckeneditor zeigt das Foto vom Handy (spart ≈ 2,4 s pro Scan). Die Zeitmessung bleibt im Log, damit die folgenden Schritte verglichen werden können (`docker logs savebill | grep scan-metrics`).
- Ob `tesseract.js` auf dem iPhone schneller ist als der Pi, ist offen. Deshalb kommt in Schritt 2 **zuerst** ein Probelauf: `tesseract.js` mit denselben vorverarbeiteten Bildern auf dem iPhone messen, bevor die Logik portiert wird. Ist es nicht schneller, bleibt die Texterkennung bis zur nativen App (Apple Vision / ML Kit) auf dem Pi.

## Schritt 1 – Quittungsform im Browser (1–2 Tage)
**Neu:** `static/detect/` (ES-Modul `detect.js`, ohne UI-Abhängigkeit, damit Logik klar getrennt bleibt).
- Netz: `onnxruntime-web` (WASM, feste Version, lokal ausgeliefert unter `static/vendor/`), Modell `app/models/receipt_seg.onnx` über eine neue statische Route ausliefern. Vorverarbeitung exakt wie `scanner.receipt_probability` (Letterbox 256, Füllwert 114, BGR, float).
- Nachverarbeitung 1:1 aus `app/scanner.py` portieren mit `opencv.js` (lokal, nur bei Bedarf geladen, Service-Worker-Cache): `_prepare`, `_whiteness`, `_main_component`, `_grow_blank_paper`, `_candidate_masks`/`_candidates`, `_quad_from_contour`, `_soft_iou`, `_refine`, `_edge_support`, `_intersect`, `order_corners`, `_detect_seg`, `_detect_with_seg` (Zoom-Durchgang). Klassischer Rückfall (`_detect_once`/`_detect_learned`) **nicht** portieren – ohne Netz rechnet der Pi.
- Perspektivischer Zuschnitt (`scanner.warp`) im Browser per `cv.warpPerspective`; `enhance` ebenfalls (CLAHE + Schärfen) für das gespeicherte Bild.
- Fähigkeitsprüfung (WASM, Speicher) → sonst bisheriger Server-Weg.

**Referenzdaten / Prüfung:**
- `tools/export_reference.py` (neu): für die 3 Prüfsätze (CORD mit `cord_labels_verified.json`, small_eval, mixed_eval) je Bild Netz-Wahrscheinlichkeitskarte (Prüfsumme + Stichproben), gewählte Ecken der Python-Fassung → JSON.
- `tools/eval_detection_js.mjs` (neu, Node mit `onnxruntime-node` + `opencv.js`): gleiche IoU-Auswertung wie `tools/eval_detection.py`; Abweichung der Ecken zur Python-Fassung.
- Abnahme: mittlere IoU je Prüfsatz höchstens 0,005 unter Python (0,958 / 0,977 / 0,960); Netz-Ausgabe max. Abweichung < 1e-3.
- Echtgerät: Zeit pro Foto auf einem Mittelklasse-Android und einem iPhone (Ziel < 1 s inkl. Laden aus Cache).

## Schritt 2 – Texterkennung und Endbetrag im Browser (2–4 Tage)
- **Zuerst Probelauf (½ Tag):** eine Testseite mit `tesseract.js` (deu+eng, PSM 4/6) auf vorverarbeiteten Bons; Zeit auf dem iPhone gegen die Pi-Werte aus Schritt 0. Nur bei klarem Gewinn weiter.
- OCR-Vorverarbeitung aus `app/ocr.py` portieren (`_scaled_gray`, `normalized`, `ink_channel`, `binarized`) mit `opencv.js`.
- `tesseract.js` (Tesseract 5, WASM) mit `deu+eng`, PSM 4 bzw. 6, Sprachdaten lokal ausliefern und cachen; Ablauf wie `ocr.analyze` (zweiter Durchgang nur bei `confidence < CONFIDENT`).
- Reine Textlogik nach JS (`static/detect/suggest.js`): `app/amount.py` komplett (Regexe, `_lines`, `evidence`, `_hand_score`, `ranked`, `suggest`, `alternatives`, `confidence`) und aus `app/ocr.py` `suggest_date`, `suggest_title`, `suggest_category`. Achtung Regex-Unterschiede Python ↔ JS (Lookbehind ok in modernen Browsern, `\b` mit Umlauten, `re.I`).

**Referenzdaten / Prüfung (Kern für die spätere native App):**
- Golden-Tests Text → Vorschläge: alle Fälle aus `tests/test_ocr.py` plus `tests/fixtures/ocr/*.txt` als JSON (`tests/fixtures/suggest_cases.json`, erzeugt aus Python). JS muss 100 % identisch sein; später Pflicht auch für Kotlin/Swift.
- Ende-zu-Ende Endbetrag: dieselben Bon-Sätze wie im README (Commons 12, SROIE-Prüfteil 93, gerendert Nadeldruck/Thermo) einmal mit Pi-Tesseract (`tools/ocr_dump.py`) und einmal mit `tesseract.js` (Node). Abnahme: nicht schlechter als die Pi-Werte (11/12, 77/93, …); Unterschiede einzeln ansehen.
- Zeit auf Echtgerät messen; Ziel: schneller als heute inkl. Upload.

## Schritt 3 – Nur Ergebnis hochladen, Pi als Rückfall (1 Tag)
- Neuer Endpunkt in `app/main.py`: `POST /api/receipts/upload` (multipart: Original-JPEG ≤ 2800 px, Zuschnitt-JPEG, JSON mit Ecken, OCR-Text, Feldern). Speichert wie `create_receipt` (Originals, Images, Thumbs, DB inkl. `corners`, `ocr_text`), validiert Ecken/Felder mit den bestehenden Pydantic-Modellen.
- Frontend-Ablauf: Foto → lokal erkennen → Eckeneditor mit lokalem Bild → lokal zuschneiden + OCR → Formular → ein Upload. Bei fehlender Fähigkeit oder Fehler: bisheriger Weg über `/api/scans` (bleibt unverändert).
- `recrop` bleibt serverseitig (nutzt das gespeicherte Original).
- Tests: `tests/test_api.py` um den neuen Endpunkt (gültig, fehlende Teile, zu groß, falsche Ecken).

## Reihenfolge, Branches, Abschluss
- Je Schritt ein eigener Branch von `develop` (`on-device-detection`, `on-device-ocr`, `on-device-upload`), Merge nach `develop` erst nach erfüllter Abnahme, dann `./deploy.sh`.
- README: neuer Abschnitt „Berechnung auf dem Gerät“ mit Messwerten und den Referenz-/Konformitätstests als Vorgabe für die native App.

## Verifikation (gesamt)
1. `pytest` (Python bleibt Referenz) und neue JS-Tests (`node --test`) grün.
2. Node-Auswertungen gegen Prüfsätze erreichen die Abnahmewerte aus Schritt 1 und 2.
3. Echtgeräte-Test über Tailscale: Zeiten vor/nach (Schritt 0 als Basis), Rückfall durch erzwungenes Abschalten des Gerätepfads prüfen.
4. Nach Deploy: Healthcheck, ein Scan vom Handy end-to-end, Quittung erscheint in Liste mit Bild, Betrag, Neu-Zuschneiden funktioniert.

## Status
| Schritt | Stand | Datum | Messwerte / Commit |
|---|---|---|---|
| Vorab – Plan im Projekt | erledigt | 15.11.2026 | dieser Commit |
| 0 – Messen | erledigt | 15.11.2026 | siehe „Ergebnis Schritt 0“; lokales Bild im Eckeneditor eingebaut |
| 1 – Quittungsform im Browser | offen | | |
| 2 – Texterkennung und Endbetrag im Browser | Probelauf in Arbeit | 15.11.2026 | Testseite /ocr-probe.html (tesseract.js 5.1.1, Pi-Sprachdaten) |
| 3 – Nur Ergebnis hochladen, Pi als Rückfall | offen | | |
