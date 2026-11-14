# SaveBill

Quittungen mit dem Handy fotografieren, automatisch zuschneiden, mit Name, Datum, Betrag und Kategorie versehen und später auf dem Handy oder Desktop wiederfinden. Die App läuft als Docker-Container auf dem Raspberry Pi 5 und ist über Tailscale erreichbar.

**Aufruf:** http://raspberrypi5:8080 (bzw. http://100.117.112.24:8080)

## Funktionen
- Foto aufnehmen oder aus der Galerie wählen. Die Ecken der Quittung werden automatisch erkannt (OpenCV) und lassen sich mit einer Lupe korrigieren. Danach wird das Bild perspektivisch entzerrt.
- Die Texterkennung (Tesseract, Deutsch und Englisch) schlägt Titel, Datum, Betrag und Kategorie vor.
- Die Liste ist nach Monaten gruppiert und zeigt die Summen. Es gibt eine Suche (auch im erkannten Text) und Filter nach Zeitraum und Kategorie.
- In der Detailansicht kannst du die Angaben bearbeiten, die Quittung neu zuschneiden (das Original bleibt erhalten) oder löschen.
- Export als JPG/PDF einzeln oder als ZIP (JPGs + CSV) bzw. Sammel-PDF für den aktuellen Filter.

## Daten
Alles liegt auf dem Pi unter `~/SaveBill/data`: die SQLite-DB plus die Ordner `images/`, `originals/` und `thumbs/`. Für ein Backup reicht es, diesen Ordner zu sichern.

## Deployment
```bash
./deploy.sh            # überträgt den aktuellen Commit nach ~/SaveBill und startet neu
```

## Erkennung mit echten Fotos prüfen
Als Testdaten dienen 200 echte Quittungsfotos aus dem öffentlichen Datensatz [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC BY 4.0). Bei 77 davon sind die Ecken der Quittung eingezeichnet. Die Bilder liegen nicht im Repo, sie werden bei Bedarf heruntergeladen:
```bash
python tools/fetch_cord.py ../cord                       # Bilder + manifest.json
python tools/eval_detection.py ../cord --overlays ../ov  # IoU-Auswertung + Bilder mit Markierung
```

| Stand | mittlere IoU | IoU ≥ 0,9 | IoU ≥ 0,8 |
|---|---|---|---|
| Erste Version | 0,753 | 30 % | 48 % |
| Weißheits-Masken, Kanten-Prüfung, Kanten-Einrasten | **0,880** | **68 %** | **81 %** |

Ein Teil der verbleibenden Abweichungen geht auf ungenau eingezeichnete Ecken im Datensatz zurück.

### Kleine Quittungen
CORD enthält kaum kleine Quittungen. Deshalb setzt `tools/make_small_set.py` echte CORD-Quittungen klein (3–25 % des Bildes) mit Perspektive, Schatten, Licht und Rauschen auf echte Untergrund-Fotos (z. B. Holz, Granit, Tischdecke, Autositz, Hand von Wikimedia Commons). Die Ecken sind dabei exakt bekannt.

Kleine Kandidaten wählt eine gelernte logistische Bewertung aus (`tools/train_scoring.py`, 13 Merkmale wie Kantenstütze, Kontrast zur Umgebung, Form, Druckdichte und Seitenverhältnis). Bei großen Quittungen bleibt die bisherige Bewertung zuständig. Kleine Kandidaten bekommen außerdem einen zweiten, vergrößerten Durchgang.
```bash
python tools/make_small_set.py ../cord <hintergruende> ../small      --seed 7    # Trainingssatz
python tools/make_small_set.py ../cord <hintergruende> ../small_eval --seed 11   # ungesehener Prüfsatz
python tools/train_scoring.py ../cord_test ../small --eval ../cord_val ../small_eval
```

| Datensatz (mittlere IoU) | vorher | jetzt |
|---|---|---|
| CORD, 77 Fotos | 0,880 | 0,877 |
| kleine Quittungen, ungesehen (150) | 0,416 (≥ 0,9: 33 %) | **0,616** (≥ 0,9: 48 %) |

Schwierig bleiben sehr kleine Quittungen (< 6 % des Bildes) auf hellem, unruhigem Untergrund (Spitzendecke, graues Holz, Granit) und andere weiße Gegenstände im Bild.

## Texterkennung und Endbetrag
Die Logik steckt in `app/amount.py`. Jeder Betrag im OCR-Text sammelt Belege dafür, der Endbetrag zu sein:
- **Schlüsselwörter:** „zu zahlen“, „Gesamtbetrag“, „SUMME EUR“, TOTAL; zusätzlich zählt die *letzte* „Summe N Pos.“.
- **Gegenproben:** Kartenzahlung, „Gegeben/Bar minus Rückgeld“, die Artikelsumme und der Brutto-Betrag in „MWST B 7,00 % 9,33 enth. 0,61“.
- **Nadeldruck:** Preise, die eine halbe Zeile über ihrem Text gedruckt sind, werden der Schlüsselwort-Zeile zugeordnet. „0:79“ zählt als Preis, „18:11“ als Uhrzeit.
- **Ignoriert:** MwSt-Tabellen, „TOTAL GST“, Rückgeld, Mengen- und Kilopreise, „Sie haben … gespart“, Datum und Uhrzeit. Ein Geldschein neben „Gegeben“ ist nie der Endbetrag.
- **Weitere Kandidaten** zeigt das Formular als antippbare Chips an („Auch erkannt: …“).

**OCR:** Der erste Durchgang läuft auf dem hintergrund-normalisierten Zuschnitt (1400 px, Tesseract-Layoutmodus 4). Ist der Betrag unsicher, folgt ein zweiter Durchgang (Modus 6) auf dem kontrastreichsten Farbkanal, z. B. Grün für violette Tinte. Dabei werden die Nadeldruck-Punkte zu Strichen verschmolzen und Otsu-binarisiert. Die Belege beider Durchgänge werden zusammengerechnet.

| Endbetrag richtig (Ende-zu-Ende auf dem Pi) | bisher | jetzt |
|---|---|---|
| Echte deutsche Bons (Commons) | 11/12 | 11/12 |
| Echte Bons SROIE, Prüfteil | 60/93 | **77/93** |
| Eigenes Foto, alter Lidl-Nadeldruck (nur lokal) | 0/1 (50,00 statt 9,33) | **1/1** |
| Gerendert, Nadeldruck-Stil, ungesehen | 8/30 | **14/30** |
| Gerendert, Nadeldruck-Stil, Training | 13/40 | 17/40 |
| Gerendert, Thermodruck (Training / ungesehen) | 59/64 · 42/48 | 59/64 · 41/48 |

Ein Scan dauert auf dem Pi etwa 4 s. Nadeldruck bleibt schwierig: Bei rund 40 % dieser Bons steht der Endbetrag gar nicht im OCR-Text.

Eine gelernte Bewertung der Kandidaten (`tools/train_amount.py`, wie bei der Quittungserkennung) wurde trainiert, ist aber *nicht aktiv*. Auf echten Bons war sie nicht verlässlich besser als die Handlogik, sie griff z. B. bei verlesenen Beträgen wie „65,00“ statt „€5,00“ daneben.

```bash
python tools/fetch_sroie.py ../bons/sroie_test --split test --n 100            # echte Bons mit Endbetrag
python tools/make_german_bons.py <hintergruende> ../bons/nadel --styles nadel   # Nadeldruck-Stil
docker exec -i savebill python - /tmp/bench/<satz> < tools/ocr_variants.py      # OCR-Varianten
docker exec -i savebill python - /tmp/bench/<satz> < tools/ocr_dump.py          # Ende-zu-Ende
python tools/train_amount.py ../bons --train sroie_train nadel synth --eval sroie_test nadel_eval
```

## Entwicklung
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8091   # Daten in ./data
```
Ohne lokal installiertes Tesseract liefert die OCR einfach keine Vorschläge.
