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
Der Endbetrag wird nicht über „erste Zeile mit Summe“ gesucht. Stattdessen werden Belege gewichtet:
- „zu zahlen“, „Gesamtbetrag“ und „SUMME EUR“ zählen stärker als „Summe“ oder „Total“.
- Kartenzahlungen und „Bar minus Rückgeld“ bestätigen einen Betrag.
- Ignoriert werden MwSt-Tabellen (außer deren Brutto-Summe als schwacher Hinweis), Rückgeld, Mengen- und Kilopreise („2 x 0,99“, „0,374 kg x“), „Sie haben … gespart“, Datum und Uhrzeit.
- Typische OCR-Fehler werden korrigiert: „$1,26“ → „51,26“, „R?ckgeld“, „SUMNE“.

Die OCR läuft auf dem *nicht* kontrastverstärkten Zuschnitt: Hintergrund-normalisiert, 1400 px breit, Tesseract-Layoutmodus 4. Ist der Betrag unsicher, folgt ein zweiter Durchgang (Otsu-binarisiert, Modus 6), und die Belege beider Durchgänge werden zusammengerechnet.

| Endbetrag richtig (Ende-zu-Ende auf dem Pi) | vorher | jetzt |
|---|---|---|
| 12 echte Bons (Lidl, Edeka, Rewe, Reno, Lotto, E-Center, Fressnapf, DDR) | 2 | **11** |
| 64 gerenderte Bons (Trainingssatz) | 56 | **59** |
| 48 gerenderte Bons, ungesehen | 42 | **42** |

Ein Scan dauert auf dem Pi dadurch etwa 3–4 s statt rund 2 s.
```bash
python tools/make_german_bons.py <hintergruende> ../bons --n 64          # gerenderte Bons mit bekanntem Endbetrag
tar c app | docker exec -i savebill sh -c "rm -rf /tmp/newapp; mkdir /tmp/newapp; tar x -C /tmp/newapp"
docker exec -i savebill python - /tmp/bench/bons < tools/ocr_dump.py      # Ende-zu-Ende-Messung im Container
```

## Entwicklung
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8091   # Daten in ./data
```
Ohne lokal installiertes Tesseract liefert die OCR einfach keine Vorschläge.
