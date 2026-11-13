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

## Entwicklung
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8091   # Daten in ./data
```
Ohne lokal installiertes Tesseract liefert die OCR einfach keine Vorschläge.
