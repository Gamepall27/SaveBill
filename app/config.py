import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("SAVEBILL_DATA") or Path(__file__).resolve().parent.parent / "data")
DB_PATH = DATA_DIR / "savebill.db"
IMAGES_DIR = DATA_DIR / "images"
THUMBS_DIR = DATA_DIR / "thumbs"
ORIGINALS_DIR = DATA_DIR / "originals"
SCANS_DIR = DATA_DIR / "scans"  # temporary uploads before a receipt is saved

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_SIDE = 3000  # uploaded photos are downscaled to this long side
SCAN_TTL_SECONDS = 24 * 3600


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGES_DIR, THUMBS_DIR, ORIGINALS_DIR, SCANS_DIR):
        d.mkdir(parents=True, exist_ok=True)
