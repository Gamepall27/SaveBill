import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    date         TEXT NOT NULL,          -- YYYY-MM-DD
    amount_cents INTEGER,                -- NULL = unknown
    category     TEXT,
    note         TEXT,
    ocr_text     TEXT,
    corners      TEXT,                   -- JSON [[x,y] x4] in original image coords
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(date);
"""

FIELDS = ("title", "date", "amount_cents", "category", "note")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    # Unicode-aware lower() so that search for "müller" matches "MÜLLER".
    con.create_function("ulower", 1, lambda s: s.lower() if isinstance(s, str) else s, deterministic=True)
    return con


@contextmanager
def connection():
    con = _connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init() -> None:
    config.ensure_dirs()
    with connection() as con:
        con.executescript(SCHEMA)


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    cents = d.pop("amount_cents")
    d["amount"] = None if cents is None else cents / 100
    d["corners"] = json.loads(d["corners"]) if d.get("corners") else None
    return d


def to_cents(amount) -> int | None:
    if amount is None or amount == "":
        return None
    return int(round(float(amount) * 100))


def insert(rec_id: str, *, title: str, date: str, amount, category, note, ocr_text, corners) -> dict:
    now = _now()
    with connection() as con:
        con.execute(
            "INSERT INTO receipts (id, title, date, amount_cents, category, note, ocr_text, corners, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rec_id, title, date, to_cents(amount), category or None, note or None, ocr_text,
             json.dumps(corners) if corners else None, now, now),
        )
    return get(rec_id)


def get(rec_id: str) -> dict | None:
    with connection() as con:
        row = con.execute("SELECT * FROM receipts WHERE id = ?", (rec_id,)).fetchone()
    return row_to_dict(row) if row else None


def update(rec_id: str, changes: dict) -> dict | None:
    sets, params = [], []
    for key, value in changes.items():
        if key == "amount":
            sets.append("amount_cents = ?")
            params.append(to_cents(value))
        elif key == "corners":
            sets.append("corners = ?")
            params.append(json.dumps(value) if value else None)
        elif key in FIELDS:
            sets.append(f"{key} = ?")
            params.append(value if value != "" or key in ("title", "date") else None)
    if not sets:
        return get(rec_id)
    sets.append("updated_at = ?")
    params.append(_now())
    params.append(rec_id)
    with connection() as con:
        con.execute(f"UPDATE receipts SET {', '.join(sets)} WHERE id = ?", params)
    return get(rec_id)


def delete(rec_id: str) -> bool:
    with connection() as con:
        cur = con.execute("DELETE FROM receipts WHERE id = ?", (rec_id,))
    return cur.rowcount > 0


def search(q: str | None = None, date_from: str | None = None, date_to: str | None = None,
           category: str | None = None) -> list[dict]:
    where, params = [], []
    if q:
        for term in q.lower().split():
            like = f"%{term}%"
            where.append("(ulower(title) LIKE ? OR ulower(COALESCE(note,'')) LIKE ?"
                         " OR ulower(COALESCE(category,'')) LIKE ? OR ulower(COALESCE(ocr_text,'')) LIKE ?)")
            params += [like] * 4
    if date_from:
        where.append("date >= ?")
        params.append(date_from)
    if date_to:
        where.append("date <= ?")
        params.append(date_to)
    if category:
        where.append("category = ?")
        params.append(category)
    sql = "SELECT * FROM receipts"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date DESC, created_at DESC"
    with connection() as con:
        rows = con.execute(sql, params).fetchall()
    return [row_to_dict(r) for r in rows]


def categories() -> list[str]:
    with connection() as con:
        rows = con.execute(
            "SELECT category, COUNT(*) AS n FROM receipts WHERE category IS NOT NULL AND category != ''"
            " GROUP BY category ORDER BY n DESC, category"
        ).fetchall()
    return [r["category"] for r in rows]
