import importlib
import io
import zipfile

import cv2
import pytest
from fastapi.testclient import TestClient

from tests.synth import photo

TRUTH = [[350, 150], [950, 260], [820, 1450], [180, 1330]]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAVEBILL_DATA", str(tmp_path))
    from app import config
    importlib.reload(config)
    from app import db, export, main
    for mod in (db, export, main):
        importlib.reload(mod)
    with TestClient(main.app) as c:
        yield c


def _upload(client):
    ok, buf = cv2.imencode(".jpg", photo(TRUTH))
    r = client.post("/api/scans", files={"file": ("r.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    return r.json()


def test_full_flow(client):
    scan = _upload(client)
    assert scan["detected"]
    r = client.post(f"/api/scans/{scan['scan_id']}/crop", json={"corners": scan["corners"]})
    assert r.status_code == 200, r.text
    assert "suggestions" in r.json()
    assert isinstance(r.json()["suggestions"]["amount_alternatives"], list)
    assert client.get(r.json()["crop_url"]).status_code == 200

    r = client.post("/api/receipts", json={
        "scan_id": scan["scan_id"], "title": "Rewe Müller", "date": "2026-11-13",
        "amount": 12.52, "category": "Lebensmittel", "note": "Wocheneinkauf",
    })
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["amount"] == 12.52
    assert rec["filename"] == "2026-11-13_Rewe-Mueller_12-52"

    # list, search, filter
    assert client.get("/api/receipts").json()["count"] == 1
    assert client.get("/api/receipts", params={"q": "müller"}).json()["count"] == 1
    assert client.get("/api/receipts", params={"q": "aldi"}).json()["count"] == 0
    assert client.get("/api/receipts", params={"from": "2026-12-01"}).json()["count"] == 0
    lst = client.get("/api/receipts", params={"category": "Lebensmittel"}).json()
    assert lst["total"] == 12.52
    assert client.get("/api/categories").json() == ["Lebensmittel"]

    # files
    assert client.get(rec["thumb_url"]).status_code == 200
    r = client.get(rec["image_url"] + "&download=1")
    assert "attachment" in r.headers["content-disposition"]
    r = client.get(f"/api/receipts/{rec['id']}/pdf")
    assert r.content.startswith(b"%PDF")

    # patch + recrop
    r = client.patch(f"/api/receipts/{rec['id']}", json={"amount": None, "title": "Edeka"})
    assert r.json()["amount"] is None and r.json()["title"] == "Edeka"
    r = client.post(f"/api/receipts/{rec['id']}/recrop", json={"corners": TRUTH})
    assert r.status_code == 200
    assert client.get(f"/api/receipts/{rec['id']}").json()["corners"] == TRUTH

    # export
    r = client.get("/api/export.zip")
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "quittungen.csv" in names and any(n.endswith(".jpg") for n in names)
    assert client.get("/api/export.pdf").content.startswith(b"%PDF")

    # delete
    assert client.delete(f"/api/receipts/{rec['id']}").json()["ok"]
    assert client.get("/api/receipts").json()["count"] == 0
    assert client.get("/api/export.zip").status_code == 404


def test_bad_ids_and_inputs(client):
    assert client.get("/api/receipts/../../etc").status_code == 404
    assert client.get("/api/scans/notanid/image").status_code == 404
    r = client.post("/api/scans", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    scan = _upload(client)
    r = client.post("/api/receipts", json={"scan_id": scan["scan_id"], "title": "x", "date": "2026-01-01"})
    assert r.status_code == 400  # not cropped yet
    r = client.post(f"/api/scans/{scan['scan_id']}/crop", json={"corners": [[0, 0]]})
    assert r.status_code == 422
    client.post(f"/api/scans/{scan['scan_id']}/crop", json={"corners": scan["corners"]})
    r = client.post("/api/receipts", json={"scan_id": scan["scan_id"], "title": "x", "date": "13.11.2026"})
    assert r.status_code == 422
