"""Render realistic German till receipts (Lidl/Aldi/Rewe/Edeka/Kaufland/dm/petrol/restaurant
styles) with a known total, photographed-looking on real backgrounds.

Usage:  python tools/make_german_bons.py <background-dir> <out-dir> [--n 60] [--seed 3]
Writes <out>/img/*.jpg and <out>/manifest.json ([{"name", "total", "style"}]).
Fonts: any monospace TTF (Consolas / Courier New / Lucida Console / DejaVu Sans Mono).
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/courbd.ttf", "C:/Windows/Fonts/lucon.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]

ITEMS = [
    "Bananen", "Vollmilch 3,5%", "H-Milch 1,5%", "Bio Eier 10er", "Butter", "Gouda jung", "Toastbrot",
    "Mineralwasser still", "Cola Zero 1,5L", "Apfelschorle", "Nudeln Penne", "Tomaten passiert", "Paprika rot",
    "Gurke", "Joghurt Natur", "Kaffee Crema", "Schokolade Vollmilch", "Chips Paprika", "Zahnpasta",
    "Duschgel", "Toilettenpapier", "Kuechenrolle", "Spuelmittel", "Haferflocken", "Muesli", "Reis Basmati",
    "Haehnchenbrust", "Hackfleisch gem.", "Salami", "Kochschinken", "Brezel", "Croissant", "Orangensaft",
    "Bier Pils 0,5", "Rotwein trocken", "Katzenfutter", "Batterien AA", "Waschmittel", "Kartoffeln 2kg",
    "Aepfel Elstar", "Zwiebeln", "Knoblauch", "Mozzarella", "Frischkaese", "Sahne 30%", "Quark",
]
STORES = {
    "lidl": ("LIDL", "Hauptstrasse 12", "12345 Musterstadt"),
    "aldi": ("ALDI SUED", "Bahnhofstr. 5", "80331 Muenchen"),
    "rewe": ("REWE Markt GmbH", "Marktplatz 3", "50667 Koeln"),
    "edeka": ("EDEKA Mueller e.K.", "Lindenweg 41", "16225 Eberswalde"),
    "kaufland": ("Kaufland", "Am Ring 2", "74172 Neckarsulm"),
    "dm": ("dm-drogerie markt", "Koenigstr. 20", "70173 Stuttgart"),
    "tank": ("ARAL Tankstelle", "B27 Nord", "72070 Tuebingen"),
    "resto": ("Pizzeria Da Mario", "Gartenstr. 7", "60311 Frankfurt"),
}


def eur(v: float) -> str:
    s = f"{abs(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("-" if v < 0 else "") + s


def row(left: str, right: str, width: int) -> str:
    left = left[: max(1, width - len(right) - 1)]
    return left + " " * max(1, width - len(left) - len(right)) + right


def build(style: str, rng: random.Random, width: int = 38) -> tuple[list[str], float]:
    name, street, city = STORES[style]
    L = [name.center(width), street.center(width), city.center(width)]
    if rng.random() < 0.5:
        L.append(f"Tel. 0{rng.randint(200, 999)} {rng.randint(100000, 999999)}".center(width))
    L += ["", "EUR".rjust(width)]
    total = 0.0
    tax = {"A": 0.0, "B": 0.0}

    def add(label, price, code=None, extra=None):
        nonlocal total
        code = code or rng.choice("AB")
        L.append(row(label, f"{eur(price)} {code}", width))
        if extra:
            L.append("  " + extra)
        total += price
        tax[code] += price

    if style == "tank":
        liters = round(rng.uniform(20, 55), 2)
        ppl = round(rng.uniform(1.62, 1.95), 3)
        price = round(liters * ppl, 2)
        L.append(row("Super E10", "", width))
        add(f"{liters:.2f} l x {ppl:.3f} EUR/l".replace(".", ","), price, "B")
        if rng.random() < 0.5:
            add("Kaffee to go", 2.49, "B")
    else:
        for _ in range(rng.randint(2, 12)):
            item = rng.choice(ITEMS)
            r = rng.random()
            if r < 0.15:
                q, unit = rng.randint(2, 6), round(rng.uniform(0.19, 3.99), 2)
                add(item, round(q * unit, 2), extra=f"{q} x {eur(unit)}")
            elif r < 0.25:
                kg, perkg = round(rng.uniform(0.2, 1.8), 3), round(rng.uniform(1.99, 14.99), 2)
                add(item, round(kg * perkg, 2), "A",
                    extra=f"{kg:.3f} kg x {eur(perkg)} EUR/kg".replace(".", ","))
            else:
                add(item, round(rng.uniform(0.29, 12.99), 2))
            if rng.random() < 0.12:
                add("Pfand 0,25", 0.25, "B")
        if rng.random() < 0.3:
            d = -round(rng.uniform(0.3, 2.0), 2)
            add(rng.choice(["Preisvorteil", "Rabatt 20%", "Coupon"]), d, "A")
        if rng.random() < 0.2:
            add("Pfandrueckgabe", -round(rng.choice([0.25, 0.75, 1.5, 3.1]), 2), "B")
    total = round(total, 2)

    paid_card = rng.random() < 0.45
    given = None
    if not paid_card:
        given = next(n for n in (5, 10, 20, 50, 100, 200) if n >= total) + (0 if rng.random() < 0.7 else 0.5)
        if rng.random() < 0.1:
            given = total  # exact cash
    sep = "-" * width
    if style == "lidl":
        L += [" " * (width - 10) + "-" * 10, row("zu zahlen", eur(total), width)]
        L.append(row("Kreditkarte" if paid_card else "Bar", eur(total if paid_card else given), width))
        if given and given > total:
            L.append(row("Rueckgeld", eur(given - total), width))
    elif style == "aldi":
        L += [sep, row("Summe", eur(total), width)]
        L.append(row("Kartenzahlung" if paid_card else "Bar", eur(total if paid_card else given), width))
        if given and given > total:
            L.append(row("Rueckgeld", eur(given - total), width))
    elif style in ("rewe", "edeka", "dm"):
        L += [sep, row("SUMME", f"EUR {eur(total)}", width)]
        if style == "edeka":
            L.append(f"Posten: {len([l for l in L if l.endswith((' A', ' B'))])}")
        if paid_card:
            L.append(row("Geg. EC-Cash" if style == "rewe" else "EC-Karte", f"EUR {eur(total)}", width))
        else:
            L.append(row("Geg. BAR", f"EUR {eur(given)}", width))
            if given > total:
                L.append(row("Rueckgeld BAR", f"EUR {eur(given - total)}", width))
    elif style == "kaufland":
        L += [sep, row("Summe", eur(total), width)]
        L.append(row("Girocard" if paid_card else "Bar", eur(total if paid_card else given), width))
        if given and given > total:
            L.append(row("Rueckgeld", eur(given - total), width))
    elif style == "tank":
        L += [sep, row("Gesamtbetrag EUR", eur(total), width)]
        L.append(row("girocard" if paid_card else "BAR", eur(total if paid_card else given), width))
        if given and given > total:
            L.append(row("Rueckgeld", eur(given - total), width))
    elif style == "resto":
        L += [sep, row("Summe", eur(total), width)]
        if rng.random() < 0.5:
            tip = round(rng.uniform(1, 5), 2)
            L.append(row("Trinkgeld", eur(tip), width))
        L.append(row("Bar" if not paid_card else "VISA", eur(given if given else total), width))
        if given and given > total:
            L.append(row("Rueckgeld", eur(given - total), width))

    # VAT table repeating amounts (the classic trap)
    a, b = round(tax["A"], 2), round(tax["B"], 2)
    na, nb = round(a / 1.07, 2), round(b / 1.19, 2)
    L += ["", row("MwSt%   MwSt    Netto", "Brutto", width)]
    L.append(row(f"A 7%   {eur(a - na):>5} {eur(na):>8}", eur(a), width))
    L.append(row(f"B 19%  {eur(b - nb):>5} {eur(nb):>8}", eur(b), width))
    L.append(row(rng.choice(["Summe", "Gesamtbetrag", "Gesamt"]) + f"  {eur(a + b - na - nb):>5} {eur(na + nb):>8}",
                 eur(total), width))
    if rng.random() < 0.4:
        L += ["", f"Sie haben {eur(rng.uniform(0.5, 4))} EUR gespart!".center(width)]
    if paid_card and rng.random() < 0.7:
        L += ["", "*** Kundenbeleg ***".center(width), "Kartenzahlung girocard", "Terminal-ID 6012" + str(rng.randint(1000, 9999)),
              row("Betrag EUR", eur(total), width), "Zahlung erfolgt"]
    d = f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.choice(['24', '25', '2025'])}"
    L += ["", f"{d}  {rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}  Bon {rng.randint(1000, 9999)}",
          "Vielen Dank fuer Ihren Einkauf!".center(width)]
    return L, total


def render(lines, rng, font_path):
    size = rng.randint(20, 26)
    font = ImageFont.truetype(font_path, size)
    cw = font.getbbox("M")[2]
    lh = int(size * rng.uniform(1.2, 1.45))
    W = int(cw * 38) + 40
    H = lh * len(lines) + 60
    paper = rng.randint(225, 250)
    im = Image.new("L", (W, H), paper)
    dr = ImageDraw.Draw(im)
    ink = rng.randint(10, 90)  # faded thermal print up to 90
    for i, l in enumerate(lines):
        dr.text((20, 30 + i * lh), l, fill=ink, font=font)
    return np.array(im)


def photograph(paper, bg, rng):
    ph, pw = paper.shape
    W, H = 1500, 2000
    bh, bw = bg.shape[:2]
    s = max(W / bw, H / bh)
    canvas = cv2.resize(bg, (int(bw * s) + 1, int(bh * s) + 1))[:H, :W].astype(np.float32)
    frac = rng.uniform(0.55, 0.9)  # receipt height relative to photo
    sc = frac * H / ph
    rw, rh = pw * sc, ph * sc
    if rw > 0.9 * W:
        sc *= 0.9 * W / rw
        rw, rh = pw * sc, ph * sc
    cx, cy = W / 2 + rng.uniform(-0.1, 0.1) * W, H / 2 + rng.uniform(-0.05, 0.05) * H
    ang = np.deg2rad(rng.uniform(-6, 6))
    rect = np.array([[-rw / 2, -rh / 2], [rw / 2, -rh / 2], [rw / 2, rh / 2], [-rw / 2, rh / 2]], np.float32)
    rect += np.array([[rng.uniform(-0.015, 0.015) * rw, rng.uniform(-0.008, 0.008) * rh] for _ in range(4)], np.float32)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    quad = rect @ rot.T + np.array([cx, cy], np.float32)
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), quad)
    rgb = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
    tint = np.array([rng.uniform(0.93, 1.0), rng.uniform(0.95, 1.0), 1.0], np.float32)  # slightly warm/cool paper
    warped = cv2.warpPerspective(rgb, m, (W, H), flags=cv2.INTER_AREA).astype(np.float32) * tint
    mask = cv2.GaussianBlur(cv2.warpPerspective(np.ones((ph, pw), np.float32), m, (W, H)), (3, 3), 0)[:, :, None]
    out = canvas * (1 - mask) + warped * mask
    light = cv2.resize(np.random.default_rng(rng.randint(0, 10**6)).uniform(0.75, 1.08, (3, 3)).astype(np.float32),
                       (W, H), interpolation=cv2.INTER_CUBIC)[:, :, None]
    out = out * light
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.4))
    out += np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(2, 7), out.shape)
    ok, buf = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 92)])
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backgrounds")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fonts = [f for f in FONT_CANDIDATES if Path(f).exists()]
    bgs = [b for b in (cv2.imread(str(p)) for p in sorted(Path(args.backgrounds).glob("*.jpg"))) if b is not None]
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    manifest = []
    styles = list(STORES)
    for i in range(args.n):
        style = styles[i % len(styles)]
        lines, total = build(style, rng)
        data = photograph(render(lines, rng, rng.choice(fonts)), rng.choice(bgs), rng)
        name = f"{style}_{i:03d}.jpg"
        (out / "img" / name).write_bytes(data)
        manifest.append({"name": name, "total": total, "style": style})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{args.n} receipts -> {out} (fonts: {len(fonts)})")


if __name__ == "__main__":
    main()
