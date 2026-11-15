"""Download real background photos (tables, hands, fabrics, ...) from Wikimedia Commons.

They are the surfaces that tools/make_small_set.py and tools/train_segmenter.py place
receipts on. Every search term gets its own file prefix, so training and evaluation can use
different backgrounds (see --split in make_small_set.py).

Usage:  python tools/fetch_backgrounds.py <out-dir> [--per-term 8]
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "SaveBill-dataset-tool/1.0 (receipt detection research)"}
TERMS = [
    "wooden table top", "wood grain texture", "granite countertop", "marble surface texture",
    "tablecloth pattern", "lace tablecloth", "car seat upholstery", "human hand palm",
    "office desk top view", "restaurant table top", "carpet texture", "computer keyboard",
    "concrete floor texture", "leather texture", "bed sheet fabric", "denim fabric texture",
    "white table surface", "kitchen counter", "laminate flooring", "cardboard texture",
    "newspaper page", "cafe table", "steering wheel car interior", "jacket fabric",
    "ceramic tiles floor", "plastic tray", "linoleum floor", "sofa fabric",
]


def get(url: str):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))  # Commons rate limit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-term", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    credits = []
    for term in TERMS:
        q = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"{term} filetype:bitmap", "gsrlimit": args.per_term * 2, "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata", "iiurlwidth": 1600,
        })
        pages = json.loads(get(f"{API}?{q}")).get("query", {}).get("pages", {})
        slug = re.sub(r"[^a-z]+", "_", term.lower()).strip("_")
        n = 0
        for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("mime") != "image/jpeg" or "thumburl" not in info:
                continue
            target = out / f"{slug}_{n:02d}.jpg"
            if not target.exists():
                try:
                    target.write_bytes(get(info["thumburl"]))
                except Exception as e:  # noqa: BLE001
                    print(f"skip {p['title']!a}: {e}")
                    continue
                time.sleep(1.5)
            meta = info.get("extmetadata", {})
            credits.append({"file": target.name, "source": info.get("descriptionurl"),
                            "license": meta.get("LicenseShortName", {}).get("value")})
            n += 1
            if n >= args.per_term:
                break
        print(f"{term}: {n}")
    (out / "credits.json").write_text(json.dumps(credits, indent=1))


if __name__ == "__main__":
    main()
