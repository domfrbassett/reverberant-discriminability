"""Overlay of the digitised Rakerd & Hartmann points on the article figures -> figures/c1_overlay.png"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "JASMAN-000128-003052_1.pdf"
DPI = 300
S = DPI / 72
# page index, plot frame (pt), y of the 100% and 50% ticks (pt), x of two ITD ticks (pt, value), crop (pt)
PANELS = {
    "2": (2, (47.27, 156.23), ((393.41, 0), (509.17, 200)), (345, 25, 535, 185), (-1.155 * 3, 1.155)),
    "3": (3, (48.27, 157.23), None, (95, 25, 262, 185), None),
    "8a": (8, (46.93, 155.81), ((393.38, 0), (509.06, 200)), (345, 25, 535, 175), None),
    "8b": (8, (183.03, 291.9), ((393.38, 0), (509.06, 200)), (345, 165, 535, 312), None),
}


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "rakerd_hartmann_2010.csv")))
    doc = pymupdf.open(PDF)
    tiles = []
    font = ImageFont.load_default()
    for fig, (pno, (y100, y50), xt, crop, jog) in PANELS.items():
        clip = pymupdf.Rect(*crop)
        pix = doc[pno].get_pixmap(dpi=DPI, clip=clip)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        dr = ImageDraw.Draw(img)
        pts = [r for r in rows if r["figure"] == fig]
        for r in pts:
            y_pt = y100 + (100 - float(r["percent"])) / 50 * (y50 - y100)
            if fig == "3":
                g = [200, 300, 400, 500].index(int(r["itd_us"]))
                k = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].index(float(r["coherence"]))
                x_pt = [140.12, 169.06, 198.0, 226.94][g] + [0, 1.58, 2.9, 4.33, 5.77, 7.24, 8.68][k]
            else:
                (xa, va), (xb, vb) = xt
                x_pt = xa + (float(r["itd_us"]) - va) / (vb - va) * (xb - xa)
                if fig == "2":
                    k = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].index(float(r["coherence"]))
                    x_pt += [-3.46, -2.31, -1.15, 0, 1.15, 2.31, 3.46][k] - 0.1
            x, y = (x_pt - crop[0]) * S, (y_pt - crop[1]) * S
            dr.line([x - 9, y, x + 9, y], fill=(220, 30, 30), width=2)
            dr.line([x, y - 9, x, y + 9], fill=(220, 30, 30), width=2)
            dr.text((x + 6, y - 22), r["coherence"], fill=(220, 30, 30), font=font)
        dr.text((10, 10), f"Fig. {fig}", fill=(0, 0, 0), font=font)
        tiles.append(img)
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    out = Image.new("RGB", (2 * w + 20, 2 * h + 20), "white")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % 2) * (w + 20), (i // 2) * (h + 20)))
    out.save(ROOT / "figures" / "c1_overlay.png")


if __name__ == "__main__":
    main()
