"""Digitise Fig. 3 of Litovsky & Shinn-Cunningham (2001): ITD jnds of lead, lag and single burst.

The figure is a 1-bit scan (page 5 image 0). Axes are calibrated from the tick-label centroids;
lead markers are filled circles (solid after a 5x5 erosion), lag markers open diamonds (template
match in each delay column), single-burst controls open squares.
Output: data/litovsky_shinn_2001_fig3.csv, figures/e2_overlay.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import correlate2d

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "2001jasa_litovsky.pdf"
SUBJECTS = {"S1": (0, 0), "S2": (1, 0), "S3": (2, 0), "S4": (0, 1), "S5": (1, 1), "S6": (2, 1)}
YLAB = {0: [30.5, 90.7, 151.7, 212.15, 273.2], 1: [358.7, 419.1, 480.2, 540.05, 599.6],
        2: [676.0, 737.2, 798.0, 858.15, 918.9]}
XLAB = {0: [185.5, 287.7, 390.25, 491.4], 1: [616.8, 717.9, 820.45, 923.4]}
YB = {0: (0, 280), 1: (300, 605), 2: (625, 930)}
XB = {0: (140, 520), 1: (570, 943)}
DELAYS = [1, 2, 3, 5, 10, 15]
LEGEND = (slice(285, 440), slice(328, 512))


def figure_image() -> np.ndarray:
    doc = pymupdf.open(PDF)
    xref = doc[4].get_images()[0][0]
    pix = pymupdf.Pixmap(doc, xref)
    return np.array(Image.frombytes("L", (pix.width, pix.height), pix.samples)) < 128


def to_value(row, ypix):
    return float(np.polyval(np.polyfit(YLAB[row], [400, 300, 200, 100, 0], 1), ypix))


def to_pixel(row, value):
    return float(np.polyval(np.polyfit([400, 300, 200, 100, 0], YLAB[row], 1), value))


def xpix(col, d):
    return XLAB[col][0] + (XLAB[col][-1] - XLAB[col][0]) / 15 * d


def main():
    im = figure_image()
    im[LEGEND] = False
    solid = ndimage.binary_erosion(im, structure=np.ones((5, 5)))
    lab, n = ndimage.label(solid)
    circles = [np.mean(np.where(lab == k), axis=1) for k in range(1, n + 1) if (lab == k).sum() > 20]
    imf = im.astype(float)
    cy, cx = int(round(to_pixel(0, 325.3))), int(round(xpix(0, 1)))
    T = imf[cy - 9:cy + 10, cx - 9:cx + 10]
    Tz = T - T.mean()
    filled = ndimage.binary_fill_holes(im)
    rows = []
    for s, (ri, ci) in SUBJECTS.items():
        ya, yb = YB[ri]
        for d in DELAYS:
            xc = xpix(ci, d)
            c = [p for p in circles if ya < p[0] < yb and abs(p[1] - xc) < 9]
            rows.append(dict(subject=s, condition="lead", delay_ms=d,
                             jnd_us=round(to_value(ri, c[0][0]), 1), flag="",
                             px=float(c[0][1]), py=float(c[0][0])))
            xa = int(xc) - 14
            W = imf[ya:yb, xa:int(xc) + 15]
            cc = correlate2d(W - W.mean(), Tz, mode="same")
            cc[:, :5] = cc[:, -5:] = -1e9
            y, x = np.unravel_index(np.argmax(cc), cc.shape)
            score = cc.max() / (Tz ** 2).sum()
            yl, flag = y + ya, ""
            if abs(yl - c[0][0]) < 6:
                # diamond over a lead circle: centre is 9 px below the diamond's top vertex
                xm = int(round(c[0][1]))
                strip = im[int(c[0][0]) - 40:int(c[0][0]), xm - 10:xm + 11].sum(1)
                top = int(c[0][0]) - 40 + int(np.argmax(strip >= 4))
                yl, flag = top + 9, "diamond over lead marker; centre from top vertex"
            rows.append(dict(subject=s, condition="lag", delay_ms=d,
                             jnd_us=round(to_value(ri, yl), 1), flag=flag,
                             px=float(xa + x), py=float(yl)))
        # single-burst control: square left of the axis break
        xs = xpix(ci, -1.35)
        sq = ndimage.binary_erosion(filled, structure=np.ones((5, 5)))
        sl, sn = ndimage.label(sq[ya:yb, int(xs) - 12:int(xs) + 12])
        cen = [np.mean(np.where(sl == k), axis=1) for k in range(1, sn + 1) if (sl == k).sum() > 20]
        rows.append(dict(subject=s, condition="single", delay_ms="", jnd_us=round(to_value(ri, cen[0][0] + ya), 1),
                         flag="", px=float(cen[0][1] + int(xs) - 12), py=float(cen[0][0] + ya)))
    keys = ["subject", "condition", "delay_ms", "jnd_us", "flag"]
    with open(ROOT / "data" / "litovsky_shinn_2001_fig3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in rows])

    gray = Image.fromarray(((~figure_image()) * 255).astype(np.uint8)).convert("RGB")
    scale = 2
    rgb = gray.resize((gray.width * scale, gray.height * scale), Image.NEAREST)
    dr = ImageDraw.Draw(rgb)
    colours = {"lead": (220, 30, 30), "lag": (30, 90, 220), "single": (20, 150, 60)}
    for r in rows:
        x, y = r["px"] * scale, r["py"] * scale
        col = colours[r["condition"]]
        dr.line([x - 14, y, x + 14, y], fill=col, width=3)
        dr.line([x, y - 14, x, y + 14], fill=col, width=3)
        dr.text((x + 10, y - 26), f"{r['jnd_us']:.0f}", fill=col)
    rgb.save(ROOT / "figures" / "e2_overlay.png")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
