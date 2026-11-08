"""E6 figure from results/e6_maps: azimuth and elevation threshold over seats, median of heads."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from experiments.e6_maps import ROOMS, SUBJECTS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e6_maps"
INK, INK2 = "#1f1f1e", "#5e5d59"
TITLE = {"lecture": "Lecture theatre", "dining": "Open-plan dining", "concert": "Concert hall"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def grid(P, v):
    xs, ys = np.unique(P[:, 0]), np.unique(P[:, 1])
    Z = np.full((len(ys), len(xs)), np.nan)
    for (x, y, _), val in zip(P, v):
        Z[np.searchsorted(ys, y), np.searchsorted(xs, x)] = val
    return xs, ys, Z


def main():
    fig = plt.figure(figsize=(7.16, 4.0))
    widths = [ROOMS[n]["L"][0] for n in ROOMS]
    gs = fig.add_gridspec(2, 3, width_ratios=widths, hspace=0.5, wspace=0.55)
    rows = [[], []]
    ims = [None, None]
    norms = {"az_deg": LogNorm(3, 40), "el_deg": LogNorm(7, 60)}
    for j, name in enumerate(ROOMS):
        d = [np.load(RES / f"{name}_{s:04d}.npz") for s in SUBJECTS]
        P = d[0]["seats"]
        L, src = ROOMS[name]["L"], ROOMS[name]["source"]
        for i, (key, lab, cmap) in enumerate((("az_deg", "Azimuth threshold (deg)", "Blues"),
                                              ("el_deg", "Elevation threshold (deg)", "Oranges"))):
            v = np.median([x[key] for x in d], 0)
            xs, ys, Z = grid(P, v)
            ax = fig.add_subplot(gs[i, j])
            step = ROOMS[name]["step"]
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#d9d8d3")
            im = ax.pcolormesh(np.append(xs, xs[-1] + step) - step / 2, np.append(ys, ys[-1] + step) - step / 2, Z,
                               cmap=cm, norm=norms[key], shading="flat")
            ax.plot(*src[:2], marker="*", ms=8, color=INK, ls="none")
            ax.set_xlim(0, L[0])
            ax.set_ylim(0, L[1])
            ax.set_aspect("equal")
            ax.set_xticks([0, L[0]])
            ax.set_yticks([0, L[1]])
            ax.set_xticklabels([0, f"{L[0]:g} m"])
            ax.set_yticklabels([0, f"{L[1]:g} m"])
            for sp in ax.spines.values():
                sp.set_color(INK2)
            letter = "abcdef"[i * 3 + j]
            ax.set_title(f"({letter}) {TITLE[name]}" if i == 0 else f"({letter})", fontsize=8, color=INK, loc="left")
            rows[i].append(ax)
            ims[i] = (im, key, lab)
    for i in range(2):
        im, key, lab = ims[i]
        cb = fig.colorbar(im, ax=rows[i], extend="max", fraction=0.02, pad=0.045, aspect=12)
        ticks = [3, 5, 10, 20, 40] if key == "az_deg" else [7, 10, 20, 40, 60]
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])
        cb.minorticks_off()
        cb.set_label(lab, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_edgecolor(INK2)
    fig.savefig(ROOT / "figures" / "e6_maps.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "e6_maps.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
