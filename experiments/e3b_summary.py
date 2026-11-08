"""E3b figure from results/e3b_cr4.npz."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID = "#1f1f1e", "#5e5d59", "#e4e3df"
COL = {"direct": "#a8a7a1", "measured": "#1f1f1e", "diffuse": "#eb6834", "early_diffuse": "#2a78d6"}
LAB = {"direct": "direct part", "measured": "measured", "diffuse": "direct + diffuse",
       "early_diffuse": "first 80 ms + diffuse"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    d = np.load(ROOT / "results" / "e3b_cr4.npz")
    ids = d["emitter_id"]
    az0 = d["az0_deg"]
    fig = plt.figure(figsize=(7.16, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.45)
    for j, ls in enumerate((7, 3)):
        e = int(np.where(ids == ls)[0][0])
        th = d[f"LS{ls}_theta_deg"]
        sel = np.abs(th - th.mean()) <= 24.01
        ax = fig.add_subplot(gs[0, j])
        for k in COL:
            ax.plot(th[sel], d[f"LS{ls}_{k}_thr"][sel], color=COL[k], lw=1.4 if k == "measured" else 1.0,
                    ls={"diffuse": "--", "early_diffuse": ":"}.get(k, "-"), label=LAB[k])
        ax.set_yscale("log")
        ax.set_ylim(1, 20)
        ax.set_yticks([1, 2, 5, 10, 20])
        ax.set_yticklabels(["1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlabel("Head azimuth of source (deg)")
        ax.set_title(f"({'ab'[j]}) LS{ls}, {az0[e]:+.1f}° at 0° head turn", fontsize=7.5, color=INK, loc="left")
        style(ax)
        if j == 0:
            ax.set_ylabel("Azimuth threshold (deg)")
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))

    ax = fig.add_subplot(gs[0, 2])
    for ls in ids:
        th = d[f"LS{ls}_theta_deg"]
        sel = np.abs(th - th.mean()) <= 24.01
        m = d[f"LS{ls}_measured_thr"][sel]
        for k, mk in (("diffuse", "o"), ("early_diffuse", "s")):
            ax.plot(m, d[f"LS{ls}_{k}_thr"][sel], mk, ms=2.5, mfc="none" if k == "diffuse" else COL[k], mec=COL[k],
                    mew=0.6, ls="none", label={"diffuse": "direct + diffuse", "early_diffuse": "80 ms + diffuse"}[k]
                    if ls == ids[0] else None)
    ax.plot([1, 30], [1, 30], color=INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4, 15)
    ax.set_ylim(4, 15)
    for setter in (ax.set_xticks, ax.set_yticks):
        setter([5, 10])
    ax.set_xticklabels(["5", "10"])
    ax.set_yticklabels(["5", "10"])
    ax.minorticks_off()
    ax.set_xlabel("Measured (deg)")
    ax.set_ylabel("Model (deg)")
    ax.set_title("(c) LS3–LS7, |head turn| ≤ 24°", fontsize=7.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    style(ax)
    fig.savefig(ROOT / "figures" / "e3b_cr4.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "e3b_cr4.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
