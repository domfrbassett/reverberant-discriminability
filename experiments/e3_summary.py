"""E3 figure from results/e3_measured.npz."""
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
ROOM_COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baa7a", "D": "#8a5cd1"}
ROOM_INFO = {"A": "A, office, 0.32 s", "B": "B, classroom, 0.47 s", "C": "C, lecture theatre, 0.68 s",
             "D": "D, seminar room, 0.89 s"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def thr(J):
    return np.degrees(1 / np.sqrt(J))


def main():
    d = np.load(ROOT / "results" / "e3_measured.npz")
    th = np.degrees(d["theta"])
    sel = np.abs(th) <= 80.01
    fig = plt.figure(figsize=(7.16, 4.6))
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)

    for i, n in enumerate("ABCD"):
        ax = fig.add_subplot(gs[0, i])
        for k in COL:
            ax.plot(th[sel], thr(d[f"{n}_{k}_J"])[sel], color=COL[k], lw=1.4 if k == "measured" else 1.0,
                    ls={"diffuse": "--", "early_diffuse": ":"}.get(k, "-"), label=LAB[k])
        ax.set_yscale("log")
        ax.set_ylim(1, 30)
        ax.set_yticks([1, 2, 5, 10, 20])
        ax.set_yticklabels(["1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xticks([-60, 0, 60])
        ax.set_xlabel("Azimuth (deg)")
        ax.set_title(ROOM_INFO[n], fontsize=7, color=INK)
        style(ax)
        if i == 0:
            ax.set_ylabel("Azimuth threshold (deg)")
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))

    ax = fig.add_subplot(gs[1, 0])
    fcc = d["fcc"]
    for n in "ABCD":
        ax.plot(fcc, d[f"{n}_gamma_late_avg_band"], color=ROOM_COL[n], lw=0.8, label=f"Room {n}")
    ax.plot(fcc, np.median(d["sonicom_diffuse_coherence"], 0), color=INK2, lw=1.4,
            label="SONICOM diffuse field (e)")
    ax.set_xscale("log")
    ax.set_xlim(80, 4000)
    ax.set_xticks([100, 300, 1000, 3000])
    ax.set_xticklabels(["0.1", "0.3", "1", "3"])
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Late-field coherence")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    room_h, room_l = ax.get_legend_handles_labels()
    style(ax)

    ax = fig.add_subplot(gs[1, 1])
    for n in "ABCD":
        m = thr(d[f"{n}_measured_J"])[sel]
        for k, mk in (("diffuse", "o"), ("early_diffuse", "s")):
            ax.plot(m, thr(d[f"{n}_{k}_J"])[sel], mk, ms=2.5, mfc="none" if k == "diffuse" else COL[k],
                    mec=COL[k], mew=0.6, ls="none", label={"diffuse": "direct + diffuse", "early_diffuse": "80 ms + diffuse"}[k] if n == "A" else None)
    ax.plot([1, 30], [1, 30], color=INK2, lw=0.6)
    for a in (ax.set_xscale, ax.set_yscale):
        a("log")
    ax.set_xlim(2, 25)
    ax.set_ylim(2, 25)
    ax.set_xticks([2, 5, 10, 20])
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.set_yticks([2, 5, 10, 20])
    ax.set_yticklabels(["2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("Measured (deg)")
    ax.set_ylabel("Model (deg)")
    ax.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    style(ax)

    ax = fig.add_subplot(gs[1, 2])
    for n in "ABCD":
        ax.plot(th[sel], thr(d[f"{n}_measured_w_J"])[sel] / thr(d[f"{n}_measured_J"])[sel],
                color=ROOM_COL[n], lw=0.9, label=n)
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_xticks([-60, 0, 60])
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Weighted / stationary")
    ax.set_ylim(0.7, 1.2)
    style(ax)

    ax = fig.add_subplot(gs[1, 3])
    chk = d["check"]
    mk = ["o", "s", "^", "D"]
    for j, lab in enumerate(["∞", "10 dB", "5 dB", "0 dB"]):
        ax.plot(chk[:, j, 0].ravel(), chk[:, j, 1].ravel(), mk[j], ms=1.8, mfc="none", mec=INK2, mew=0.4,
                ls="none", alpha=0.7, label=lab)
    ax.plot([1, 30], [1, 30], color=INK, lw=0.6)
    for a in (ax.set_xscale, ax.set_yscale):
        a("log")
    ax.set_xlim(1, 30)
    ax.set_ylim(1, 30)
    ax.set_xticks([1, 3, 10, 30])
    ax.set_xticklabels(["1", "3", "10", "30"])
    ax.set_yticks([1, 3, 10, 30])
    ax.set_yticklabels(["1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("SH metric (deg)")
    ax.set_ylabel("1-D estimate (deg)")
    ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=1, handletextpad=0.1, labelspacing=0.3,
              borderaxespad=0.1)
    ax.set_title("SONICOM, by DRR", fontsize=7, color=INK)
    style(ax)

    fig.legend(room_h, room_l, frameon=False, fontsize=7, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.02))
    for ax, letter in zip(fig.axes, "abcdefgh"):
        ax.text(-0.3, 1.08, f"({letter})", transform=ax.transAxes, fontsize=8, color=INK)
    fig.savefig(ROOT / "figures" / "e3_measured.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "e3_measured.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
