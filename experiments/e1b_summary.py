"""Summary table of E1b -> results/e1b_shoebox/summary.json, figures/e1b_shoebox.pdf"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sdr import fisher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e1b_shoebox"
INK, INK2, GRID = "#1f1f1e", "#5e5d59", "#e4e3df"
COL = {"stationary": "#eb6834", "weighted": "#2a78d6", "diffuse": "#5e5d59"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
DIRS = {"front": (0, 0, "az"), "left 90": (90, 0, "az"), "right 90": (-90, 0, "az"),
        "elevation front": (0, 0, "el")}
TITLES = {"front": "Azimuth, front", "left 90": "Azimuth, left 90°\n(room side)",
          "right 90": "Azimuth, right 90°\n(wall side)", "elevation front": "Elevation, front"}


def index(r, az, el):
    a = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
    e = np.degrees(np.arcsin(r[:, 2]))
    return int(np.argmin(np.abs(((a - az + 180) % 360) - 180) + np.abs(e - el)))


def main():
    files = sorted(RES.glob("subject_*.npz"))
    d0 = np.load(files[0])
    r, labels = d0["r"], [str(x) for x in d0["labels"]]
    G = np.stack([np.load(f)["G"] for f in files])          # (S, C, D, 2, 2)
    drr = np.load(files[0])["drr"]
    gam = np.stack([np.load(f)["gamma"] for f in files])
    e_az, e_el = fisher.azimuth_vector(r), fisher.elevation_vector(r)
    out = {"subjects": len(files), "configs": {}}
    for ci, lab in enumerate(labels):
        row = {"drr_db": round(float(drr[ci]), 1)}
        for name, (az, el, kind) in DIRS.items():
            i = index(r, az, el)
            e = (e_az if kind == "az" else e_el)[i:i + 1]
            info = np.array([fisher.directional_information(G[s, ci, i:i + 1], r[i:i + 1], e)[0]
                             for s in range(len(files))])
            row[name + "_deg"] = round(float(np.median(np.degrees(1 / np.sqrt(info)))), 2)
        row["gamma_front"] = round(float(np.median(gam[:, ci, index(r, 0, 0)])), 3)
        out["configs"][lab] = row
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    for lab, row in out["configs"].items():
        print(f"{lab:28s}", row)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.2), sharey=False, gridspec_kw=dict(wspace=0.45))
    positions = ["centre", "side", "corner"]
    for ax, name in zip(axes, DIRS):
        for k, alpha in enumerate((0.1, 0.3)):
            for j, kind in enumerate(("stationary", "weighted")):
                y = [out["configs"][f"a{alpha}_{p}_{kind}"][name + "_deg"] for p in positions]
                x = np.arange(3) + (k - 0.5) * 0.36 + (j - 0.5) * 0.14
                ax.plot(x, y, "o" if alpha == 0.1 else "s", color=COL[kind], ms=3.5, ls="none",
                        label=f"{kind}, α={alpha}")
            yd = out["configs"][f"a{alpha}_diffuse"][name + "_deg"]
            ax.axhline(yd, color=COL["diffuse"], lw=0.7, ls=":" if alpha == 0.1 else "--")
        ax.set_xticks(range(3))
        ax.set_xticklabels(positions)
        ax.set_title(TITLES[name], fontsize=7, color=INK)
        ax.grid(True, axis="y", color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Threshold (deg)")
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color=COL["diffuse"], lw=0.7, ls=":"), Line2D([], [], color=COL["diffuse"], lw=0.7, ls="--")]
    l += ["diffuse only, α=0.1", "diffuse only, α=0.3"]
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))
    fig.savefig(ROOT / "figures" / "e1b_shoebox.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "e1b_shoebox.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
