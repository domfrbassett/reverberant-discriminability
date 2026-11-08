"""Figures and summary of the E1 pilot -> results/e1_drr/summary.json, figures/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

from sdr import fisher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e1_drr"
FIG = ROOT / "figures"
INK, INK2, GRID = "#1f1f1e", "#5e5d59", "#e4e3df"
BLUES = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
FAMILY = [("Spectral", "#2a78d6"), ("ILD", "#eb6834"), ("ITD", "#1baf7a")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def load():
    files = sorted(RES.glob("subject_*.npz"))
    d = [np.load(f) for f in files]
    return d[0]["r"], d[0]["drr_db"], d[0]["rho"], d


def index(r, az, el):
    a = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
    e = np.degrees(np.arcsin(r[:, 2]))
    return int(np.argmin(np.abs(((a - az + 180) % 360) - 180) + np.abs(e - el)))


def thresholds(G, r, e):
    """Threshold (deg) at d' = 1 along unit tangent e; G (..., D, 2, 2)."""
    info = np.stack([fisher.directional_information(g, r, e) for g in G.reshape(-1, *G.shape[-3:])])
    return np.degrees(1 / np.sqrt(info)).reshape(G.shape[:-2])


def log_ticks(ax, ticks):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels([f"{t:g}" for t in ticks])


def ax_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5)


def main():
    r, drr, rho, data = load()
    nr, nd = len(rho), len(drr)
    G = np.stack([x["G"].reshape(nr, nd, *x["G"].shape[1:]) for x in data])      # (S, rho, drr, D, 2, 2)
    fam = np.stack([x["fam_info"].reshape(nr, nd, *x["fam_info"].shape[1:]) for x in data])
    itd = np.stack([x["itd_s"].reshape(nr, nd, -1) for x in data])
    gam = np.stack([x["gamma"].reshape(nr, nd, -1) for x in data])
    Gv = np.stack([x["Gv"].reshape(nr, nd, *x["Gv"].shape[1:]) for x in data])
    S = len(data)
    xs = np.where(np.isinf(drr), 30.0, drr)
    e_az, e_el = fisher.azimuth_vector(r), fisher.elevation_vector(r)

    az_dirs = [0, 30, 60, 90]
    idx_az = [index(r, a, 0) for a in az_dirs]
    maa = thresholds(G[..., idx_az, :, :], r[idx_az], e_az[idx_az])       # (S, rho, drr, 4)
    idx_el = [index(r, 0, 0), index(r, 0, 30)]
    elt = thresholds(G[..., idx_el, :, :], r[idx_el], e_el[idx_el])

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.45))
    main_rho = 0
    for j, (a, c) in enumerate(zip(az_dirs, BLUES)):
        m = np.median(maa[:, main_rho, :, j], 0)
        lo, hi = np.median(maa[:, 1, :, j], 0), np.median(maa[:, 2, :, j], 0)
        axes[0].fill_between(xs, np.minimum(lo, hi), np.maximum(lo, hi), color=c, alpha=0.18, lw=0)
        axes[0].plot(xs, m, "-o", color=c, ms=3, lw=1.4, label=f"{a}°")
    log_ticks(axes[0], [1, 2, 5, 10, 20, 50])
    axes[0].set_ylabel("Azimuth threshold (deg)")
    axes[0].legend(title="Azimuth", frameon=False, fontsize=7, title_fontsize=7)
    for j, (lab, c) in enumerate(zip(["0° elevation", "30° elevation"], [BLUES[1], BLUES[3]])):
        m = np.median(elt[:, main_rho, :, j], 0)
        lo, hi = np.median(elt[:, 1, :, j], 0), np.median(elt[:, 2, :, j], 0)
        axes[1].fill_between(xs, np.minimum(lo, hi), np.maximum(lo, hi), color=c, alpha=0.18, lw=0)
        axes[1].plot(xs, m, "-o", color=c, ms=3, lw=1.4, label=lab)
    log_ticks(axes[1], [2, 5, 10, 20, 50])
    axes[1].set_ylabel("Elevation threshold (deg)")
    axes[1].legend(frameon=False, fontsize=7, title="Median plane", title_fontsize=7)
    k = idx_az[0]
    share = np.trace(fam[:, main_rho, :, k], axis1=-2, axis2=-1)          # (S, drr, 3)
    share = np.median(share / share.sum(-1, keepdims=True), 0)
    for j, (lab, c) in enumerate(FAMILY):
        axes[2].plot(xs, 100 * share[:, j], "-o", color=c, ms=3, lw=1.4, label=lab)
    axes[2].set_ylabel("Share of information, front (%)")
    axes[2].set_ylim(0, 85)
    axes[2].legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.56, 1.02), handlelength=1.2,
                   columnspacing=0.8)
    for ax in axes:
        ax_style(ax)
        shown = [v for v in drr if np.isinf(v) or v % 10 == 0]
        ax.set_xticks([30.0 if np.isinf(v) else v for v in shown])
        ax.set_xticklabels(["∞" if np.isinf(v) else f"{v:g}" for v in shown])
        ax.set_xlabel("DRR (dB)")
        ax.invert_xaxis()
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.22, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8)
    fig.savefig(FIG / "e1_drr.pdf", bbox_inches="tight")
    fig.savefig(FIG / "e1_drr.png", dpi=200, bbox_inches="tight")

    sd = np.sqrt(np.linalg.det(G[:, main_rho]))                         # (S, drr, D)
    rel = np.median(sd / sd[:, :1], 0)                                   # (drr, D)
    summary = {
        "subjects": S, "drr_db": [None if np.isinf(v) else float(v) for v in drr], "rho": rho.tolist(),
        "maa_deg_median": {f"az{a}": {f"rho{p}": np.median(maa[:, i, :, j], 0).round(2).tolist()
                                      for i, p in enumerate(rho)} for j, a in enumerate(az_dirs)},
        "elev_threshold_deg_median": {lab: {f"rho{p}": np.median(elt[:, i, :, j], 0).round(2).tolist()
                                            for i, p in enumerate(rho)}
                                      for j, lab in enumerate(["el0", "el30"])},
        "front_info_share": dict(zip([n for n, _ in FAMILY], share.T.round(3).tolist())),
        "itd_us_median": {f"az{a}": (np.median(itd[:, main_rho, :, i], 0) * 1e6).round(1).tolist()
                          for a, i in zip(az_dirs, idx_az)},
        "gamma_itd_median": {f"az{a}": np.median(gam[:, main_rho, :, i], 0).round(3).tolist()
                             for a, i in zip(az_dirs, idx_az)},
        "sqrtdet_rel_median_over_dirs": np.median(rel, 1).round(3).tolist(),
        "variance_term_share_front": np.median(
            np.trace(Gv[:, main_rho, :, k], axis1=-2, axis2=-1)
            / np.trace(G[:, main_rho, :, k], axis1=-2, axis2=-1), 0).round(3).tolist(),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
