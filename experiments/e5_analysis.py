"""E5 analysis: cross-validated variance of acuity loss explained by room parameters.

Models (ordinary least squares on standardised predictors): DRR alone; DRR, C80, LF, IACC and
log T60; the same with all squares and pairwise products; and the last with source-direction
indicators. Folds leave out whole configurations (room, absorption, distance, position).
Output: results/e5_redundancy/analysis.json, figures/e5_redundancy.pdf
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e5_redundancy"
INK, INK2, GRID = "#1f1f1e", "#5e5d59", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
PARAMS = ["drr_db", "c80_db", "lf", "iacc", "log_t60"]


def load():
    rows = []
    for f in sorted(RES.glob("subject_*.csv")):
        rows += list(csv.DictReader(open(f)))
    for r in rows:
        r["log_t60"] = np.log(float(r["t60_s"]))
        r["config"] = f'{r["subject"]}_{r["room"]}_{r["alpha"]}_{r["distance_m"]}_{r["position"]}'
        r["seat"] = f'{r["room"]}_{r["alpha"]}_{r["distance_m"]}_{r["position"]}'
        r["target"] = f'{r["az"]}_{r["el"]}'
    return rows


def design(rows, kind):
    X = np.array([[float(r[p]) for p in PARAMS] for r in rows])
    X = (X - X.mean(0)) / X.std(0)
    if kind == "drr":
        X = X[:, :1]
    elif kind in ("quadratic", "quadratic+direction"):
        X = np.hstack([X, X ** 2] + [X[:, [i]] * X[:, [j]] for i, j in combinations(range(X.shape[1]), 2)])
    if kind == "quadratic+direction":
        tg = sorted({r["target"] for r in rows})
        X = np.hstack([X, np.array([[r["target"] == t for t in tg[1:]] for r in rows], float)])
    return np.hstack([np.ones((len(rows), 1)), X])


def cv_r2(X, y, groups, k=10, seed=0):
    ug = np.array(sorted(set(groups)))
    fold = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug)) % k))
    f = np.array([fold[g] for g in groups])
    pred = np.empty_like(y)
    for i in range(k):
        tr, te = f != i, f == i
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred[te] = X[te] @ beta
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), pred


def main():
    rows = load()
    groups = np.array([r["config"] for r in rows])
    out = {"n_rows": len(rows), "n_configs": len(set(groups)), "subjects": len({r["subject"] for r in rows})}
    for target in ("loss", "loss_az", "loss_el"):
        y = np.array([float(r[target]) for r in rows])
        res = {}
        preds = {}
        for kind in ("drr", "linear", "quadratic", "quadratic+direction"):
            r2, pred = cv_r2(design(rows, kind), y, groups)
            res[kind] = round(float(r2), 3)
            preds[kind] = pred
        seat = np.array([r["config"] for r in rows])
        means = {g: y[seat == g].mean() for g in set(seat)}
        between = np.var([means[g] for g in seat])
        res["between_config_share"] = round(float(between / np.var(y)), 3)
        out[target] = res
        if target == "loss":
            best_pred, y_loss = preds["quadratic"], y
    corr = {p: round(float(np.corrcoef([float(r[p]) for r in rows], [float(r["loss"]) for r in rows])[0, 1]), 3)
            for p in PARAMS}
    out["correlation_with_loss"] = corr
    out["loss_range_db"] = [round(float(np.percentile(y_loss, q)), 2) for q in (5, 50, 95)]
    (RES / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    tg = np.array([r["target"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.35))
    cmap = plt.get_cmap("viridis")
    names = sorted(set(tg), key=lambda s: (int(s.split("_")[1]), int(s.split("_")[0])))
    for i, t in enumerate(names):
        sel = tg == t
        az, el = t.split("_")
        axes[0].plot(best_pred[sel], y_loss[sel], "o", ms=2, alpha=0.5, color=cmap(i / (len(names) - 1)),
                     label=f"{az}°" + (f", el {el}°" if el != "0" else ""))
    lim = [min(y_loss.min(), best_pred.min()), max(y_loss.max(), best_pred.max())]
    axes[0].plot(lim, lim, color=INK2, lw=0.6)
    axes[0].set_xlabel("Predicted from room parameters (quadratic)")
    axes[0].set_ylabel("Acuity loss, ½ log det ratio")
    axes[0].legend(frameon=False, fontsize=6, markerscale=2, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), title="Source", title_fontsize=6, columnspacing=0.8, handletextpad=0.2)
    labels = {"drr": "DRR", "linear": "All five,\nlinear", "quadratic": "All five,\nquadratic",
              "quadratic+direction": "Quadratic\n+ direction"}
    x = np.arange(4)
    for j, (target, col, lab) in enumerate((("loss", "#2a78d6", "Overall"), ("loss_az", "#eb6834", "Azimuth"),
                                             ("loss_el", "#1baf7a", "Elevation"))):
        axes[1].bar(x + (j - 1) * 0.25, [out[target][k] for k in labels], width=0.23, color=col, label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels.values(), fontsize=6.5)
    axes[1].set_ylabel("Cross-validated R²")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
                   handlelength=1.0, columnspacing=1.0)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, lw=0.5)
    fig.savefig(ROOT / "figures" / "e5_redundancy.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "e5_redundancy.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()


def head_gain(sid: int, targets) -> dict:
    """Direct-to-diffuse power gain of the head at each target direction (dB), both ears, 200 Hz-16 kHz."""
    from sdr import hrtf, ism, room
    interp = ism.HrtfInterpolator(sid)
    g11, g22, _ = room.diffuse_matrix(interp.H, interp.w)
    band = (interp.f >= 200) & (interp.f <= 16000)
    out = {}
    for az, el in targets:
        a, e = np.radians(az), np.radians(el)
        H = interp(np.array([[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]]))[0]
        num = (np.abs(H[:, band]) ** 2).sum()
        out[f"{az}_{el}"] = float(10 * np.log10(num / (g11[band] + g22[band]).sum()))
    return out


def local_drr_test():
    """CV R^2 when the direction indicators are replaced by the head's direct-to-diffuse gain."""
    rows = load()
    targets = sorted({(int(r["az"]), int(r["el"])) for r in rows})
    gains = {s: head_gain(int(s), targets) for s in sorted({r["subject"] for r in rows})}
    for r in rows:
        r["head_gain_db"] = gains[r["subject"]][r["target"]]
        r["local_drr_db"] = float(r["drr_db"]) + r["head_gain_db"]
    groups = np.array([r["config"] for r in rows])
    out = {}
    for target in ("loss", "loss_az", "loss_el"):
        y = np.array([float(r[target]) for r in rows])
        Xq = design(rows, "quadratic")
        g = np.array([[r["head_gain_db"]] for r in rows])
        g = (g - g.mean()) / g.std()
        d = np.array([[float(r["drr_db"])] for r in rows])
        d = (d - d.mean()) / d.std()
        ld = np.array([[r["local_drr_db"]] for r in rows])
        ld = (ld - ld.mean()) / ld.std()
        out[target] = {
            "local_drr_only": round(float(cv_r2(np.hstack([np.ones((len(y), 1)), ld, ld ** 2]), y, groups)[0]), 3),
            "quadratic+head_gain": round(float(cv_r2(np.hstack([Xq, g, g * d]), y, groups)[0]), 3),
        }
    return out
