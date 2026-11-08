"""C1: internal coherence ceiling rho from Rakerd & Hartmann (2010).

Data: data/rakerd_hartmann_2010.csv, psychometric functions digitised from the vector
figures (Figs. 2, 3, 8). Model of a single-interval lateralisation judgement:

    P(consistent) = Phi(dt / sigma(gamma)),
    sigma(gamma)^2 = sigma0^2 * m(gamma),  m = [(1 - r2 g2)/(r2 g2)] / [(1 - r2)/r2],

with r2 = rho^2 and g2 the squared coherence driving the estimate (waveform coherence for the
low and mid bands; waveform or envelope, gamma^2, for the high band).
Output: results/c1_calibration.json, figures/c1_calibration.pdf
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["low", "mid", "high"]
TITLES = {"low": "225 Hz band", "mid": "715 Hz band", "high": "2850 Hz band (envelope)"}
INK, INK2 = "#1f1f1e", "#5e5d59"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def load(figures=("2", "3", "8a")):
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "rakerd_hartmann_2010.csv"))
            if r["figure"] in figures]
    return {b: np.array([[float(r["coherence"]), float(r["itd_us"]), float(r["percent"]) / 100]
                         for r in rows if r["band"] == b]) for b in BANDS}


def m_factor(g, rho):
    r2, g2 = rho ** 2, np.clip(g, 1e-6, 1) ** 2
    return ((1 - r2 * g2) / (r2 * g2)) / ((1 - r2) / r2)


def predict(d, sigma0, rho, envelope=False):
    g = d[:, 0] ** 2 if envelope else d[:, 0]
    return norm.cdf(d[:, 1] / (sigma0 * np.sqrt(m_factor(g, rho))))


def nll(p, d, sigma0, rho, envelope=False):
    q = np.clip(predict(d, sigma0, rho, envelope), 1e-6, 1 - 1e-6)
    y = np.clip(d[:, 2], 0, 1)
    return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))


def fit(data, shared: bool, envelope_high: bool):
    def unpack(x):
        s0 = dict(zip(BANDS, np.exp(x[:3])))
        if shared:
            rho = dict.fromkeys(BANDS, 1 / (1 + np.exp(-x[3])))
        else:
            rho = dict(zip(BANDS, 1 / (1 + np.exp(-x[3:6]))))
        return s0, rho

    def obj(x):
        s0, rho = unpack(x)
        return sum(nll(None, data[b], s0[b], rho[b], envelope_high and b == "high") for b in BANDS)

    x0 = np.r_[np.log([40, 30, 150]), [2.0] if shared else [2.0, 2.0, 2.0]]
    best = None
    for start in ([1.0], [2.0], [3.5]):
        x = x0.copy()
        x[3:] = start[0]
        res = minimize(obj, x, method="Nelder-Mead", options=dict(maxiter=40000, xatol=1e-8, fatol=1e-10))
        if best is None or res.fun < best.fun:
            best = res
    s0, rho = unpack(best.x)
    rms = {b: float(100 * np.sqrt(np.mean((predict(data[b], s0[b], rho[b], envelope_high and b == "high")
                                            - data[b][:, 2]) ** 2))) for b in BANDS}
    return {"sigma0_us": {b: float(v) for b, v in s0.items()}, "rho": {b: float(v) for b, v in rho.items()},
            "nll": float(best.fun), "rms_pp": rms, "n_params": len(best.x)}


def main():
    data = load()
    out = {}
    for name, shared, env in (("separate", False, False), ("shared", True, False),
                              ("separate_env", False, True), ("shared_env", True, True)):
        out[name] = fit(data, shared, env)
    check = load(("8b",))["mid"]
    s0, rho = out["shared_env"]["sigma0_us"]["mid"], out["shared_env"]["rho"]["mid"]
    out["check_fig8b_rms_pp"] = float(100 * np.sqrt(np.mean((predict(check, s0, rho) - check[:, 2]) ** 2)))
    (ROOT / "results" / "c1_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    best = out["shared_env"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True, gridspec_kw=dict(wspace=0.12))
    blues = plt.get_cmap("Blues")
    colour = lambda c: blues(0.3 + 0.7 * (c - 0.2) / 0.8)
    handles = {}
    for ax, b in zip(axes, BANDS):
        d = data[b]
        for c in sorted(set(d[:, 0])):
            sel = d[:, 0] == c
            t = np.linspace(0, d[:, 1].max() * 1.05, 200)
            q = predict(np.c_[np.full_like(t, c), t, t], best["sigma0_us"][b], best["rho"][b], b == "high")
            ax.plot(t, 100 * q, color=colour(c), lw=1.2)
            handles[c] = ax.plot(d[sel, 1], 100 * d[sel, 2], "o", color=colour(c), ms=3.5,
                                 markeredgecolor="white", markeredgewidth=0.4)[0]
        ax.set_title(TITLES[b], fontsize=8, color=INK)
        ax.set_xlabel("ITD (µs)")
        ax.set_ylim(45, 102)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    keys = sorted(handles)
    fig.legend([handles[k] for k in keys], [f"{k:g}" for k in keys], title="Coherence", frameon=False,
               loc="center left", bbox_to_anchor=(0.91, 0.5), fontsize=7, title_fontsize=7)
    axes[0].set_ylabel("Percent consistent")
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "c1_calibration.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "c1_calibration.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
