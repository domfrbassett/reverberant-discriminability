"""E2: temporal weighting fitted to lead and lag ITD discrimination (Litovsky & Shinn-Cunningham 2001).

The target burst's ITD is estimated against the other burst, which dilutes its effective
interaural coherence within the binaural integration window:

    gamma = a_t / (a_t + a_n c(D)),   lead: a_t = 1, a_n = w(D);   lag: a_t = w(D), a_n = 1,
    w(D) = 1 - exp(-D / T_s)          recovery of the weight of a later arrival,
    c(D) = exp(-D / T_b)              overlap of the binaural integration window,
    jnd  = A_s * sqrt(m(gamma; rho)),

with m the Knapp-Carter variance ratio of C1, rho = 0.984 from C1 or free, and a scale A_s per
listener (the single-burst control lacks the roving non-target ITD, so it does not set the scale).
The stationary model has no temporal weighting (w = 1). Log jnds have Gaussian error; jnds at or
above the adaptive-track ceiling are right-censored.
Output: results/e2_precedence.json, figures/e2_precedence.pdf
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
RHO = 0.984
CEILING = 320.0
INK, INK2, LEAD, LAG = "#1f1f1e", "#5e5d59", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def load():
    rows = list(csv.DictReader(open(ROOT / "data" / "litovsky_shinn_2001_fig3.csv")))
    single = {r["subject"]: float(r["jnd_us"]) for r in rows if r["condition"] == "single"}
    data = [(r["subject"], r["condition"], float(r["delay_ms"]), float(r["jnd_us"]))
            for r in rows if r["condition"] != "single"]
    return single, data


def m_factor(g, rho):
    r2, g2 = rho ** 2, np.clip(g, 1e-9, 1) ** 2
    return ((1 - r2 * g2) / (r2 * g2)) / ((1 - r2) / r2)


def predict(A, cond, D, Ts, Tb, rho=RHO, weighted=True):
    w = 1 - np.exp(-D / Ts) if weighted else 1.0
    c = np.exp(-D / Tb)
    at, an = (1.0, w) if cond == "lead" else (w, 1.0)
    return A * np.sqrt(m_factor(at / (at + an * c), rho))


SUBJ = ["S1", "S2", "S3", "S4", "S5", "S6"]


def unpack(x, fit_rho, weighted):
    k = 0
    A = dict(zip(SUBJ, np.exp(x[k:k + 6])))
    k += 6
    Ts = np.exp(x[k]) if weighted else np.inf
    k += 1 if weighted else 0
    Tb, sl = np.exp(x[k]), np.exp(x[k + 1])
    k += 2
    rho = 1 / (1 + np.exp(-x[k])) if fit_rho else RHO
    return A, Ts, Tb, sl, rho


def nll(x, data, fit_rho, weighted):
    A, Ts, Tb, sl, rho = unpack(x, fit_rho, weighted)
    out = 0.0
    for s, cond, D, j in data:
        mu = np.log(predict(A[s], cond, D, Ts, Tb, rho, weighted))
        out -= norm.logsf(np.log(CEILING), mu, sl) if j >= CEILING else norm.logpdf(np.log(j), mu, sl)
    return out


def fit(data, fit_rho=False, weighted=True):
    best = None
    for ts0 in (2.0, 6.0):
        for tb0 in (8.0, 30.0, 100.0):
            x0 = np.r_[np.log(np.full(6, 30.0)), [np.log(ts0)] if weighted else [], np.log([tb0, 0.3]),
                       [3.0] if fit_rho else []]
            r = minimize(nll, x0, args=(data, fit_rho, weighted), method="Nelder-Mead",
                         options=dict(maxiter=60000, xatol=1e-7, fatol=1e-9, adaptive=True))
            if best is None or r.fun < best.fun:
                best = r
    A, Ts, Tb, sl, rho = unpack(best.x, fit_rho, weighted)
    k = len(best.x)
    return dict(A_us={s: float(v) for s, v in A.items()}, T_s_ms=float(Ts) if weighted else None,
                T_b_ms=float(Tb), sigma_log=float(sl), rho=float(rho), nll=float(best.fun),
                n_params=k, aic=float(2 * k + 2 * best.fun))


def main():
    single, data = load()
    out = {"weighted_fixed_rho": fit(data), "weighted_free_rho": fit(data, fit_rho=True),
           "stationary_fixed_rho": fit(data, weighted=False),
           "stationary_free_rho": fit(data, fit_rho=True, weighted=False)}
    p = out["weighted_fixed_rho"]
    res = [np.log(predict(p["A_us"][s], c, D, p["T_s_ms"], p["T_b_ms"])) - np.log(j)
           for s, c, D, j in data if j < CEILING]
    out["rms_log_uncensored_weighted_fixed_rho"] = float(np.sqrt(np.mean(np.square(res))))
    out["n_censored"] = int(sum(j >= CEILING for *_, j in data))
    (ROOT / "results" / "e2_precedence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "A_us"})
                      for k, v in out.items()}, indent=1))

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.1, hspace=0.25))
    Dg = np.linspace(0.5, 16, 200)
    for ax, s in zip(axes.T.ravel(), sorted(single)):
        for cond, col in (("lead", LEAD), ("lag", LAG)):
            pts = [(D, j) for ss, c, D, j in data if ss == s and c == cond]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], "o" if cond == "lead" else "D", color=col,
                    ms=3.5, mfc=col if cond == "lead" else "white", mew=1, label=cond.capitalize())
            ax.plot(Dg, [min(predict(p["A_us"][s], cond, D, p["T_s_ms"], p["T_b_ms"]), 450) for D in Dg],
                    color=col, lw=1.2)
        ax.axhline(single[s], color=INK2, lw=0.6, ls=":")
        ax.axhspan(CEILING, 460, color="#e4e3df", lw=0)
        ax.set_title(s, fontsize=8, color=INK)
        ax.set_ylim(0, 460)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Delay (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ITD jnd (µs)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(ROOT / "figures" / "e2_precedence.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "e2_precedence.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()


def profile_Ts(grid=(2, 3, 4, 5, 5.6, 7, 9, 12, 16)):
    """Profile negative log-likelihood of T_s, other parameters refitted (rho fixed)."""
    single, data = load()
    si = np.array([SUBJ.index(s) for s, *_ in data])
    lead = np.array([c == "lead" for _, c, *_ in data])
    D = np.array([d for *_, d, _ in data])
    j = np.array([v for *_, v in data])
    cens = j >= CEILING

    def f(x, Ts):
        A, Tb, sl = np.exp(x[:6]), np.exp(x[6]), np.exp(x[7])
        w = 1 - np.exp(-D / Ts)
        c = np.exp(-D / Tb)
        at = np.where(lead, 1.0, w)
        an = np.where(lead, w, 1.0)
        mu = np.log(A[si] * np.sqrt(m_factor(at / (at + an * c), RHO)))
        return -(norm.logsf(np.log(CEILING), mu[cens], sl).sum()
                 + norm.logpdf(np.log(j[~cens]), mu[~cens], sl).sum())

    out = {}
    for Ts in grid:
        best = min((minimize(f, np.r_[np.log(np.full(6, 150.0)), np.log([tb, 0.3])], args=(Ts,),
                             method="Nelder-Mead", options=dict(maxiter=40000, adaptive=True))
                    for tb in (20.0, 50.0, 150.0)), key=lambda r: r.fun)
        out[float(Ts)] = float(best.fun)
    return out
