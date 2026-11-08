"""E5: is predicted acuity redundant with DRR, C80, LF, IACC and T60?

Shoebox rooms of three sizes and five absorptions, random listener positions, sources at 1 m
(ears at 1.2 m) and, in the 5 m high room only, at 2 m (ears at 1.8 m), so that every source
direction lies inside the room. The Fisher metric is computed as elsewhere, by order-6 SH fits of the cue fields, here on
a Fibonacci set of source directions above -45 deg elevation plus the target directions, with
the room and the anechoic case treated alike; the acuity loss at a target is half the log ratio
of their determinants. Room parameters come from the same
arrivals: images to order 3 plus the statistical late field, decaying at the Eyring rate after
t0 = 4 min(L) / c.
Output: results/e5_redundancy/subject_XXXX.csv
"""
from __future__ import annotations

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdr import cues, fisher, hrtf, ism

OUT = Path(__file__).resolve().parents[1] / "results" / "e5_redundancy"
SUBJECTS = hrtf.available_subjects()[::27]
ROOMS = [(5.0, 4.0, 3.0), (8.0, 6.0, 3.0), (14.0, 10.0, 5.0)]
ALPHA = [0.05, 0.1, 0.2, 0.35, 0.5]
DISTANCES = [1.0, 2.0]
N_POS = 3
TARGETS = [(0, 0), (30, 0), (60, 0), (90, 0), (120, 0), (150, 0), (180, 0), (0, 30)]
N_FIB = 200
RHO, TS_MS = 0.984, 5.6
EAR_HEIGHT = {1.0: 1.2, 2.0: 1.8}


def unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def direction_set():
    """Fibonacci directions above -45 deg elevation, with the targets first."""
    k = np.arange(N_FIB) + 0.5
    z = 1 - 2 * k / N_FIB
    phi = np.pi * (1 + 5 ** 0.5) * k
    fib = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], 1)
    fib = fib[fib[:, 2] >= np.sin(np.radians(-45))]
    tgt = np.array([unit(az, el) for az, el in TARGETS])
    keep = np.min(np.arccos(np.clip(fib @ tgt.T, -1, 1)), axis=1) > np.radians(4)
    return np.vstack([tgt, fib[keep]])


def metric(cm, dirs, R, Ran):
    """G (n_targets, 2, 2) in the (azimuth, elevation) basis at the target directions."""
    c = cm.cues(R, Ran)
    G = fisher.fisher_metric(dirs, c.q, c.var)[0][:len(TARGETS)]
    t = dirs[:len(TARGETS)]
    e = np.stack([fisher.azimuth_vector(t), fisher.elevation_vector(t)], 1)
    u1, u2 = fisher.tangent_basis(t)
    B = np.stack([np.einsum("nkd,nd->nk", e, u1), np.einsum("nkd,nd->nk", e, u2)], 2)
    return np.einsum("nki,nij,nlj->nkl", B, G, B)


def room_parameters(box, listener, u, r, interp, R):
    """DRR, C80, LF and T60 from the arrivals, and IACC (500-2000 Hz octaves) from R."""
    src = listener + r * u
    pos, orders = box.images(src)
    vec = pos - listener
    dist = np.linalg.norm(vec, axis=1)
    t = (dist - dist[0]) / ism.C
    e = (box.beta ** orders * dist[0] / dist) ** 2
    e_rev = 16 * np.pi * dist[0] ** 2 / box.room_constant()
    e_late = ism.late_energy(box, e_rev, e[1:].sum())
    T60 = box.rt60()
    k = 13.82 / T60
    t0 = (box.order + 1) * min(box.L) / ism.C

    def late(a, b):
        a, b = max(a, t0), max(b, t0)
        return e_late * (np.exp(-k * (a - t0)) - np.exp(-k * (b - t0)))

    early = e[t <= 0.08].sum() + late(0, 0.08)
    lateE = e[t > 0.08].sum() + late(0.08, np.inf)
    cos2 = (vec[:, 1] / dist) ** 2
    lat = ((t >= 0.005) & (t <= 0.08))
    lf = ((e * cos2)[lat].sum() + late(0.005, 0.08) / 3) / early
    drr = 10 * np.log10(1.0 / (e[1:].sum() + e_late))
    R11, R22, R12 = (x[0] for x in R)
    f = interp.f
    taus = np.linspace(-1e-3, 1e-3, 401)
    iacc = []
    for fc in (500, 1000, 2000):
        b = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        cc = np.real(R12[b] @ np.exp(2j * np.pi * np.outer(f[b], taus)))
        iacc.append(np.max(np.abs(cc)) / np.sqrt(R11[b].sum() * R22[b].sum()))
    return dict(drr_db=drr, c80_db=10 * np.log10(early / lateE), lf=lf, iacc=float(np.mean(iacc)), t60_s=T60)


def positions(L, r, rng):
    """Listener positions with every source at distance r at least 0.3 m inside the room."""
    z = EAR_HEIGHT[r]
    if z - r * np.sin(np.pi / 4) < 0.3 or z + r > L[2] - 0.3 or min(L[:2]) < 2 * (r + 0.3) + 0.5:
        return []
    m = r + 0.3
    return [np.array([rng.uniform(m, L[0] - m), rng.uniform(m, L[1] - m), z]) for _ in range(N_POS)]


def run(sid: int) -> str:
    interp = ism.HrtfInterpolator(sid)
    cm = cues.CueModel(interp.f, rho_int=RHO)
    dirs = direction_set()
    H = interp(dirs)
    Ran = (np.abs(H[:, 0]) ** 2, np.abs(H[:, 1]) ** 2, H[:, 0] * np.conj(H[:, 1]))
    G0 = metric(cm, dirs, Ran, Ran)
    rng = np.random.default_rng(0)
    rows = []
    for L in ROOMS:
        for r in DISTANCES:
            for pi, p in enumerate(positions(L, r, rng)):
                for alpha in ALPHA:
                    box = ism.Shoebox(L, alpha, 3)
                    R, _ = ism.binaural_cross_spectra(box, p, dirs, r, interp, TS_MS)
                    G = metric(cm, dirs, R, Ran)
                    for n, (az, el) in enumerate(TARGETS):
                        Rn = tuple(x[n:n + 1] for x in R)
                        prm = room_parameters(box, p, dirs[n], r, interp, Rn)
                        rows.append(dict(subject=sid, room="x".join(f"{v:g}" for v in L), alpha=alpha,
                                         distance_m=r, position=pi, az=az, el=el,
                                         loss=0.5 * np.log(np.linalg.det(G0[n]) / np.linalg.det(G[n])),
                                         loss_az=0.5 * np.log(G0[n, 0, 0] / G[n, 0, 0]),
                                         loss_el=0.5 * np.log(G0[n, 1, 1] / G[n, 1, 1]), **prm))
    with open(OUT / f"subject_{sid:04d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return f"{sid} done ({len(rows)} rows)"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SUBJECTS if not (OUT / f"subject_{s:04d}.csv").exists()]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        for msg in pool.imap_unordered(run, todo):
            print(msg, flush=True)
