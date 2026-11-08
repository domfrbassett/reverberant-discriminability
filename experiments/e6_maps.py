"""E6: acuity maps over seats in shoebox analogues of three room types.

Lecture theatre 20 x 15 x 6 m (absorption 0.3), open-plan dining space 30 x 20 x 3.5 m (0.2)
and concert hall 40 x 24 x 16 m (0.25), one source each, listeners on a grid at ear height
1.2 m facing the source. Reflections from images to order 3 are held fixed per seat while the
direct sound moves (sdr/maps.py), with precedence weighting; two heads.
Output: results/e6_maps/<room>.npz, results/e6_maps/summary.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.e5_redundancy import N_FIB, RHO, TS_MS
from sdr import cues, fisher, hrtf, ism, maps

OUT = Path(__file__).resolve().parents[1] / "results" / "e6_maps"
SUBJECTS = hrtf.available_subjects()[::81][:2]
EAR = 1.2
ROOMS = {
    "lecture": dict(L=(20.0, 15.0, 6.0), alpha=0.3, source=(2.5, 7.5, 1.6), step=2.0),
    "dining": dict(L=(30.0, 20.0, 3.5), alpha=0.2, source=(15.0, 10.0, 1.2), step=3.0),
    "concert": dict(L=(40.0, 24.0, 16.0), alpha=0.25, source=(4.0, 12.0, 1.6), step=4.0),
}


def unit(v):
    return v / np.linalg.norm(v)


def directions(target):
    k = np.arange(N_FIB) + 0.5
    z = 1 - 2 * k / N_FIB
    phi = np.pi * (1 + 5 ** 0.5) * k
    fib = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], 1)
    fib = fib[fib[:, 2] >= np.sin(np.radians(-45))]
    fib = fib[np.arccos(np.clip(fib @ target, -1, 1)) > np.radians(4)]
    return np.vstack([target, fib])


def thresholds(cm, dirs, R, Ran):
    c = cm.cues(R, Ran)
    G = fisher.fisher_metric(dirs, c.q, c.var)[0][:1]
    t = dirs[:1]
    az = fisher.directional_information(G, t, fisher.azimuth_vector(t))[0]
    el = fisher.directional_information(G, t, fisher.elevation_vector(t))[0]
    return np.degrees(1 / np.sqrt(az)), np.degrees(1 / np.sqrt(el)), 0.5 * np.log(np.linalg.det(G[0]))


def seats(spec):
    L, s, d = np.array(spec["L"]), np.array(spec["source"]), spec["step"]
    xs = np.arange(1.0, L[0] - 0.99, d)
    ys = np.arange(1.0, L[1] - 0.99, d)
    pts = [np.array([x, y, EAR]) for x in xs for y in ys if np.hypot(x - s[0], y - s[1]) >= 2.0]
    return np.array(pts)


def run(job):
    name, sid = job
    spec = ROOMS[name]
    box = ism.Shoebox(spec["L"], spec["alpha"], 3)
    interp = ism.HrtfInterpolator(sid)
    cm = cues.CueModel(interp.f, rho_int=RHO)
    src = np.array(spec["source"])
    P = seats(spec)
    out = {k: [] for k in ("az_deg", "el_deg", "az_an_deg", "el_an_deg", "logdet_loss", "drr_db", "dist_m", "src_el_deg")}
    for p in P:
        view = src - p
        seat = maps.from_shoebox(box, p, src, view)
        target = maps.head_frame(unit(src - p)[None, :], view)[0]
        dirs = directions(target)
        H = interp(dirs)
        Ran = (np.abs(H[:, 0]) ** 2, np.abs(H[:, 1]) ** 2, H[:, 0] * np.conj(H[:, 1]))
        a0, e0, l0 = thresholds(cm, dirs, Ran, Ran)
        a, e, l = thresholds(cm, dirs, maps.cross_spectra(seat, dirs, interp, TS_MS), Ran)
        e_rev = (seat.amp ** 2).sum() + seat.e_late
        for k, v in zip(out, (a, e, a0, e0, l0 - l, 10 * np.log10(1 / e_rev), np.linalg.norm(src - p),
                              np.degrees(np.arcsin(target[2])))):
            out[k].append(float(v))
    np.savez_compressed(OUT / f"{name}_{sid:04d}.npz", seats=P, **{k: np.array(v) for k, v in out.items()})
    return f"{name} {sid} done ({len(P)} seats)"


def summarise():
    summary = {"subjects": [int(s) for s in SUBJECTS], "rooms": {}}
    for name, spec in ROOMS.items():
        d = [np.load(OUT / f"{name}_{s:04d}.npz") for s in SUBJECTS]
        az = np.median([x["az_deg"] for x in d], 0)
        el = np.median([x["el_deg"] for x in d], 0)
        loss = np.median([x["logdet_loss"] for x in d], 0)
        drr = d[0]["drr_db"]
        A = np.column_stack([np.ones_like(drr), drr, drr ** 2])
        fit = A @ np.linalg.lstsq(A, loss, rcond=None)[0]
        res = loss - fit
        box = ism.Shoebox(spec["L"], spec["alpha"], 3)
        summary["rooms"][name] = {
            "L": spec["L"], "alpha": spec["alpha"], "rt60_s": round(box.rt60(), 2), "n_seats": int(len(drr)),
            "drr_db_range": [round(float(drr.min()), 1), round(float(drr.max()), 1)],
            "az_deg_range": [round(float(az.min()), 2), round(float(az.max()), 2)],
            "el_deg_range": [round(float(el.min()), 2), round(float(el.max()), 2)],
            "az_deg_median": round(float(np.median(az)), 2), "el_deg_median": round(float(np.median(el)), 2),
            "az_anechoic_deg_median": round(float(np.median([x["az_an_deg"] for x in d])), 2),
            "loss_r2_on_drr_quadratic": round(float(1 - res.var() / loss.var()), 3),
            "loss_residual_sd": round(float(res.std()), 3),
            "corr_az_log_with_drr": round(float(np.corrcoef(np.log(az), drr)[0, 1]), 3),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarise()
    else:
        jobs = [(n, s) for n in ROOMS for s in SUBJECTS if not (OUT / f"{n}_{s:04d}.npz").exists()]
        with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
            for msg in pool.imap_unordered(run, jobs):
                print(msg, flush=True)
        summarise()
