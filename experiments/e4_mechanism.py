"""Mechanism split: cue-mean compression against interaural variance inflation.

For each DRR the metric is recomputed with room means and anechoic variances
(compression only) and with anechoic means and room variances (inflation only).
Output: results/e4_mechanism.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdr import cues, fisher, hrtf, room

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = hrtf.available_subjects()
DRR_DB = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0]
RHO = 0.984
DIRS = {"az0": (0, 0), "az30": (30, 0), "az60": (60, 0), "az90": (90, 0), "el30": (0, 30)}


def index(r, az, el):
    a = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
    e = np.degrees(np.arcsin(r[:, 2]))
    return int(np.argmin(np.abs(((a - az + 180) % 360) - 180) + np.abs(e - el)))


def sqrtdet(G):
    return np.sqrt(np.linalg.det(G))


def subject(sid):
    r, f, H, w = hrtf.dtf_spectra(sid)
    cm = cues.CueModel(f, rho_int=RHO)
    an = room.cross_spectra(H, w, np.inf)
    ca = cm.cues(an)
    G0 = fisher.fisher_metric(r, ca.q, ca.var)[0]
    idx = {k: index(r, *v) for k, v in DIRS.items()}
    rows = {k: {m: [] for m in ("full", "compression", "inflation")} for k in DIRS}
    for drr in DRR_DB:
        c = cm.cues(room.cross_spectra(H, w, drr), an)
        Gs = {"full": fisher.fisher_metric(r, c.q, c.var)[0],
              "compression": fisher.fisher_metric(r, c.q, ca.var)[0],
              "inflation": fisher.fisher_metric(r, ca.q, c.var)[0]}
        for k, i in idx.items():
            for m, G in Gs.items():
                rows[k][m].append(float(sqrtdet(G[i]) / sqrtdet(G0[i])))
    return rows


def main(workers):
    out = {k: {m: [] for m in ("full", "compression", "inflation")} for k in DIRS}
    with Pool(workers) as pool:
        for sid, rows in zip(SUBJECTS, pool.imap(subject, SUBJECTS)):
            for k in DIRS:
                for m in rows[k]:
                    out[k][m].append(rows[k][m])
            print(sid, flush=True)
    med = {k: {m: np.median(np.array(v), 0).round(3).tolist() for m, v in d.items()}
           for k, d in out.items()}
    (ROOT / "results" / "e4_mechanism.json").write_text(
        json.dumps({"drr_db": DRR_DB, "rho": RHO, "subjects": len(SUBJECTS),
                    "sqrtdet_relative_to_anechoic_median": med}, indent=1))
    print(json.dumps(med, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
