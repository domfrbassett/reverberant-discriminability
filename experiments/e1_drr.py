"""E1 pilot: Fisher metric over source direction against direct-to-reverberant ratio.

Direct path plus diffuse late field, all available SONICOM subjects.
Output: results/e1_drr/subject_XXXX.npz
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdr import cues, fisher, hrtf, room

OUT = Path(__file__).resolve().parents[1] / "results" / "e1_drr"
SUBJECTS = hrtf.available_subjects()
DRR_DB = [np.inf, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0]
RHO = [0.984, 0.978, 0.989]


def run(sid: int) -> str:
    r, f, H, w = hrtf.dtf_spectra(sid)
    an = room.cross_spectra(H, w, np.inf)
    e_az, e_el = fisher.azimuth_vector(r), fisher.elevation_vector(r)
    out = {k: [] for k in ("G", "Gm", "Gv", "fam_info", "itd_s", "gamma", "var_ild", "var_itd")}
    for rho in RHO:
        cm = cues.CueModel(f, rho_int=rho)
        for drr in DRR_DB:
            c = cm.cues(room.cross_spectra(H, w, drr), an)
            G, Gm, Gv = fisher.fisher_metric(r, c.q, c.var)
            J = fisher.field_jacobian(r, c.q, 6)
            fam = np.stack([np.einsum("dmi,dm,dmj->dij", J[:, c.fam == k],
                                      1 / c.var[:, c.fam == k], J[:, c.fam == k]) for k in range(3)], 1)
            for k, v in zip(out, (G, Gm, Gv, fam, c.itd_s, c.gamma_itd, c.var[:, -2], c.var[:, -1])):
                out[k].append(v)
    np.savez_compressed(OUT / f"subject_{sid:04d}.npz", r=r, drr_db=np.array(DRR_DB),
                        rho=np.array(RHO), **{k: np.array(v) for k, v in out.items()})
    return f"{sid} done"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SUBJECTS if not (OUT / f"subject_{s:04d}.npz").exists()]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for msg in pool.imap_unordered(run, todo):
            print(msg, flush=True)
