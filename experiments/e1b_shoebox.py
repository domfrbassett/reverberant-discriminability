"""E1b: Fisher metric in a shoebox room from the image-source model, by listener position.

Room 8 x 6 x 3 m, absorption 0.1 and 0.3, listener at the centre, 1 m from a side wall (right
ear towards it) and 1 m from a corner (back and right walls), facing +x, ears at 1.2 m. Sources
at 0.9 m in every measured direction. Images to order 3, late field from the statistical model,
with and without precedence weighting (T_s = 5.6 ms); diffuse-only model at the same DRR as
reference. Output: results/e1b_shoebox/subject_XXXX.npz
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdr import cues, fisher, hrtf, ism, room

OUT = Path(__file__).resolve().parents[1] / "results" / "e1b_shoebox"
SUBJECTS = hrtf.available_subjects()[::14]
L = (8.0, 6.0, 3.0)
ALPHA = [0.1, 0.3]
POSITIONS = {"centre": (4.0, 3.0, 1.2), "side": (4.0, 1.0, 1.2), "corner": (1.0, 1.0, 1.2)}
R_SRC = 0.9
TS_MS = 5.6
RHO = 0.984


def run(sid: int) -> str:
    interp = ism.HrtfInterpolator(sid)
    r, f, H, w = hrtf.dtf_spectra(sid)
    cm = cues.CueModel(f, rho_int=RHO)
    an = room.cross_spectra(H, w, np.inf)
    labels, G_all, drr_all, coh_all = [], [], [], []
    for alpha in ALPHA:
        box = ism.Shoebox(L, alpha, 3)
        for pos_name, p in POSITIONS.items():
            for weighted in (False, True):
                R, drr = ism.binaural_cross_spectra(box, np.array(p), r, R_SRC, interp,
                                                    TS_MS if weighted else None)
                c = cm.cues(R, an)
                G_all.append(fisher.fisher_metric(r, c.q, c.var)[0])
                labels.append(f"a{alpha}_{pos_name}_{'weighted' if weighted else 'stationary'}")
                drr_all.append(float(np.median(drr)))
                coh_all.append(c.gamma_itd)
        c = cm.cues(room.cross_spectra(H, w, drr_all[-1]), an)
        G_all.append(fisher.fisher_metric(r, c.q, c.var)[0])
        labels.append(f"a{alpha}_diffuse")
        drr_all.append(drr_all[-1])
        coh_all.append(c.gamma_itd)
    np.savez_compressed(OUT / f"subject_{sid:04d}.npz", r=r, labels=np.array(labels),
                        G=np.array(G_all), drr=np.array(drr_all), gamma=np.array(coh_all))
    return f"{sid} done"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SUBJECTS if not (OUT / f"subject_{s:04d}.npz").exists()]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        for msg in pool.imap_unordered(run, todo):
            print(msg, flush=True)
