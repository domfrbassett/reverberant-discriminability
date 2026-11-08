"""E6 check: frozen reflections against image sources that move with the source.

Shoebox configurations as in E5 (three rooms, absorption 0.1 and 0.35, two listener positions,
sources at 1 m, ears at 1.2 m), four heads, precedence weighting. For each target direction
the metric is computed twice: with every image recomputed for every source direction (E5), and
with the reflections of the target source held fixed while only the direct sound moves.
Output: results/e6_frozen_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e5_redundancy import RHO, TS_MS, TARGETS, direction_set, metric, positions
from sdr import cues, hrtf, ism, maps

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = hrtf.available_subjects()[::54]
ROOMS = [(5.0, 4.0, 3.0), (8.0, 6.0, 3.0), (14.0, 10.0, 5.0)]
ALPHA = [0.1, 0.35]
R_SRC = 1.0
N_POS = 2


def main():
    rows = []
    view = np.array([1.0, 0.0, 0.0])
    for sid in SUBJECTS:
        interp = ism.HrtfInterpolator(sid)
        cm = cues.CueModel(interp.f, rho_int=RHO)
        dirs = direction_set()
        H = interp(dirs)
        Ran = (np.abs(H[:, 0]) ** 2, np.abs(H[:, 1]) ** 2, H[:, 0] * np.conj(H[:, 1]))
        G0 = metric(cm, dirs, Ran, Ran)
        rng = np.random.default_rng(1)
        for L in ROOMS:
            for p in positions(L, R_SRC, rng)[:N_POS]:
                for alpha in ALPHA:
                    box = ism.Shoebox(L, alpha, 3)
                    R, _ = ism.binaural_cross_spectra(box, p, dirs, R_SRC, interp, TS_MS)
                    Gm = metric(cm, dirs, R, Ran)
                    for n, (az, el) in enumerate(TARGETS):
                        seat = maps.from_shoebox(box, p, p + R_SRC * dirs[n], view)
                        Gf = metric(cm, dirs, maps.cross_spectra(seat, dirs, interp, TS_MS), Ran)
                        loss = lambda G: 0.5 * np.log(np.linalg.det(G0[n]) / np.linalg.det(G[n]))
                        rows.append(dict(subject=sid, room="x".join(f"{v:g}" for v in L), alpha=alpha, az=az, el=el,
                                         loss_moving=float(loss(Gm)), loss_frozen=float(loss(Gf)),
                                         az_moving=float(np.degrees(1 / np.sqrt(Gm[n, 0, 0]))),
                                         az_frozen=float(np.degrees(1 / np.sqrt(Gf[n, 0, 0]))),
                                         el_moving=float(np.degrees(1 / np.sqrt(Gm[n, 1, 1]))),
                                         el_frozen=float(np.degrees(1 / np.sqrt(Gf[n, 1, 1])))))
        print(sid, "done", flush=True)
    lm = np.array([r["loss_moving"] for r in rows])
    lf = np.array([r["loss_frozen"] for r in rows])
    az = np.array([[r["az_moving"], r["az_frozen"]] for r in rows])
    el = np.array([[r["el_moving"], r["el_frozen"]] for r in rows])
    tgt = np.array([f"{r['az']}_{r['el']}" for r in rows])
    summary = {
        "n": len(rows), "subjects": [int(s) for s in SUBJECTS],
        "loss_correlation": round(float(np.corrcoef(lm, lf)[0, 1]), 3),
        "loss_median_abs_diff": round(float(np.median(np.abs(lf - lm))), 3),
        "loss_median_diff": round(float(np.median(lf - lm)), 3),
        "az_threshold_median_abs_log_ratio": round(float(np.median(np.abs(np.log(az[:, 1] / az[:, 0])))), 3),
        "el_threshold_median_abs_log_ratio": round(float(np.median(np.abs(np.log(el[:, 1] / el[:, 0])))), 3),
        "az_threshold_median_log_ratio_by_target": {t: round(float(np.median(np.log(az[tgt == t, 1] / az[tgt == t, 0]))), 3)
                                                    for t in dict.fromkeys(tgt)},
        "el_threshold_median_log_ratio_by_target": {t: round(float(np.median(np.log(el[tgt == t, 1] / el[tgt == t, 0]))), 3)
                                                    for t in dict.fromkeys(tgt)},
    }
    (ROOT / "results" / "e6_frozen_check.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
