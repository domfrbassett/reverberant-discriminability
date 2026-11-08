"""E1d: sensitivity of the diffuse-field matrix to the unmeasured cap below -45 deg elevation.

Three treatments of the cap in Gamma: 'ring', the default, where the Voronoi cells of the lowest
measured ring extend over the cap; 'mirror', where directions above +45 deg are reflected to
below -45 deg and carry the DTFs of their mirror images; 'none', where the cap is given no weight.
Direct path plus diffuse field, every 14th available SONICOM subject, rho = 0.984.
Output: results/e1d_lower_cap.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sdr import cues, fisher, hrtf, room

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = hrtf.available_subjects()[::14]
DRR_DB = [10.0, 5.0, 0.0, -5.0, -10.0]
RHO = 0.984
TARGETS = {"az0": (0, 0, "az"), "az90": (90, 0, "az"), "el0": (0, 0, "el")}
CAP = np.sin(np.radians(45.0))


def index(r, az, el):
    a = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
    e = np.degrees(np.arcsin(r[:, 2]))
    return int(np.argmin(np.abs(((a - az + 180) % 360) - 180) + np.abs(e - el)))


def gammas(r, H, w):
    up = r[:, 2] > CAP + 1e-6
    ra = np.vstack([r, r[up] * [1, 1, -1]])
    wa = hrtf.quadrature_weights(ra)
    Ha = np.concatenate([H, H[up]])
    n = len(r)
    w_none = wa[:n] / wa[:n].sum()
    return {"ring": room.diffuse_matrix(H, w), "mirror": room.diffuse_matrix(Ha, wa),
            "none": room.diffuse_matrix(H, w_none)}, float(wa[n:].sum()), float(w[r[:, 2] < -CAP + 1e-6].sum())


def main():
    out = {k: {t: [] for t in TARGETS} | {"gamma": []} for k in ("ring", "mirror", "none")}
    cap_w, ring_w = [], []
    for sid in SUBJECTS:
        r, f, H, w = hrtf.dtf_spectra(sid)
        cm = cues.CueModel(f, rho_int=RHO)
        an = room.direct_spectra(H)
        G_all, cw, rw = gammas(r, H, w)
        cap_w.append(cw)
        ring_w.append(rw)
        e = {"az": fisher.azimuth_vector(r), "el": fisher.elevation_vector(r)}
        idx = {t: index(r, a, el) for t, (a, el, _) in TARGETS.items()}
        for k, Gm in G_all.items():
            rows = {t: [] for t in TARGETS} | {"gamma": []}
            for drr in DRR_DB:
                eta = 10 ** (-drr / 10)
                R = tuple(x + eta * g[None, :] for x, g in zip(an, Gm))
                c = cm.cues(R, an)
                G = fisher.fisher_metric(r, c.q, c.var)[0]
                for t, (_, _, kind) in TARGETS.items():
                    i = idx[t]
                    J = fisher.directional_information(G[i:i + 1], r[i:i + 1], e[kind][i:i + 1])[0]
                    rows[t].append(float(np.degrees(1 / np.sqrt(J))))
                rows["gamma"].append(float(c.gamma_itd[idx["az0"]]))
            for t in rows:
                out[k][t].append(rows[t])
        print(sid, "done", flush=True)
    res = {k: {t: np.array(v) for t, v in d.items()} for k, d in out.items()}
    summary = {"subjects": [int(s) for s in SUBJECTS], "drr_db": DRR_DB, "rho": RHO,
               "cap_weight_mirror_median": round(float(np.median(cap_w)), 4),
               "lowest_ring_weight_default_median": round(float(np.median(ring_w)), 4),
               "threshold_deg_median": {k: {t: np.round(np.median(v, 0), 2).tolist() for t, v in d.items() if t != "gamma"}
                                        for k, d in res.items()},
               "frontal_coherence_median": {k: np.round(np.median(d["gamma"], 0), 3).tolist() for k, d in res.items()},
               "ratio_to_ring_median": {k: {t: np.round(np.median(res[k][t] / res["ring"][t], 0), 3).tolist()
                                            for t in TARGETS} for k in ("mirror", "none")},
               "ratio_to_ring_max_abs_log": {k: round(float(max(np.max(np.abs(np.log(res[k][t] / res["ring"][t])))
                                                              for t in TARGETS)), 3) for k in ("mirror", "none")}}
    (ROOT / "results" / "e1d_lower_cap.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
