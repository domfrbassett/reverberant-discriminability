"""E0: the spectral-domain anechoic metric against the time-domain metric of paper 1.

Needs the paper-1 cue pipeline (perceptual-hrtf-sampling/design_metric/cue_pipeline.py).
Output: results/e0_anechoic.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from sdr import cues, fisher, hrtf, room

ROOT = Path(__file__).resolve().parents[1]
P1 = Path(os.environ.get("SDR_PAPER1_PIPELINE") or ROOT.parent / "perceptual-hrtf-sampling" / "design_metric")
sys.path.insert(0, str(P1))
os.environ.setdefault("FISHERRAO_DATASET_ROOT", str(hrtf.SOFA_ROOT))
import cue_pipeline as cp  # noqa: E402

SUBJECTS = hrtf.available_subjects()[::20]


def airm(A, B):
    w = np.linalg.eigvalsh(np.linalg.solve(A, B))
    return np.sqrt((np.log(np.abs(w)) ** 2).sum(-1))


def main():
    rows = []
    for sid in SUBJECTS:
        r, f, H, w = hrtf.dtf_spectra(sid)
        c = cues.CueModel(f).cues(room.cross_spectra(H, w, np.inf))
        J = fisher.field_jacobian(r, c.q, 6)
        G = np.einsum("dmi,dm,dmj->dij", J, 1 / c.var, J)
        r1, _, q1, fam1 = cp.extract_cues(sid)
        J1 = cp.tangent_jacobian(cp.fit_sh(r1, q1, 6), r1, 6)
        G1 = cp.fisher_tensor(J1, fam1)
        d = np.array([airm(G[i], G1[i]) for i in range(len(r))])
        fam = {}
        for k, name in enumerate(("spectral", "ild", "itd")):
            a = np.einsum("dmi,dmi->d", J[:, c.fam == k], J[:, c.fam == k])
            b = np.einsum("dmi,dmi->d", J1[:, fam1 == k], J1[:, fam1 == k])
            fam[name] = {"gradient_energy_ratio": float(np.median(a / b)),
                         "cue_correlation": float(np.corrcoef(c.q[:, c.fam == k].ravel(),
                                                              q1[:, fam1 == k].ravel())[0, 1])}
        rows.append({"subject": sid, "airm_median": float(np.median(d)),
                     "airm_p90": float(np.percentile(d, 90)),
                     "sqrtdet_ratio_median": float(np.median(np.sqrt(np.linalg.det(G) / np.linalg.det(G1)))),
                     **fam})
        print(rows[-1], flush=True)
    (ROOT / "results" / "e0_anechoic.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
