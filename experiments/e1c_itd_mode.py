"""E1c: envelope ITD (3 kHz low-pass) against fine-structure ITD (1.5 kHz low-pass).

Direct path plus diffuse field, every 14th available SONICOM subject, rho = 0.984. For each
ITD mode: azimuth thresholds on the horizontal plane, elevation threshold at the front, the
frontal ITD share of the information, and the ITD estimate at 90 deg. The measured Surrey
rooms are recomputed with the fine-structure ITD for the E3 model comparison.
Output: results/e1c_itd_mode.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sdr import brir, cues, fisher, hrtf, room

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = hrtf.available_subjects()[::14]
DRR_DB = [np.inf, 10.0, 5.0, 0.0, -5.0, -10.0]
RHO = 0.984
MODES = ("envelope", "fine")
AZ = (0, 30, 60, 90)


def index(r, az, el):
    a = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
    e = np.degrees(np.arcsin(r[:, 2]))
    return int(np.argmin(np.abs(((a - az + 180) % 360) - 180) + np.abs(e - el)))


def threshold(G, r, e, i):
    return float(np.degrees(1 / np.sqrt(fisher.directional_information(G[i:i + 1], r[i:i + 1], e[i:i + 1])[0])))


def heads():
    out = {m: {k: [] for k in ("az", "el", "itd_share", "itd90_us")} for m in MODES}
    for sid in SUBJECTS:
        r, f, H, w = hrtf.dtf_spectra(sid)
        ea, ee = fisher.azimuth_vector(r), fisher.elevation_vector(r)
        idx = {a: index(r, a, 0) for a in AZ}
        an = room.cross_spectra(H, w, np.inf)
        for m in MODES:
            cm = cues.CueModel(f, rho_int=RHO, itd_mode=m)
            rows = {k: [] for k in out[m]}
            for drr in DRR_DB:
                c = cm.cues(room.cross_spectra(H, w, drr), an)
                G = fisher.fisher_metric(r, c.q, c.var)[0]
                J = fisher.field_jacobian(r, c.q, 6)
                i0 = idx[0]
                fam = [float(fisher.directional_information(
                    np.einsum("dmi,dm,dmj->dij", J[i0:i0 + 1, c.fam == k], 1 / c.var[i0:i0 + 1, c.fam == k],
                              J[i0:i0 + 1, c.fam == k]), r[i0:i0 + 1], ea[i0:i0 + 1])[0]) for k in range(3)]
                rows["az"].append([threshold(G, r, ea, idx[a]) for a in AZ])
                rows["el"].append(threshold(G, r, ee, i0))
                rows["itd_share"].append(fam[2] / sum(fam))
                rows["itd90_us"].append(float(c.itd_s[idx[90]] * 1e6))
            for k in rows:
                out[m][k].append(rows[k])
        print(sid, "done", flush=True)
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in out.items()}


def rooms(f):
    """Median |log| error of the two E3 models against the measured rooms, fine-structure ITD."""
    import importlib
    e3 = importlib.import_module("experiments.e3_measured")
    out = {}
    for m in MODES:
        cm = cues.CueModel(f, rho_int=RHO, itd_mode=m)
        err = {"diffuse": [], "early_diffuse": [], "measured_over_direct": []}
        for name in brir.ROOMS:
            th, res, _ = e3.room_case(name, cm, f)
            sel = np.abs(th) <= np.radians(80) + 1e-9
            t = {k: np.degrees(1 / np.sqrt(v["J"]))[sel] for k, v in res.items()}
            for k in ("diffuse", "early_diffuse"):
                err[k].append(np.log(t[k] / t["measured"]))
            err["measured_over_direct"].append(t["measured"] / t["direct"])
        out[m] = {
            "diffuse_median_abs_log_error": round(float(np.median(np.abs(np.concatenate(err["diffuse"])))), 3),
            "diffuse_median_log_bias": round(float(np.median(np.concatenate(err["diffuse"]))), 3),
            "early_diffuse_median_abs_log_error": round(float(np.median(np.abs(np.concatenate(err["early_diffuse"])))), 3),
            "measured_over_direct_median_by_room": [round(float(np.median(x)), 2) for x in err["measured_over_direct"]],
        }
    return out


def main():
    res = heads()
    f = hrtf.dtf_spectra(SUBJECTS[0])[1]
    summary = {"subjects": [int(s) for s in SUBJECTS], "drr_db": [str(d) for d in DRR_DB], "rho": RHO, "modes": {}}
    for m in MODES:
        d = res[m]
        summary["modes"][m] = {
            "az_threshold_deg_median": {f"az{a}": np.round(np.median(d["az"][:, :, j], 0), 2).tolist()
                                        for j, a in enumerate(AZ)},
            "el_threshold_deg_median_front": np.round(np.median(d["el"], 0), 2).tolist(),
            "itd_share_front_median": np.round(np.median(d["itd_share"], 0), 3).tolist(),
            "itd90_us_median": np.round(np.median(d["itd90_us"], 0), 1).tolist(),
        }
    ratio = res["fine"]["az"] / res["envelope"]["az"]
    summary["fine_over_envelope_az_threshold_median"] = {
        f"az{a}": np.round(np.median(ratio[:, :, j], 0), 3).tolist() for j, a in enumerate(AZ)}
    summary["fine_over_envelope_el_threshold_median"] = np.round(np.median(res["fine"]["el"] / res["envelope"]["el"], 0), 3).tolist()
    summary["measured_rooms"] = rooms(f)
    (ROOT / "results" / "e1c_itd_mode.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
