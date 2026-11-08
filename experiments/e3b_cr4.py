"""E3b: azimuth acuity from the measured BRIRs of BRAS CR4 (large auditorium) against the
diffuse late-field model.

FABIAN at MP6 (8.5, 0, 1.09) with head-above-torso orientations -44 to 44 deg in 2-deg steps,
loudspeakers LS3-LS7 on the stage about 10 m away. Turning the head by phi moves a source at
azimuth theta0 to head azimuth theta0 - phi, so each loudspeaker gives a cue field over 88 deg of
head azimuth (torso fixed). Responses are resampled from 44.1 to 48 kHz and processed as in E3:
direct window [-1, 2.5] ms, cross-spectra at full length summed into the HRTF grid, common
transfer function of the direct parts, late field after 80 ms averaged over all orientations
and loudspeakers.
Output: results/e3b_cr4.npz, results/e3b_cr4.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from sdr import brir, cues, fisher, hrtf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bras_cr4"
RHO, TS = 0.984, 5.6e-3
T_LATE = 80e-3
HALF = np.radians(20)
MP6 = np.array([8.5, 0.0, 1.09])


def load():
    meta = np.load(DATA / "CR4_BRIRs_meta.npz")
    part = np.load(DATA / "CR4_BRIRs_part00.npz")
    ir = signal.resample_poly(part["ir"].astype(np.float64), 160, 147, axis=-1)  # (M, 2, E, N)
    view = part["ListenerView"][:, 0]
    emit = meta["EmitterPosition"][:, :, 0]
    ids = meta["EmitterID"][:, 0].astype(int)
    return ir, view, emit, ids


def source_azimuths(emit):
    """Azimuth (deg, positive left) of each loudspeaker for a listener at MP6 viewing LS7 at 0 deg."""
    fwd = emit[-1, :2] - MP6[:2]
    fwd /= np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0]])
    v = emit[:, :2] - MP6[:2]
    return np.degrees(np.arctan2(v @ left, v @ fwd))


def energy(R, f):
    b = (f >= 80) & (f <= 16000)
    return (R[0][:, b] + R[1][:, b]).sum(1).real


def threshold_deg(J):
    return np.degrees(1 / np.sqrt(J))


def main():
    ir_all, view, emit, ids = load()
    az0 = source_azimuths(emit)
    f = hrtf.dtf_spectra(hrtf.available_subjects()[0])[1]
    cm = cues.CueModel(f, rho_int=RHO)
    ir_all = brir.truncate(ir_all.transpose(0, 2, 1, 3).reshape(-1, 2, ir_all.shape[-1]))
    M, E = len(view), len(ids)
    ir_all = ir_all.reshape(M, E, 2, -1)
    n = ir_all.shape[-1]

    spectra = {}
    for e in range(E):
        ir = ir_all[:, e]
        on = brir.onsets(ir)
        t = (np.arange(n)[None, :] - on[:, None]) / brir.FS
        wd = brir.direct_window(n, on)[:, None, :]
        gain = brir.precedence_gain(n, on, TS)[:, None, :]
        late = (t > T_LATE)[:, None, :].astype(float)
        spectra[e] = {k: brir.cross_spectra(x) for k, x in {
            "direct": ir * wd, "full": ir, "rev": ir * (1 - wd), "full_w": ir * gain, "rev_w": ir * gain * (1 - wd),
            "early": ir * (1 - late), "late": ir * late, "early_w": ir * gain * (1 - late),
            "late_w": ir * gain * late}.items()}
        print("LS", ids[e], "spectra done", flush=True)

    d11 = np.concatenate([spectra[e]["direct"][0] for e in range(E)])
    d22 = np.concatenate([spectra[e]["direct"][1] for e in range(E)])
    c = brir.ctf_power(d11, d22, f)
    for e in range(E):
        spectra[e] = {k: brir.equalise(v, c) for k, v in spectra[e].items()}
    G = tuple(np.concatenate([spectra[e]["late"][i] for e in range(E)]).mean(0) for i in range(3))
    eG = energy(tuple(g[None] for g in G), f)[0]
    wl = cm.Lc * cm.bw

    save, summary = {"view_deg": view, "az0_deg": az0, "emitter_id": ids}, {"rho": RHO, "Ts_ms": TS * 1e3,
                                                                            "source_azimuth_deg": dict(zip(map(str, ids), np.round(az0, 2).tolist())),
                                                                            "late_coherence_summary": round(float((cm.coherence(*cm.band_spectra(*(g[None] for g in G)))[0] * wl).sum() / wl.sum()), 3),
                                                                            "loudspeakers": {}}
    pooled = {k: [] for k in ("diffuse", "early_diffuse", "diffuse_w", "early_diffuse_w")}
    for e in range(E):
        S = spectra[e]
        th = np.radians(az0[e] - view)                      # head azimuth of the source
        order = np.argsort(th)
        S = {k: tuple(x[order] for x in v) for k, v in S.items()}
        th = th[order]
        eta = {k: energy(S[k], f) / eG for k in ("rev", "rev_w", "late", "late_w")}
        add = lambda R, h: tuple(x + h[:, None] * g[None, :] for x, g in zip(R, G))
        models = {"direct": S["direct"], "measured": S["full"], "diffuse": add(S["direct"], eta["rev"]),
                  "early_diffuse": add(S["early"], eta["late"]), "measured_w": S["full_w"],
                  "diffuse_w": add(S["direct"], eta["rev_w"]), "early_diffuse_w": add(S["early_w"], eta["late_w"])}
        thr, gam = {}, {}
        for k, R in models.items():
            cs = cm.cues(R, S["direct"])
            thr[k] = threshold_deg(fisher.arc_information(th, cs.q, cs.var, HALF))
            gam[k] = cs.gamma_itd
            save[f"LS{ids[e]}_{k}_thr"] = thr[k]
        save[f"LS{ids[e]}_theta_deg"] = np.degrees(th)
        sel = np.abs(th - th.mean()) <= np.radians(24) + 1e-9      # interior of the arc
        front = int(np.argmin(np.abs(th)))
        drr = 10 * np.log10(energy(S["direct"], f) / energy(S["rev"], f))
        for k, ref in (("diffuse", "measured"), ("early_diffuse", "measured"), ("diffuse_w", "measured_w"),
                       ("early_diffuse_w", "measured_w")):
            pooled[k].append(np.log(thr[k][sel] / thr[ref][sel]))
        summary["loudspeakers"][f"LS{ids[e]}"] = {
            "drr_db_median": round(float(np.median(drr)), 2),
            "late_energy_fraction_median": round(float(np.median(eta["late"] / eta["rev"])), 3),
            "threshold_deg_front": {k: round(float(v[front]), 2) for k, v in thr.items()},
            "threshold_deg_median_interior": {k: round(float(np.median(v[sel])), 2) for k, v in thr.items()},
            "coherence_front": {k: round(float(v[front]), 3) for k, v in gam.items()},
        }
    summary["pooled"] = {k: {"median_abs_log_error": round(float(np.median(np.abs(np.concatenate(v)))), 3),
                             "median_log_bias": round(float(np.median(np.concatenate(v))), 3),
                             "p90_abs_log_error": round(float(np.percentile(np.abs(np.concatenate(v)), 90)), 3)}
                         for k, v in pooled.items()}
    np.savez_compressed(ROOT / "results" / "e3b_cr4.npz", **save)
    (ROOT / "results" / "e3b_cr4.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
