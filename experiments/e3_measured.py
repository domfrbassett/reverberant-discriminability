"""E3: azimuth acuity from measured BRIRs against the diffuse late-field model.

Surrey real rooms A-D (Cortex Mk.2 HATS, frontal horizontal plane, 5 deg steps, 1.5 m). For each
room and direction the direct part of the measured response is the anechoic reference; the
measured response is compared with models that replace part of it by the room's own late field,
averaged over directions, at the measured energy:

    measured        full response
    diffuse         direct part + eta Gamma_late, eta = energy outside the direct window
    early+diffuse   response to 80 ms + eta Gamma_late, eta = energy after 80 ms

with Gamma_late the cross-spectral matrix after 80 ms averaged over the 37 directions. The
weighted variants apply the precedence recovery (T_s = 5.6 ms) to the response and to eta.
Spectra are equalised by each room's common transfer function of the direct parts. Information
along azimuth is computed by local quadratic regression over +-20 deg, checked against the
order-6 SH metric on the SONICOM horizontal plane.
Output: results/e3_measured.npz, results/e3_measured.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sdr import brir, cues, fisher, hrtf, room

OUT = Path(__file__).resolve().parents[1] / "results"
RHO, TS = 0.984, 5.6e-3
T_LATE = 80e-3
REPORT = np.radians(80)
CHECK_SUBJECTS = hrtf.available_subjects()[::14]
CHECK_DRR = [np.inf, 10.0, 5.0, 0.0]


def energy(R, f):
    b = (f >= 80) & (f <= 16000)
    return (R[0][:, b] + R[1][:, b]).sum(1).real


def add(R, eta, G):
    return tuple(x + eta[:, None] * g[None, :] for x, g in zip(R, G))


def threshold_deg(J):
    return np.degrees(1 / np.sqrt(J))


def room_case(name, cm, f):
    th, ir = brir.load(name)
    ir = brir.truncate(ir)
    n = ir.shape[-1]
    on = brir.onsets(ir)
    t = (np.arange(n)[None, :] - on[:, None]) / brir.FS
    wd = brir.direct_window(n, on)[:, None, :]
    gain = brir.precedence_gain(n, on, TS)[:, None, :]
    late = (t > T_LATE)[:, None, :].astype(float)

    S = {k: brir.cross_spectra(x) for k, x in {
        "direct": ir * wd, "full": ir, "rev": ir * (1 - wd), "full_w": ir * gain,
        "rev_w": ir * gain * (1 - wd), "early": ir * (1 - late), "late": ir * late,
        "late_w": ir * gain * late, "early_w": ir * gain * (1 - late)}.items()}
    c = brir.ctf_power(S["direct"][0], S["direct"][1], f)
    S = {k: brir.equalise(v, c) for k, v in S.items()}
    G = tuple(x.mean(0) for x in S["late"])
    eG = energy(tuple(g[None] for g in G), f)[0]
    eta = {k: energy(S[k], f) / eG for k in ("rev", "rev_w", "late", "late_w")}

    models = {
        "direct": S["direct"],
        "measured": S["full"],
        "diffuse": add(S["direct"], eta["rev"], G),
        "early_diffuse": add(S["early"], eta["late"], G),
        "measured_w": S["full_w"],
        "diffuse_w": add(S["direct"], eta["rev_w"], G),
        "early_diffuse_w": add(S["early_w"], eta["late_w"], G),
    }
    out = {}
    for k, R in models.items():
        cs = cm.cues(R, S["direct"])
        out[k] = dict(J=fisher.arc_information(th, cs.q, cs.var), gamma=cs.gamma_itd,
                      ild=cs.q[:, -2], itd_us=cs.itd_s * 1e6, var_ild=cs.var[:, -2], var_itd=cs.var[:, -1])

    Bl = cm.band_spectra(*S["late"])
    wl = cm.Lc * cm.bw
    g_late = (cm.coherence(*Bl) * wl).sum(1) / wl.sum()
    g_late_avg = cm.coherence(*cm.band_spectra(*(g[None] for g in G)))[0]
    lev_late = 10 * np.log10(energy(S["late"], f))
    drr = 10 * np.log10(energy(S["direct"], f) / energy(S["rev"], f))
    tw = np.arange(n) / brir.FS
    p = (ir ** 2).sum(1)
    c50 = 10 * np.log10(np.array([p[i, tw - on[i] / brir.FS <= 0.05].sum() / p[i, tw - on[i] / brir.FS > 0.05].sum()
                                  for i in range(len(th))]))
    return th, out, dict(drr_db=drr, c50_db=c50, gamma_late=g_late, gamma_late_avg_band=g_late_avg, late_level_db=lev_late - lev_late.mean(),
                         eta_late_frac=eta["late"] / eta["rev"], length_ms=n / brir.FS * 1e3)


def anechoic_case(cm, f):
    th, ir = brir.load("Anechoic")
    on = brir.onsets(ir)
    R = brir.cross_spectra(ir * brir.direct_window(ir.shape[-1], on)[:, None, :])
    R = brir.equalise(R, brir.ctf_power(R[0], R[1], f))
    cs = cm.cues(R, R)
    return th, fisher.arc_information(th, cs.q, cs.var)


def estimator_check():
    """Azimuth thresholds on the SONICOM frontal horizontal arc: 1-D estimator against SH metric.

    Also returns each head's diffuse-field coherence per auditory band.
    """
    rows, coh = [], []
    for sid in CHECK_SUBJECTS:
        r, f, H, w = hrtf.dtf_spectra(sid)
        el = np.degrees(np.arcsin(r[:, 2]))
        az = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
        m = np.isclose(el, 0, atol=0.5) & (np.abs(az) <= 90.1)
        idx = np.where(m)[0][np.argsort(az[m])]
        th = np.radians(az[idx])
        sel = np.abs(th) <= REPORT + 1e-9
        cm = cues.CueModel(f, rho_int=RHO)
        an = room.cross_spectra(H, w, np.inf)
        coh.append(cm.coherence(*cm.band_spectra(*(g[None] for g in room.diffuse_matrix(H, w))))[0])
        res = []
        for drr in CHECK_DRR:
            cs = cm.cues(room.cross_spectra(H, w, drr), an)
            G = fisher.fisher_metric(r, cs.q, cs.var)[0]
            j_sh = fisher.directional_information(G, r, fisher.azimuth_vector(r))[idx]
            j_arc = fisher.arc_information(th, cs.q[idx], cs.var[idx])
            res.append((threshold_deg(j_sh)[sel], threshold_deg(j_arc)[sel]))
        res = np.array(res)                                   # (drr, 2, D)
        rows.append(res)
    return np.array(rows), np.array(coh)                      # (S, drr, 2, D), (S, B)


def main():
    f = hrtf.dtf_spectra(hrtf.available_subjects()[0])[1]
    cm = cues.CueModel(f, rho_int=RHO)
    save, summary = {}, {"rho": RHO, "Ts_ms": TS * 1e3, "rooms": {}}
    for name in brir.ROOMS:
        th, out, prm = room_case(name, cm, f)
        sel = np.abs(th) <= REPORT + 1e-9
        save["theta"] = th
        for k, v in out.items():
            for q, x in v.items():
                save[f"{name}_{k}_{q}"] = x
        for q, x in prm.items():
            save[f"{name}_{q}"] = x
        thr = {k: threshold_deg(v["J"]) for k, v in out.items()}
        loss = {k: 0.5 * np.log(out["direct"]["J"] / v["J"]) for k, v in out.items()}
        deg = np.round(np.degrees(th)).astype(int)
        at = lambda x, a: round(float(x[deg == a][0]), 2)
        summary["rooms"][name] = {
            "length_ms": round(prm["length_ms"], 1),
            "drr_db_median": round(float(np.median(prm["drr_db"])), 2),
            "drr_db_at": {a: round(float(prm["drr_db"][deg == a][0]), 2) for a in (0, 45, -45, 90, -90)},
            "c50_db_median": round(float(np.median(prm["c50_db"])), 2),
            "late_coherence_range": [round(float(prm["gamma_late"][sel].min()), 3), round(float(prm["gamma_late"][sel].max()), 3)],
            "late_level_range_db": [round(float(prm["late_level_db"][sel].min()), 2), round(float(prm["late_level_db"][sel].max()), 2)],
            "late_energy_fraction_median": round(float(np.median(prm["eta_late_frac"])), 3),
            "threshold_deg": {k: {a: at(v, a) for a in (0, 30, -30, 60, -60)} for k, v in thr.items()},
            "threshold_deg_median": {k: round(float(np.median(v[sel])), 2) for k, v in thr.items()},
            "loss_median": {k: round(float(np.median(v[sel])), 3) for k, v in loss.items()},
            "model_error_median_abs_log_threshold": {
                k: round(float(np.median(np.abs(np.log(thr[k][sel] / thr[ref][sel])))), 3)
                for k, ref in (("diffuse", "measured"), ("early_diffuse", "measured"),
                               ("diffuse_w", "measured_w"), ("early_diffuse_w", "measured_w"))},
            "model_bias_median_log_threshold": {
                k: round(float(np.median(np.log(thr[k][sel] / thr[ref][sel]))), 3)
                for k, ref in (("diffuse", "measured"), ("early_diffuse", "measured"),
                               ("diffuse_w", "measured_w"), ("early_diffuse_w", "measured_w"))},
            "coherence_median": {k: round(float(np.median(v["gamma"][sel])), 3) for k, v in out.items()},
        }
        print(name, "done", flush=True)

    tha, Ja = anechoic_case(cm, f)
    sel = np.abs(tha) <= REPORT + 1e-9
    save["anechoic_J"] = Ja
    ref = np.array([save[f"{n}_direct_J"] for n in brir.ROOMS])
    summary["pseudo_anechoic"] = {
        "threshold_deg_median": round(float(np.median(threshold_deg(Ja)[sel])), 2),
        "room_direct_threshold_deg_median": {n: round(float(np.median(threshold_deg(r)[sel])), 2)
                                             for n, r in zip(brir.ROOMS, ref)},
        "median_abs_log_ratio_room_direct_vs_anechoic": [round(float(np.median(np.abs(0.5 * np.log(Ja[sel] / r[sel])))), 3)
                                                         for r in ref],
    }

    chk, coh = estimator_check()
    save["check"] = chk
    save["sonicom_diffuse_coherence"] = coh
    save["fcc"] = cm.fcc
    wl = cm.Lc * cm.bw
    summary["late_field_coherence_summary"] = {
        "sonicom_diffuse_median": round(float(np.median((coh * wl).sum(1) / wl.sum())), 3),
        **{n: round(float((save[f"{n}_gamma_late_avg_band"] * wl).sum() / wl.sum()), 3) for n in brir.ROOMS}}
    err = np.abs(np.log(chk[:, :, 1] / chk[:, :, 0]))
    ratio_err = np.abs(np.log((chk[:, 1:, 1] / chk[:, :1, 1]) / (chk[:, 1:, 0] / chk[:, :1, 0])))
    summary["estimator_check"] = {
        "subjects": [int(s) for s in CHECK_SUBJECTS],
        "drr_db": [str(d) for d in CHECK_DRR],
        "median_abs_log_threshold_error": np.round(np.median(err, (0, 2)), 3).tolist(),
        "median_abs_log_loss_error": np.round(np.median(ratio_err, (0, 2)), 3).tolist(),
        "median_log_threshold_bias": np.round(np.median(np.log(chk[:, :, 1] / chk[:, :, 0]), (0, 2)), 3).tolist(),
    }
    np.savez_compressed(OUT / "e3_measured.npz", **save)
    (OUT / "e3_measured.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
