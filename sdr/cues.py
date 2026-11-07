"""Expected localisation cues and their external variance from binaural cross-spectra.

Cross-spectra are given per direction and frequency as R11, R22 (real) and R12 (complex),
shape (D, F). Cues follow the Barumerli (2023) observer: positive spectral gradients of
the band levels of each ear, broadband ILD, and the envelope ITD below 3 kHz on the
Reijniers JND scale. Internal noise is an internal interaural decorrelation rho_int, so the
variance of each interaural cue scales with the delta-method variance of its estimator at
coherence rho_int * gamma, normalised to the anechoic value at the same direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import auditory as aud

SIGMA_MON, SIGMA_ILD, SIGMA_ITD = 1.25, 1.0, 0.569
LN10 = np.log(10.0)


@dataclass
class CueSet:
    q: np.ndarray        # (D, M) cue values
    fam: np.ndarray      # (M,) 0 monaural, 1 ILD, 2 ITD
    var: np.ndarray      # (D, M) total cue variance
    itd_s: np.ndarray    # (D,) ITD in seconds
    gamma_itd: np.ndarray  # (D,) coherence summary over the ITD band


class CueModel:
    def __init__(self, f: np.ndarray, rho_int: float = 0.98, tau_max: float = 1.0e-3,
                 tau_step: float = 2.0e-6):
        self.f = f
        self.fc = aud.erb_centres()
        self.G = aud.gammatone_power(f, self.fc)
        self.dg = aud.gradient_step(self.fc)
        self.W = aud.broadband_weight(f)
        self.L = aud.lowpass_power(f)
        self.rho = rho_int
        band = (f > 0) & (self.L > 1e-6)
        self.itd_bins = np.where(band)[0]
        self.tau = np.arange(-tau_max, tau_max + tau_step / 2, tau_step)
        self.kernel = np.exp(2j * np.pi * np.outer(f[self.itd_bins], self.tau))  # (Fi, T)

    # ---- expected cues ----

    def monaural(self, R11, R22):
        lev = np.stack([5 * np.log10(R @ self.G.T) for R in (R11, R22)], 1)   # (D, 2, B)
        return np.maximum(lev[:, :, self.dg:] - lev[:, :, :-self.dg], 0)

    def ild(self, R11, R22):
        return 10 * np.log10((R11 @ self.W) / (R22 @ self.W))

    def itd(self, R12):
        """Peak of the envelope of the low-passed interaural cross-correlation, in seconds."""
        c = np.abs((R12[:, self.itd_bins] * self.L[self.itd_bins]) @ self.kernel)  # (D, T)
        k = np.clip(np.argmax(c, axis=1), 1, len(self.tau) - 2)
        i = np.arange(len(k))
        y0, y1, y2 = c[i, k - 1], c[i, k], c[i, k + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 0, 0.5 * (y0 - y2) / den, 0.0)
        return self.tau[k] + off * (self.tau[1] - self.tau[0])

    # ---- external variance multipliers ----

    def coherence(self, R11, R22, R12):
        return np.abs(R12) / np.sqrt(R11 * R22)

    def itd_information(self, gamma):
        """Knapp-Carter information of a delay estimate, sum_f L w^2 g^2/(1-g^2), g = rho*gamma."""
        g2 = (self.rho * gamma[:, self.itd_bins]) ** 2
        w2 = (2 * np.pi * self.f[self.itd_bins]) ** 2
        return (self.L[self.itd_bins] * w2 * g2 / (1 - g2)) @ np.ones(len(self.itd_bins))

    def ild_variance(self, R11, R22, R12):
        """Delta-method variance of the broadband ILD estimator, up to a common factor."""
        S1, S2 = (R11 @ self.W)[:, None], (R22 @ self.W)[:, None]
        v = (R11 / S1) ** 2 + (R22 / S2) ** 2 - 2 * self.rho ** 2 * np.abs(R12) ** 2 / (S1 * S2)
        return v @ self.W

    # ---- full cue set ----

    def cues(self, R, R_anechoic=None) -> CueSet:
        """R = (R11, R22, R12); R_anechoic sets the reference of the variance multipliers."""
        R11, R22, R12 = R
        A11, A22, A12 = R if R_anechoic is None else R_anechoic
        pge = self.monaural(R11, R22)
        mon = np.concatenate([pge[:, 0, :], pge[:, 1, :]], axis=1)
        tau = self.itd(R12)
        q = np.concatenate([mon, self.ild(R11, R22)[:, None], aud.transform_itd(tau)[:, None]], 1)
        fam = np.array([0] * mon.shape[1] + [1, 2])

        gam = self.coherence(R11, R22, R12)
        gam_an = self.coherence(A11, A22, A12)
        m_itd = self.itd_information(gam_an) / self.itd_information(gam)
        m_ild = self.ild_variance(R11, R22, R12) / self.ild_variance(A11, A22, A12)
        var = np.empty_like(q)
        var[:, fam == 0] = SIGMA_MON ** 2
        var[:, -2] = SIGMA_ILD ** 2 * m_ild
        var[:, -1] = SIGMA_ITD ** 2 * m_itd
        wl = self.L[self.itd_bins]
        g_sum = (gam[:, self.itd_bins] * wl).sum(1) / wl.sum()
        return CueSet(q, fam, var, tau, g_sum)
