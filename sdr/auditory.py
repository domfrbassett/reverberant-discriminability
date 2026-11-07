"""Auditory filters and cue transforms of the Barumerli (2023) observer, in the power-spectral domain."""
from __future__ import annotations

import math

import numpy as np


def freqtoerb(f):
    return 9.2645 * np.sign(f) * np.log(1 + np.abs(f) * 0.00437)


def erbtofreq(e):
    return (1 / 0.00437) * np.sign(e) * (np.exp(np.abs(e) / 9.2645) - 1)


def audfiltbw(fc):
    return 24.7 + fc / 9.265


def erb_centres(flow: float = 700.0, fhigh: float = 16000.0, bw: float = 1.0) -> np.ndarray:
    lo, hi = freqtoerb(np.array([flow, fhigh]))
    n = int(math.floor((hi - lo) / bw))
    rem = (hi - lo) - n * bw
    return erbtofreq(lo + np.arange(n + 1) * bw + rem / 2)


def gammatone_power(f: np.ndarray, fc: np.ndarray, order: int = 4) -> np.ndarray:
    """Power responses |G_b(f)|^2 of 4th-order gammatone filters (B, F), unit peak."""
    b = 1.019 * audfiltbw(fc)
    return (1 + ((f[None, :] - fc[:, None]) / b[:, None]) ** 2) ** (-order)


def gradient_step(fc: np.ndarray) -> int:
    """Band offset of the positive spectral gradient (one ERB)."""
    return int(round(np.mean(audfiltbw(fc)[1:] / np.diff(fc))))


def lowpass_power(f: np.ndarray, fcut: float = 3000.0, order: int = 10) -> np.ndarray:
    """Power response of the Butterworth low-pass before ITD extraction."""
    return 1.0 / (1.0 + (f / fcut) ** (2 * order))


def broadband_weight(f: np.ndarray) -> np.ndarray:
    """Weight of the broadband ILD: the whole band-limited DTF spectrum."""
    return np.ones_like(f)


ITD_A, ITD_B = 32.5e-6, 0.095


def transform_itd(tau):
    """Reijniers JND transform of ITD (s) to the internal scale on which sigma_ITD is constant."""
    return np.sign(tau) * (np.log(ITD_A + ITD_B * np.abs(tau)) - np.log(ITD_A)) / ITD_B


def transform_itd_slope(tau):
    return 1.0 / (ITD_A + ITD_B * np.abs(tau))
