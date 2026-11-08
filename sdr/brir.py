"""Measured BRIRs of the Surrey real-room set (Hummersone, Cortex Mk.2 HATS, 1.5 m, ear height).

Directions are converted to the head frame of the rest of the package: azimuth positive to the
left, ear 0 left. Each response is split by a window around the direct sound, [-1, 2.5] ms about
the onset with a 1-ms raised-cosine fall, into a direct part and its complement. Cross-spectra
are formed at the full response length and summed into the bins of the HRTF grid, which keeps
the in-band coherence of arrivals that a shorter analysis would lose.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from . import hrtf as hr

ROOT = Path(__file__).resolve().parents[1] / "data" / "surrey_brirs" / "RealRoomBRIRs-master"
ROOMS = ("A", "B", "C", "D")
FS = hr.FS
PRE, HOLD, FALL = 1.0e-3, 2.5e-3, 1.0e-3


def load(name: str):
    """Head-frame azimuths (D,) in radians, ascending, and IRs (D, 2, N); name 'Anechoic' or 'A'-'D'."""
    tag = "Anechoic" if name == "Anechoic" else f"Room_{name}"
    with h5py.File(ROOT / f"UniS_{tag}_BRIR_48k.sofa", "r") as f:
        ir = np.asarray(f["Data.IR"][()], float)
        az = np.asarray(f["SourcePosition"][()], float)[:, 0]
    az = -((az + 180.0) % 360.0 - 180.0)  # SOFA set is clockwise; ear 0 is ipsilateral at 270
    i = np.argsort(az)
    return np.radians(az[i]), hr.band_limit(ir[i])


def truncate(ir: np.ndarray, margin_db: float = 3.0) -> np.ndarray:
    """Cut all responses where the 10-ms energy envelope meets the noise floor (last 5%) + margin."""
    p = (ir ** 2).mean((0, 1))
    n = len(p)
    floor = p[int(0.95 * n):].mean()
    env = np.convolve(p, np.ones(480) / 480, "same")
    above = np.where(env > floor * 10 ** (margin_db / 10))[0]
    return ir[..., :above.max() + 1] if len(above) else ir


def onsets(ir: np.ndarray, frac: float = 0.2) -> np.ndarray:
    a = np.abs(ir).max(1)
    return np.argmax(a > frac * a.max(1, keepdims=True), axis=1)


def direct_window(n: int, onset: np.ndarray) -> np.ndarray:
    """(D, n) window, 1 over [onset - PRE, onset + HOLD], falling to 0 over FALL."""
    t = (np.arange(n)[None, :] - onset[:, None]) / FS
    w = ((t >= -PRE) & (t <= HOLD)).astype(float)
    fall = (t > HOLD) & (t < HOLD + FALL)
    w[fall] = 0.5 * (1 + np.cos(np.pi * (t[fall] - HOLD) / FALL))
    return w


def precedence_gain(n: int, onset: np.ndarray, Ts: float) -> np.ndarray:
    """(D, n) amplitude gain: 1 on the direct part, sqrt(1 - exp(-t / Ts)) after it."""
    t = np.maximum((np.arange(n)[None, :] - onset[:, None]) / FS, 0.0)
    wd = direct_window(n, onset)
    return wd + (1 - wd) * np.sqrt(1 - np.exp(-t / Ts))


def cross_spectra(ir: np.ndarray):
    """(R11, R22, R12), each (D, F) on the HRTF grid (NFFT = hr.NFFT), from IRs (D, 2, N)."""
    m = max(1, int(np.ceil(ir.shape[-1] / hr.NFFT)))
    X = np.fft.rfft(ir, n=hr.NFFT * m, axis=-1)
    S = (np.abs(X[:, 0]) ** 2, np.abs(X[:, 1]) ** 2, X[:, 0] * np.conj(X[:, 1]))
    nf, F = S[0].shape[-1], hr.NFFT // 2 + 1
    lo = np.clip(np.arange(F) * m - m // 2, 0, nf)
    hi = np.clip(np.arange(F) * m - m // 2 + m, 0, nf)
    out = []
    for s in S:
        c = np.concatenate([np.zeros(s.shape[:-1] + (1,), s.dtype), np.cumsum(s, -1)], -1)
        out.append(c[:, hi] - c[:, lo])
    return tuple(out)


def ctf_power(R11: np.ndarray, R22: np.ndarray, f: np.ndarray, f1: float = 50.0, f2: float = 18000.0):
    """Common transfer function per ear (2, F) in power: log-mean over directions, 1 outside [f1, f2]."""
    c = np.stack([np.exp(np.log(R + 1e-30).mean(0)) for R in (R11, R22)])
    c[:, (f < f1) | (f > f2)] = 1.0
    return c


def equalise(R, c):
    R11, R22, R12 = R
    return R11 / c[0], R22 / c[1], R12 / np.sqrt(c[0] * c[1])
