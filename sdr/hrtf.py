"""SONICOM HRTFs as directional transfer functions on a fine frequency grid."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from scipy import signal
from scipy.spatial import SphericalVoronoi

FS = 48000.0
NFFT = 4096
ROOT = Path(__file__).resolve().parents[1]
SOFA_ROOT = Path(os.environ.get("SDR_SONICOM_ROOT")
                 or ROOT.parent / "perceptual-hrtf-sampling" / "dependencies" / "Sonicom_HRTFs")


def sofa_path(sid: int) -> Path:
    name = f"P{sid:04d}_FreeFieldCompMinPhase_48kHz.sofa"
    p = SOFA_ROOT / f"P{sid:04d}" / "HRTF" / "HRTF" / "48kHz" / name
    return p if p.exists() else next(SOFA_ROOT.rglob(name))


def available_subjects() -> list[int]:
    return sorted(int(p.name[1:5]) for p in SOFA_ROOT.rglob("*_FreeFieldCompMinPhase_48kHz.sofa"))


def load_sofa(sid: int):
    with h5py.File(sofa_path(sid), "r") as f:
        ir = np.asarray(f["Data.IR"][()], float)[:, :2, :]
        sp = np.asarray(f["SourcePosition"][()], float)
    az, el = np.radians(sp[:, 0]), np.radians(sp[:, 1])
    r = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    return ir, r


def band_limit(ir: np.ndarray, order: int = 256, cutoff: float = 15500.0, beta: float = 8.0):
    b = signal.firwin(order + 1, cutoff / (FS / 2), window=("kaiser", beta))
    padded = np.concatenate([ir, np.zeros(ir.shape[:-1] + (order,))], axis=-1)
    return signal.lfilter(b, 1.0, padded, axis=-1)


def hrtf2dtf_log(ir: np.ndarray, f1: float = 50.0, f2: float = 18000.0, atten: float = 20.0):
    """SOFAhrtf2dtf('log'): divide by the minimum-phase log-average common transfer function."""
    n = ir.shape[-1]
    f = np.arange(n) * FS / n
    idx = (f >= f1) & (f <= f2)
    idx[n // 2 + 1:] = idx[1:n // 2][::-1]
    H = np.fft.fft(ir, axis=-1)
    logmag = np.log(np.abs(H) + np.finfo(float).eps)
    avg = logmag.mean(axis=0, keepdims=True)
    cep = np.fft.ifft(avg, axis=-1)
    cep[..., n // 2 + 1:] = 0
    cep[..., 1:n // 2] *= 2
    ctf = np.exp(avg) * np.exp(1j * np.angle(np.exp(np.fft.fft(cep, axis=-1))))
    D = H.copy()
    D[..., idx] = H[..., idx] / ctf[..., idx]
    return np.real(np.fft.ifft(D, axis=-1)) / 10 ** (atten / 20)


def quadrature_weights(r: np.ndarray) -> np.ndarray:
    """Spherical Voronoi areas of the measurement directions, normalised to sum to one."""
    sv = SphericalVoronoi(r, radius=1.0)
    w = sv.calculate_areas()
    return w / w.sum()


@lru_cache(maxsize=8)
def dtf_spectra(sid: int):
    """Directions r (D,3), frequencies f (F,), one-sided DTF spectra H (D,2,F), weights w (D,)."""
    ir, r = load_sofa(sid)
    dtf = hrtf2dtf_log(band_limit(ir))
    H = np.fft.rfft(dtf, n=NFFT, axis=-1)
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    return r, f, H, quadrature_weights(r)
