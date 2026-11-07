"""Expected binaural cross-spectra in a room: direct path plus diffuse late field.

With source power 1 and an omnidirectional direct-to-reverberant ratio DRR,

    R(f, theta) = h(f, theta) h(f, theta)^H + Gamma(f) / DRR,
    Gamma(f)    = sum_k w_k h(f, phi_k) h(f, phi_k)^H ,

where Gamma is the diffuse-field cross-spectral matrix of the same head, integrated over the
measured directions with Voronoi weights w_k summing to one.
"""
from __future__ import annotations

import numpy as np


def diffuse_matrix(H: np.ndarray, w: np.ndarray):
    """Gamma11, Gamma22 (F,) and Gamma12 (F,) from DTF spectra H (D, 2, F)."""
    g11 = w @ np.abs(H[:, 0]) ** 2
    g22 = w @ np.abs(H[:, 1]) ** 2
    g12 = w @ (H[:, 0] * np.conj(H[:, 1]))
    return g11, g22, g12


def direct_spectra(H: np.ndarray):
    return np.abs(H[:, 0]) ** 2, np.abs(H[:, 1]) ** 2, H[:, 0] * np.conj(H[:, 1])


def cross_spectra(H: np.ndarray, w: np.ndarray, drr_db: float):
    """(R11, R22, R12), each (D, F), for every measured source direction."""
    R11, R22, R12 = direct_spectra(H)
    if np.isinf(drr_db):
        return R11, R22, R12
    g11, g22, g12 = diffuse_matrix(H, w)
    k = 10 ** (-drr_db / 10)
    return R11 + k * g11, R22 + k * g22, R12 + k * g12
