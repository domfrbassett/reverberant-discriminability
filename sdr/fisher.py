"""Fisher metric over source direction from cue fields sampled on the sphere.

Cue means and log-variances are fitted with real spherical harmonics and differentiated
along an orthonormal tangent basis. For independent Gaussian cues with direction-dependent
variance the Fisher information is

    G_ij = sum_m d_i mu_m d_j mu_m / v_m + 1/2 sum_m d_i log v_m d_j log v_m .
"""
from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y


def real_sh(r: np.ndarray, L: int) -> np.ndarray:
    theta = np.arccos(np.clip(r[:, 2], -1, 1))
    phi = np.arctan2(r[:, 1], r[:, 0])
    cols = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m < 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.imag)
            elif m == 0:
                cols.append(Y.real)
            else:
                cols.append(np.sqrt(2) * (-1) ** m * Y.real)
    return np.stack(cols, 1)


def tangent_basis(r: np.ndarray):
    aux = np.where(np.abs(r[:, 2:3]) > 0.9, np.array([[0, 1, 0]]), np.array([[0, 0, 1]]))
    u1 = np.cross(aux, r)
    u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
    return u1, np.cross(r, u1)


def tangent_jacobian(coef: np.ndarray, r: np.ndarray, L: int, h: float = 1e-4) -> np.ndarray:
    """Derivatives (D, M, 2) of an SH field along the tangent basis, per radian."""
    out = []
    for u in tangent_basis(r):
        rp, rm = r + h * u, r - h * u
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        rm /= np.linalg.norm(rm, axis=1, keepdims=True)
        out.append((real_sh(rp, L) @ coef - real_sh(rm, L) @ coef) / (2 * np.arctan(h)))
    return np.stack(out, -1)


def fit_sh(r: np.ndarray, q: np.ndarray, L: int) -> np.ndarray:
    return np.linalg.lstsq(real_sh(r, L), q, rcond=None)[0]


def field_jacobian(r: np.ndarray, q: np.ndarray, L: int) -> np.ndarray:
    return tangent_jacobian(fit_sh(r, q, L), r, L)


def fisher_metric(r: np.ndarray, q: np.ndarray, var: np.ndarray, L: int = 6,
                  variance_term: bool = True, var_floor: float = 1e-12):
    """Return G (D,2,2) and its mean and variance parts."""
    J = field_jacobian(r, q, L)
    Gm = np.einsum("dmi,dm,dmj->dij", J, 1.0 / np.maximum(var, var_floor), J)
    Gv = np.zeros_like(Gm)
    if variance_term:
        lv = np.log(np.maximum(var, var_floor))
        moving = np.ptp(lv, axis=0) > 1e-9
        if moving.any():
            K = field_jacobian(r, lv[:, moving], L)
            Gv = 0.5 * np.einsum("dmi,dmj->dij", K, K)
    return Gm + Gv, Gm, Gv


def directional_information(G: np.ndarray, r: np.ndarray, e: np.ndarray) -> np.ndarray:
    """e^T G e for unit tangent vectors e (D,3)."""
    u1, u2 = tangent_basis(r)
    c = np.stack([(e * u1).sum(1), (e * u2).sum(1)], 1)
    return np.einsum("di,dij,dj->d", c, G, c)


def azimuth_vector(r: np.ndarray) -> np.ndarray:
    az = np.arctan2(r[:, 1], r[:, 0])
    return np.stack([-np.sin(az), np.cos(az), np.zeros_like(az)], 1)


def elevation_vector(r: np.ndarray) -> np.ndarray:
    az = np.arctan2(r[:, 1], r[:, 0])
    el = np.arcsin(np.clip(r[:, 2], -1, 1))
    return np.stack([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)], 1)
