"""Shoebox image-source model with HRTF-filtered arrivals and a statistical late field.

Room [0, Lx] x [0, Ly] x [0, Lz] with one frequency-independent absorption coefficient alpha.
The listener faces +x with ears along +y (left) at position p; source directions are given in
that head frame. Images up to order N are summed coherently, each filtered by the DTF of its
arrival direction; reverberant energy not carried by them is added as the diffuse field of the
head. Arrivals can be weighted by the precedence recovery w(D) = 1 - exp(-D / T_s), D being the
delay after the direct sound.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull

from . import hrtf as hr
from . import room

C = 343.0


@dataclass
class Shoebox:
    L: tuple[float, float, float]
    alpha: float
    order: int = 3

    @property
    def beta(self) -> float:
        return float(np.sqrt(1.0 - self.alpha))

    def surface(self) -> float:
        Lx, Ly, Lz = self.L
        return 2 * (Lx * Ly + Lx * Lz + Ly * Lz)

    def volume(self) -> float:
        return float(np.prod(self.L))

    def rt60(self) -> float:
        """Eyring reverberation time (s)."""
        return 0.161 * self.volume() / (-self.surface() * np.log(1 - self.alpha))

    def room_constant(self) -> float:
        return self.surface() * self.alpha / (1 - self.alpha)

    def images(self, src: np.ndarray):
        """Image positions (M, 3) and orders (M,) up to self.order, direct path first."""
        n = self.order
        out, orders = [], []
        rng = range(-n, n + 1)
        for i in rng:
            for j in rng:
                for k in rng:
                    pos, order = [], 0
                    for q, Lq, s in zip((i, j, k), self.L, src):
                        if q % 2 == 0:
                            pos.append(q * Lq + s)
                        else:
                            pos.append((q + 1) * Lq - s)
                        order += abs(q)
                    if order <= n:
                        out.append(pos)
                        orders.append(order)
        out, orders = np.array(out), np.array(orders)
        idx = np.argsort(orders, kind="stable")
        return out[idx], orders[idx]


class HrtfInterpolator:
    """Barycentric interpolation of DTFs on the convex hull of the measured directions.

    Each ear's spectrum is split into a delay, from the phase slope between 200 and 1500 Hz, and
    a residual; both are interpolated and recombined. A virtual vertex at the south pole carries
    the mean of the lowest measured ring, so arrivals from below it are blended towards that ring.
    """

    def __init__(self, sid: int):
        r, f, H, w = hr.dtf_spectra(sid)
        self.r, self.f, self.w = r, f, w
        band = (f >= 200) & (f <= 1500)
        ph = np.unwrap(np.angle(H[..., band]), axis=-1)
        om = 2 * np.pi * f[band]
        slope = ((ph - ph.mean(-1, keepdims=True)) * (om - om.mean())).sum(-1) / ((om - om.mean()) ** 2).sum()
        self.tau = -slope                                           # (D, 2) seconds
        self.res = H * np.exp(2j * np.pi * f * self.tau[..., None])  # delay removed
        self.H = H
        ring = np.isclose(r[:, 2], r[:, 2].min())
        self.r = np.vstack([r, [[0.0, 0.0, -1.0]]])
        self.res = np.concatenate([self.res, self.res[ring].mean(0, keepdims=True)])
        self.tau = np.concatenate([self.tau, self.tau[ring].mean(0, keepdims=True)])
        hull = ConvexHull(self.r)
        self.tri = hull.simplices
        self.eq = hull.equations

    def _bary(self, tri: np.ndarray, d: np.ndarray, t: np.ndarray, score: np.ndarray):
        P = d / score[np.arange(len(d)), t][:, None]
        A, B, Cc = self.r[tri[:, 0]], self.r[tri[:, 1]], self.r[tri[:, 2]]
        v0, v1, v2 = B - A, Cc - A, P - A
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 ** 2
        b1 = (d11 * d20 - d01 * d21) / den
        b2 = (d00 * d21 - d01 * d20) / den
        return np.stack([1 - b1 - b2, b1, b2], 1)

    def weights(self, d: np.ndarray):
        """Triangle vertex indices (n, 3) and barycentric weights (n, 3) for unit vectors d (n, 3).

        Among facets hit by the ray (ties on coplanar facets included), the one containing the
        intersection point is chosen.
        """
        n, off = self.eq[:, :3], self.eq[:, 3]
        score = (d @ n.T) / (-off)
        best = score.max(1, keepdims=True)
        cand = np.argsort(-score, axis=1)[:, :12]
        tri = np.empty((len(d), 3), int)
        wts = np.empty((len(d), 3))
        worst = np.full(len(d), -np.inf)
        for c in range(cand.shape[1]):
            t = cand[:, c]
            ok = score[np.arange(len(d)), t] >= best[:, 0] - 1e-9
            b = self._bary(self.tri[t], d, t, score)
            mn = b.min(1)
            upd = ok & (mn > worst)
            tri[upd], wts[upd], worst[upd] = self.tri[t][upd], b[upd], mn[upd]
        wts = np.clip(wts, 0, None)
        return tri, wts / wts.sum(1, keepdims=True)

    def __call__(self, d: np.ndarray) -> np.ndarray:
        """Interpolated DTFs (n, 2, F) at unit vectors d (n, 3)."""
        tri, wts = self.weights(d)
        res = np.einsum("nk,nkef->nef", wts, self.res[tri])
        tau = np.einsum("nk,nke->ne", wts, self.tau[tri])
        return res * np.exp(-2j * np.pi * self.f * tau[..., None])


def late_energy(box: Shoebox, e_rev: float, e_img: float) -> float:
    """Late-field energy: the statistical remainder, and at least the statistical energy after t0."""
    t0 = (box.order + 1) * min(box.L) / C
    return max(e_rev - e_img, e_rev * np.exp(-13.82 * t0 / box.rt60()))


def precedence_weight(delay: np.ndarray, Ts: float | None) -> np.ndarray:
    if Ts is None:
        return np.ones_like(delay)
    return 1.0 - np.exp(-delay / Ts)


def binaural_cross_spectra(box: Shoebox, listener: np.ndarray, dirs: np.ndarray, r_src: float,
                           interp: HrtfInterpolator, Ts_ms: float | None = None, late: bool = True):
    """(R11, R22, R12) each (D, F) for sources at distance r_src in head-frame directions dirs.

    Also returns per-direction DRR (dB) and the energy fraction of the late field.
    """
    f = interp.f
    g11, g22, g12 = room.diffuse_matrix(interp.H, interp.w)
    R11 = np.zeros((len(dirs), len(f)))
    R22 = np.zeros_like(R11)
    R12 = np.zeros((len(dirs), len(f)), complex)
    drr = np.zeros(len(dirs))
    Ts = None if Ts_ms is None else Ts_ms * 1e-3
    for n, u in enumerate(dirs):
        src = listener + r_src * u
        pos, orders = box.images(src)
        vec = pos - listener
        dist = np.linalg.norm(vec, axis=1)
        delay = (dist - dist[0]) / C
        amp = box.beta ** orders * dist[0] / dist
        wgt = precedence_weight(delay, Ts)
        wgt[0] = 1.0
        Hk = interp(vec / dist[:, None])
        X = np.einsum("k,kef->ef", amp * np.sqrt(wgt), Hk * np.exp(-2j * np.pi * f * delay[:, None, None]))
        e_img = (amp[1:] ** 2).sum()
        e_rev = 16 * np.pi * dist[0] ** 2 / box.room_constant()
        e_late = late_energy(box, e_rev, e_img)
        drr[n] = 10 * np.log10(1.0 / e_rev)
        k = e_late if late else 0.0
        if late and Ts is not None:
            t0 = (box.order + 1) * min(box.L) / C
            k *= 1.0 - np.exp(-t0 / Ts)
        R11[n] = np.abs(X[0]) ** 2 + k * g11
        R22[n] = np.abs(X[1]) ** 2 + k * g22
        R12[n] = X[0] * np.conj(X[1]) + k * g12
    return (R11, R22, R12), drr
