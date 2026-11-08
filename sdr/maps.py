"""Acuity at a seat from a fixed set of reflections, for room maps.

A seat is described by its arrivals relative to the direct sound: delay, head-frame direction
and amplitude (broadband, or per octave band), and by the energy and onset of the late field.
The cue field over source direction moves the direct sound and holds the reflections fixed, so
one set of paths per seat, from a shoebox model or a geometric-acoustics export, suffices.

Path CSV format, one row per arrival including the direct sound (delay 0):
    delay_s, ux, uy, uz, amp | e63 e125 e250 e500 e1000 e2000 e4000 e8000
with (ux, uy, uz) the unit vector from the receiver towards the arriving sound in room
coordinates, amp the linear pressure relative to the direct sound, and e* band energies relative
to the direct sound's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ism, room

OCTAVES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)


@dataclass
class Seat:
    delay: np.ndarray      # (K,) s after the direct sound, reflections only
    dirs: np.ndarray       # (K, 3) head-frame unit vectors
    amp: np.ndarray        # (K,) or (K, F) linear pressure relative to the direct sound
    e_late: float | np.ndarray      # late-field energy relative to the direct sound, scalar or (F,)
    t_late: float                   # onset of the late field after the direct sound (s)
    e_late_weighted: float | np.ndarray | None = None  # late energy under the precedence weighting


def head_frame(u: np.ndarray, view: np.ndarray) -> np.ndarray:
    """Room-frame unit vectors u (K, 3) in the frame of a listener facing horizontal direction view."""
    x = np.array([view[0], view[1], 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return u @ np.stack([x, y, z], 1)


def from_shoebox(box: ism.Shoebox, listener: np.ndarray, source: np.ndarray, view: np.ndarray) -> Seat:
    pos, orders = box.images(source)
    vec = pos - listener
    dist = np.linalg.norm(vec, axis=1)
    amp = box.beta ** orders * dist[0] / dist
    e_rev = 16 * np.pi * dist[0] ** 2 / box.room_constant()
    e_late = ism.late_energy(box, e_rev, (amp[1:] ** 2).sum())
    return Seat(delay=(dist[1:] - dist[0]) / ism.C, dirs=head_frame(vec[1:] / dist[1:, None], view),
                amp=amp[1:], e_late=e_late, t_late=(box.order + 1) * min(box.L) / ism.C)


def from_paths(table: np.ndarray, f: np.ndarray, view: np.ndarray, e_late: float, t_late: float) -> Seat:
    """Seat from a path table (rows as in the module docstring, direct sound first)."""
    t = np.asarray(table, float)
    order = np.argsort(t[:, 0], kind="stable")
    t = t[order]
    u = t[:, 1:4] / np.linalg.norm(t[:, 1:4], axis=1, keepdims=True)
    if t.shape[1] == 5:
        amp = t[1:, 4]
    else:
        lf = np.log(np.clip(f, OCTAVES[0], OCTAVES[-1]))
        amp = np.sqrt(np.stack([np.interp(lf, np.log(OCTAVES), e) for e in t[1:, 4:12]]))
    return Seat(delay=t[1:, 0] - t[0, 0], dirs=head_frame(u[1:], view), amp=amp, e_late=e_late, t_late=t_late)


def cross_spectra(seat: Seat, dirs: np.ndarray, interp: ism.HrtfInterpolator, Ts_ms: float | None = None):
    """(R11, R22, R12), each (D, F), for the direct sound from each of dirs and the seat's reflections."""
    f = interp.f
    Ts = None if Ts_ms is None else Ts_ms * 1e-3
    w = ism.precedence_weight(seat.delay, Ts)
    a = seat.amp if seat.amp.ndim == 2 else seat.amp[:, None] * np.ones_like(f)
    Hk = interp(seat.dirs)
    X = np.einsum("kf,kef->ef", a * np.sqrt(w)[:, None], Hk * np.exp(-2j * np.pi * f * seat.delay[:, None, None]))
    if Ts is None:
        k = seat.e_late
    elif seat.e_late_weighted is not None:
        k = seat.e_late_weighted
    else:
        k = seat.e_late * (1.0 - np.exp(-seat.t_late / Ts))
    g11, g22, g12 = room.diffuse_matrix(interp.H, interp.w)
    Hd = interp(dirs)
    D0, D1 = Hd[:, 0] + X[0], Hd[:, 1] + X[1]
    return np.abs(D0) ** 2 + k * g11, np.abs(D1) ** 2 + k * g22, D0 * np.conj(D1) + k * g12
