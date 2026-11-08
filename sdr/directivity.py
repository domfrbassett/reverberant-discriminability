"""Loudspeaker directivity from the BRAS third-octave magnitude/phase files (front-pole grid).

Rows are directions labelled PxxxTyyy: T is the angle from the on-axis direction (+x), P the
rotation about the axis, 0 towards +z and 90 towards +y. Octave-band powers sum the three
third-octave bands of each octave from 63 Hz to 8 kHz and are given relative to on-axis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OCTAVES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)


def read_mps(path: str | Path):
    """Unit vectors (N, 3) in the loudspeaker frame, third-octave centres (B,), complex spectra (N, B)."""
    with open(path) as fh:
        head = [x.strip() for x in fh.readline().split(",")]
        fc = np.array([float(x) for x in head[1:] if x])
        labels, rows = [], []
        for line in fh:
            parts = [x.strip() for x in line.split(",") if x.strip()]
            labels.append(parts[0])
            rows.append([complex(p.replace(" ", "").replace("i", "j")) for p in parts[1:]])
    P = np.radians([int(s[1:4]) for s in labels])
    T = np.radians([int(s[5:8]) for s in labels])
    u = np.stack([np.cos(T), np.sin(T) * np.sin(P), np.sin(T) * np.cos(P)], 1)
    return u, fc, np.array(rows)


def octave_power(fc: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Octave-band power (N, 8) relative to on-axis (row 0)."""
    p = np.abs(spec) ** 2
    out = np.stack([p[:, np.argmin(np.abs(np.log2(fc / c))) - 1:np.argmin(np.abs(np.log2(fc / c))) + 2].sum(1)
                    for c in OCTAVES], 1)
    return out / out[0]


def sphere_weights(u: np.ndarray) -> np.ndarray:
    """Quadrature weights of the equal-angle front-pole grid, summing to one."""
    T = np.arccos(np.clip(u[:, 0], -1, 1))
    w = np.sin(T)
    pole = np.isclose(T, 0) | np.isclose(T, np.pi)
    w[pole] = (1 - np.cos(np.radians(0.5))) * 180
    return w / w.sum()


class Directivity:
    def __init__(self, path: str | Path, axis=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
        u, fc, spec = read_mps(path)
        self.u, self.power = u, octave_power(fc, spec)
        self.w = sphere_weights(u)
        x = np.asarray(axis, float) / np.linalg.norm(axis)
        z = np.asarray(up, float) - np.dot(up, x) * x
        z /= np.linalg.norm(z)
        self.frame = np.stack([x, np.cross(z, x), z])     # rows: loudspeaker x, y, z in room coordinates

    def mean_power(self) -> np.ndarray:
        """Power averaged over the sphere relative to on-axis (8,): 1 / directivity factor."""
        return self.w @ self.power

    def __call__(self, d_room: np.ndarray) -> np.ndarray:
        """Octave-band power relative to on-axis (K, 8) for room-frame unit vectors (K, 3), nearest grid point."""
        d = d_room @ self.frame.T
        idx = np.argmax(d @ self.u.T, axis=1)
        return self.power[idx]
