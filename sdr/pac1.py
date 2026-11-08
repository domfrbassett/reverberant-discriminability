"""Seats from a Pachyderm results file converted by data/convert_pac1.py.

Per receiver: the direct sound sets the reference; every image-source path becomes a reflection
with its delay, arrival direction and octave-band amplitude; the ray-traced octave energy
histogram becomes the late field, its precedence-weighted energy integrated over its own time
course. Octave values are interpolated in log frequency onto the HRTF grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import maps

OCTAVES = maps.OCTAVES


def load(path: str | Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def octave_to_grid(e: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Octave-band energies (..., 8) to the frequency grid (..., F), linear in log f, held at the ends."""
    lf = np.log(np.clip(f, OCTAVES[0], OCTAVES[-1]))
    lo = np.log(OCTAVES)
    flat = e.reshape(-1, 8)
    return np.stack([np.interp(lf, lo, row) for row in flat]).reshape(e.shape[:-1] + (len(f),))


def seats(data: dict, f: np.ndarray, src: int = 0, Ts: float = 5.6e-3, view: np.ndarray | None = None):
    """Seats (list), receiver positions (R, 3) and direct-sound validity (R,) for source index src.

    Listeners face the source unless view (3,) gives a common horizontal viewing direction.
    """
    S = data["sources"][src]
    R = data["receivers"]
    td = data[f"src{src}_direct_time"]
    Ed = data[f"src{src}_direct_energy"]
    rec = data[f"src{src}_is_rec"]
    t = data[f"src{src}_is_time"]
    e = data[f"src{src}_is_energy"]
    u = data[f"src{src}_is_dir"]
    hist = data[f"src{src}_ray_hist"]
    tb = (np.arange(hist.shape[-1]) + 0.5) * float(data[f"src{src}_ray_bin_s"])
    out = []
    for q in range(len(R)):
        v = (S - R[q]) if view is None else view
        m = rec == q
        Edq = np.maximum(Ed[q], 1e-30)
        amp = np.sqrt(octave_to_grid(e[m] / Edq, f)) if m.any() else np.zeros((0, len(f)))
        tau = tb - td[q]
        w = 1.0 - np.exp(-np.maximum(tau, 0.0) / Ts)
        late = octave_to_grid(hist[q].sum(-1) / Edq, f)
        late_w = octave_to_grid((hist[q] * w).sum(-1) / Edq, f)
        onset = float(tau[np.argmax(hist[q].sum(0) > 0)]) if hist[q].any() else 0.0
        out.append(maps.Seat(delay=t[m] - td[q], dirs=maps.head_frame(u[m], v), amp=amp,
                             e_late=late, t_late=max(onset, 0.0), e_late_weighted=late_w))
    return out, R, data[f"src{src}_direct_valid"]
