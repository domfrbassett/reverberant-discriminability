"""Convert a Pachyderm .pac1 results file to a compact .npz for sdr/maps.py.

Usage: python convert_pac1.py <file.pac1> [bin_ms]
Writes <file>.npz beside the input with, per source: direct sound per receiver (arrival time,
validity, octave energies), every image-source path (receiver, arrival time, octave energies,
unit vector from the receiver towards the last reflection point, unit vector of departure from
the source, reflection order), and the
ray-traced octave energy histograms summed into bins of bin_ms (default 1 ms). Directional
histogram channels and pressure histograms are not kept.
"""
import struct
import sys
from pathlib import Path

import numpy as np


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected end of file")
        return b

    def i16(self):
        return struct.unpack("<h", self.raw(2))[0]

    def i32(self):
        return struct.unpack("<i", self.raw(4))[0]

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def f64(self):
        return struct.unpack("<d", self.raw(8))[0]

    def boolean(self):
        return self.raw(1) != b"\x00"

    def doubles(self, n):
        return np.frombuffer(self.raw(8 * n), "<f8")

    def string(self):
        n, shift = 0, 0
        while True:
            b = self.raw(1)[0]
            n |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                break
        return self.raw(n).decode("utf-8")


def version_number(v):
    return float(v[:3])


def read_direct(r, n_rec, v):
    if v < 2.0:
        r.doubles(8)
    src_type = r.string() if v >= 1.1 else "Geodesic Source"
    swl = r.doubles(8).copy() if v >= 1.1 else np.full(8, 120.0)
    delay = r.f64() if v >= 2.0 else 0.0
    time = np.zeros(n_rec)
    valid = np.zeros(n_rec, bool)
    energy = np.zeros((n_rec, 8))
    for q in range(n_rec):
        ns = r.i32()
        valid[q] = r.boolean()
        time[q] = r.f64()
        for _ in range(ns):
            energy[q] += r.doubles(8)
            if v == 1.7:
                r.raw(16 * 4)
            r.raw(48 * 4)
    return dict(type=src_type, swl=swl, delay_ms=delay, time=time, valid=valid, energy=energy)


def read_image_sources(r, n_rec, c_sound=343.0):
    rec, time, energy, dirs, dep, order, kind = [], [], [], [], [], [], []
    for q in range(n_rec):
        r.i32()
        n_paths = r.i32()
        for _ in range(n_paths):
            t = r.i16()
            if t == 0:
                npts = r.i32()
                pts = r.doubles(3 * npts).reshape(npts, 3)
                e = r.doubles(8).copy()
                if r.boolean():
                    r.raw(16 * r.i32())
                r.doubles(8)
                tt = r.f64()
                r.raw(4 * (npts - 2))
                d = pts[-2] - pts[-1]
                s0 = pts[1] - pts[0]
            elif t == 1:
                nsub = r.i32()
                npts = r.i32()
                pts = r.doubles(3 * nsub * npts).reshape(nsub, npts, 3)
                if r.boolean():
                    raise ValueError("compound path with a special filter is not supported")
                if r.boolean():
                    r.doubles(8 * nsub)
                tt = r.f64()
                r.raw(4 * (npts - 2))
                L = r.i32()
                h = r.doubles(8 * L).reshape(L, 8)
                r.doubles(6 * 8 * L)
                e = (h ** 2).sum(0)
                d = (pts[:, -2] - pts[:, -1]).mean(0)
                s0 = (pts[:, 1] - pts[:, 0]).mean(0)
            else:
                raise ValueError(f"unknown path kind {t}")
            rec.append(q)
            time.append(tt)
            energy.append(e)
            dirs.append(d / np.linalg.norm(d))
            dep.append(s0 / np.linalg.norm(s0))
            order.append(npts - 2)
            kind.append(t)
    return dict(rec=np.array(rec, np.int32), time=np.array(time), energy=np.array(energy).reshape(-1, 8),
                dir=np.array(dirs).reshape(-1, 3), dep=np.array(dep).reshape(-1, 3), order=np.array(order, np.int16), kind=np.array(kind, np.int8))


def read_ray_traced(r, n_rec, v, bin_ms):
    rec_type = r.u32()
    fs = r.i32()
    n = r.i32()
    co_time = r.f64()
    if v == 1.7:
        raise ValueError("version 1.7 ray data is not supported")
    per = 8 if v > 2.1 else 7
    step = max(1, int(round(bin_ms * 1e-3 * fs)))
    nb = -(-n // step)
    hist = np.zeros((n_rec, 8, nb))
    speed = np.zeros(n_rec)
    for q in range(n_rec):
        speed[q] = r.f64()
        a = r.doubles(n * 8 * per).reshape(n, 8, per)[:, :, 0]
        pad = np.zeros((nb * step, 8))
        pad[:n] = a
        hist[q] = pad.reshape(nb, step, 8).sum(1).T
    return dict(rec_type=rec_type, sample_rate=fs, n_samples=n, cutoff_s=co_time, bin_s=step / fs, hist=hist,
                sound_speed=speed)


def convert(path, bin_ms=1.0):
    out = {}
    with open(path, "rb") as fh:
        r = Reader(fh)
        out["saved"] = r.string()
        version = r.string()
        v = version_number(version)
        out["version"] = version
        out["cutoff_s"] = r.f64()
        out["sample_rate"] = r.i32()
        if v >= 2.7 and r.boolean():
            raise ValueError("third-octave results are not supported")
        n_src = r.i32() if v >= 1.1 else 1
        out["sources"] = r.doubles(3 * n_src).reshape(n_src, 3).copy()
        n_rec = r.i32()
        rec = np.zeros((n_rec, 3))
        rho_c = np.full(n_rec, 400.0)
        for q in range(n_rec):
            rec[q] = r.doubles(3)
            if v >= 2.0:
                rho_c[q] = r.f64()
        out["receivers"] = rec
        out["rho_c"] = rho_c
        counts = {"direct": 0, "is": 0, "ray": 0}
        while True:
            tag = r.string()
            if tag in ("End", ""):
                break
            if tag.startswith("Direct_Sound"):
                d = read_direct(r, n_rec, v)
                s = counts["direct"]
                for k, val in d.items():
                    out[f"src{s}_direct_{k}"] = val
                counts["direct"] += 1
            elif tag == "Image-Source_Data":
                d = read_image_sources(r, n_rec)
                s = counts["is"]
                for k, val in d.items():
                    out[f"src{s}_is_{k}"] = val
                counts["is"] += 1
            elif tag == "Ray-Traced_Data":
                d = read_ray_traced(r, n_rec, v, bin_ms)
                s = counts["ray"]
                for k, val in d.items():
                    out[f"src{s}_ray_{k}"] = val
                counts["ray"] += 1
            else:
                raise ValueError(f"unknown block '{tag}'")
            print(f"read {tag} ({counts})", flush=True)
    out["counts"] = np.array([counts["direct"], counts["is"], counts["ray"]])
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    bin_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    data = convert(src, bin_ms)
    dst = src.with_suffix(".npz")
    np.savez_compressed(dst, **data)
    print(f"version {data['version']}, {len(data['sources'])} source(s), {len(data['receivers'])} receiver(s)")
    for s in range(int(data["counts"][1])):
        n = len(data[f"src{s}_is_rec"])
        print(f"source {s}: {n} image-source paths, orders {np.bincount(data[f'src{s}_is_order']).tolist()}")
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
