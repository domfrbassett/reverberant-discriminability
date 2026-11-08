"""Split the BRAS CR4 BRIR SOFA file into parts under 300 MB and print its layout.

Usage: python split_bras_cr4.py "<path to CR4_BRIRs.sofa>"
Writes data/bras_cr4/CR4_BRIRs_partNN.npz next to this script: Data.IR as float32 and every
per-measurement dataset sliced to the same measurements; datasets without a measurement
dimension and the file attributes go into CR4_BRIRs_meta.npz.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

MAX_BYTES = 300e6

src = Path(sys.argv[1])
out = Path(__file__).resolve().parent / "bras_cr4"
out.mkdir(exist_ok=True)

with h5py.File(src, "r") as f:
    ir = f["Data.IR"]
    M = ir.shape[0]
    print("Data.IR", ir.shape, ir.dtype, "fs", f["Data.SamplingRate"][()])
    per, meta = {}, {}
    for k, v in f.items():
        if k == "Data.IR" or not isinstance(v, h5py.Dataset):
            continue
        a = v[()]
        if a.ndim and a.shape[0] == M and M > 1:
            per[k] = a
        else:
            meta[k] = a
        print(f"{k:24s} {a.shape} {a.dtype}", a.reshape(-1)[:6] if a.size else "")
    attrs = {f"attr_{k}": np.bytes_(v) if isinstance(v, (bytes, str)) else np.asarray(v) for k, v in f.attrs.items()}
    for k, v in f.attrs.items():
        print(f"attr {k}: {str(v)[:300]}")
    np.savez(out / "CR4_BRIRs_meta.npz", **meta, **attrs)

    step = max(1, int(MAX_BYTES // (np.prod(ir.shape[1:]) * 4)))
    for n, i in enumerate(range(0, M, step)):
        j = min(M, i + step)
        np.savez(out / f"CR4_BRIRs_part{n:02d}.npz", ir=ir[i:j].astype(np.float32), index=np.arange(i, j),
                 **{k: v[i:j] for k, v in per.items()})
        print(f"part {n:02d}: measurements {i}-{j - 1}")
print("done:", out)
