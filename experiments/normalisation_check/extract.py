"""MPEG-7 features under both line families of the local normalisation.

For every shape the four cones are computed ONCE per direction (pair) and the
three rules (phi, phi_D, phi_A) are normalised twice -- with the published line
family ("code") and with lines along the cone generators ("generator"); see
qsig/normcheck.py. Writes, for each variant, the same CSV files the
experiments/pairs_and_rules scripts read, so those scripts can be rerun on
either directory unchanged:

    <out>/<variant>/mpeg7_full_features.csv   classical + sig0..63   (phi)
    <out>/<variant>/mpeg7_disjunctive.csv     sigd0..63              (phi_D)
    <out>/<variant>/mpeg7_asymmetry.csv       siga0..63              (phi_A)
    <out>/<variant>/mpeg7_generalpairs.csv    gp/gpd/gpa 0..11       (Farey pairs)
    <out>/<variant>/mpeg7_widepairs.csv       wp/wpd/wpa 0..11       (wide pairs)
    <out>/<variant>/mpeg7_fourier.csv         fd1..32 (identical in both)
    <out>/bounds.csv                          per shape, largest pointwise phi_hat

Same preprocessing, pool, pair sets and kernel ("rows") as the original
extractors. The "code" variant must reproduce them exactly; analyse.py checks
it against the stored DGMM tables.

Usage: python extract.py [--workers 16] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.join(HERE, "..", "pairs_and_rules")
sys.path.insert(0, PR)
import paths  # noqa: E402,F401  -- puts the repository root on sys.path

import qsig.threadguard  # noqa: E402,F401  -- before numpy

from qsig import fast as _fast  # noqa: E402  -- import numba BEFORE any scipy stub

assert _fast.HAVE_NUMBA, "numba missing: the kernel would fall back to pure Python"

# extract_features imports scipy.io for the SkeView .mat masks only; MPEG-7 never
# reaches it, and the recorded extraction venv has no scipy. Stubbed only after
# numba is imported: numba probes scipy at import time and a stub breaks that.
if "scipy.io" not in sys.modules:
    try:
        import scipy.io  # noqa: F401
    except ImportError:
        sys.modules["scipy"] = types.ModuleType("scipy")
        sys.modules["scipy.io"] = types.ModuleType("scipy.io")

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from qsig import directions as D  # noqa: E402
from qsig import normcheck as NC  # noqa: E402
from qsig.dataset import BACKGROUND, OBJECT, load_mpeg7  # noqa: E402
from qsig.descriptor import warm_up  # noqa: E402

import general_pairs as GP  # noqa: E402
from extract_features import classical_features  # noqa: E402
from extract_fourier import fourier_descriptor  # noqa: E402

DIRS = D.by_angle(D.pool(max_norm2=130))
assert len(DIRS) == 64
GEN = [(tuple(d["r"]), tuple(d["s"])) for d in GP.spread(12)]
WIDE = [(tuple(d["r"]), tuple(d["s"])) for d in json.load(open(os.path.join(PR, "wide_pairs.json")))]
VARIANTS = NC.NORMS


def _init():
    warm_up("rows")


def one(job):
    sid, cls, img = job
    out = {"sid": sid, "cls": cls, "cl": classical_features(img), "fd": fourier_descriptor(img)}
    hat = {v: 0.0 for v in VARIANTS}
    for key, items, fn in (("orth", DIRS, lambda d: NC.orthogonal(img, d.vec, OBJECT, BACKGROUND)),
                           ("gen", GEN, lambda p: NC.pair(img, p[0], p[1], OBJECT, BACKGROUND)),
                           ("wide", WIDE, lambda p: NC.pair(img, p[0], p[1], OBJECT, BACKGROUND))):
        vals = {v: ([], [], []) for v in VARIANTS}
        for it in items:
            res = fn(it)
            for v in VARIANTS:
                r = res[v]
                vals[v][0].append(r["q1"]); vals[v][1].append(r["q1_d"]); vals[v][2].append(r["q1_a"])
                hat[v] = max(hat[v], r["phi_hat_max"])
        out[key] = vals
    out["hat"] = hat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    shapes = load_mpeg7(paths.data("MPEG7dataset.zip"))
    jobs = [(s.shape_id, s.cls, s.img) for s in shapes]
    with Pool(args.workers, initializer=_init) as pool:
        res = pool.map(one, jobs, chunksize=4)

    fmt = repr
    for v in VARIANTS:
        d = os.path.join(args.out, v)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "mpeg7_full_features.csv"), "w") as f:
            f.write(",".join(["shape_id", "cls", "area_ratio", "circularity"]
                             + [f"hu{i+1}" for i in range(7)] + [f"sig{i}" for i in range(64)]) + "\n")
            for r in res:
                f.write(",".join([r["sid"], r["cls"]] + [str(x) for x in r["cl"]]
                                 + [str(np.float64(x)) for x in r["orth"][v][0]]) + "\n")
        for fn, pre, idx in (("mpeg7_disjunctive.csv", "sigd", 1), ("mpeg7_asymmetry.csv", "siga", 2)):
            with open(os.path.join(d, fn), "w") as f:
                f.write(",".join(["shape_id"] + [f"{pre}{i}" for i in range(64)]) + "\n")
                for r in res:
                    f.write(",".join([r["sid"]] + [fmt(x) for x in r["orth"][v][idx]]) + "\n")
        for fn, key, pre in (("mpeg7_generalpairs.csv", "gen", "gp"), ("mpeg7_widepairs.csv", "wide", "wp")):
            with open(os.path.join(d, fn), "w") as f:
                f.write(",".join(["shape_id"] + [f"{pre}{i}" for i in range(12)]
                                 + [f"{pre}d{i}" for i in range(12)] + [f"{pre}a{i}" for i in range(12)]) + "\n")
                for r in res:
                    a, b, c = r[key][v]
                    f.write(",".join([r["sid"]] + [fmt(x) for x in a + b + c]) + "\n")
        with open(os.path.join(d, "mpeg7_fourier.csv"), "w") as f:
            f.write(",".join(["shape_id", "cls"] + [f"fd{m}" for m in range(1, 33)]) + "\n")
            for r in res:
                f.write(",".join([r["sid"], r["cls"]] + [f"{x:.10g}" for x in r["fd"]]) + "\n")
    with open(os.path.join(args.out, "bounds.csv"), "w") as f:
        f.write("shape_id,phi_hat_max_code,phi_hat_max_generator\n")
        for r in res:
            f.write(f"{r['sid']},{r['hat']['code']!r},{r['hat']['generator']!r}\n")
    print(f"wrote {len(res)} shapes x {len(VARIANTS)} variants to {args.out}")


if __name__ == "__main__":
    main()
