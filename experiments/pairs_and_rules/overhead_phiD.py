"""Timing: what computing phi_D (and phi_A) alongside phi costs.

fast.compute (row-prefix kernel, method="rows") returns phi, phi_D and phi_A
from the same four cone counts. This script times one full call, then re-times
the phi_D and the phi_A blocks alone on the returned arrays, and reports

    overhead_D(d) = median T_D / median (T_full - T_D - T_A)

per direction d of the 64-direction pool, and per direction set (sum over the
set's directions). Run it on the timing machine, pinned and single-threaded,
like scripts/calibrate.py:

    sudo cpupower frequency-set -g performance
    numactl --cpunodebind=0 --membind=0 taskset -c 2 python3 overhead_phiD.py

Writes overhead_phiD.json and prints a summary.
"""
from __future__ import annotations
import paths  # noqa: F401
import qsig.threadguard  # noqa: F401  -- MUST precede numpy
import json, os, pickle, platform, time
import numpy as np
from qsig import dataset, descriptor, directions
from qsig import fast as _fast
from qsig.dataset import OBJECT, BACKGROUND

N_SHAPES, REPEATS = int(os.environ.get("OV_SHAPES", 24)), int(os.environ.get("OV_REPEATS", 7))
INT = _fast.INT


def phi_d_block(q1, q2, q3, q4, b, norm_part):
    phi_d = (q1 * q4 + q2 * q3) * b.astype(INT)
    card = int(np.count_nonzero(phi_d))
    denom = np.power(norm_part, 2).astype(np.float64)
    v = 4.0 * float((phi_d.astype(np.float64) / denom).sum())
    return 0.0 if card == 0 else v / card


def phi_a_block(q1, q2, q3, q4, b, norm_part):
    phi_a = np.abs(q1 * q4 - q2 * q3) * b.astype(INT)
    card = int(np.count_nonzero(phi_a))
    denom = np.power(norm_part, 2).astype(np.float64)
    v = 4.0 * float((phi_a.astype(np.float64) / denom).sum())
    return 0.0 if card == 0 else v / card


def main():
    if not _fast.HAVE_NUMBA:
        raise SystemExit("numba is not available: the timings would be of the pure-Python fallback, "
                         "not of the JIT kernel the paper reports. Install numba first.")
    shapes = dataset.load_mpeg7(paths.data("MPEG7dataset.zip"))
    shapes = shapes[::max(1, len(shapes) // N_SHAPES)][:N_SHAPES]
    pool = directions.by_angle(directions.pool(max_norm2=130))
    descriptor.warm_up("rows")
    per_dir = {}
    for d in pool:
        tf, td, ta = [], [], []
        for sh in shapes:
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                r = _fast.compute(sh.img, OBJECT, BACKGROUND, d.vec, "rows")
                t1 = time.perf_counter()
                b = (np.asarray(sh.img) == BACKGROUND)
                args = (*r["_quads"], b, r["_norm_part"])
                t2 = time.perf_counter(); vd = phi_d_block(*args); t3 = time.perf_counter()
                phi_a_block(*args); t4 = time.perf_counter()
                assert abs(vd - r["q1_d"]) <= 1e-12 * max(1.0, abs(vd))
                tf.append(t1 - t0); td.append(t3 - t2); ta.append(t4 - t3)
        Tf, Td, Ta = (float(np.median(x)) for x in (tf, td, ta))
        per_dir[f"{d.p}x{d.q}"] = dict(p=d.p, q=d.q, full=Tf, phiD=Td, phiA=Ta, phi_only=Tf - Td - Ta,
                                       overhead_D=Td / (Tf - Td - Ta))
        print(f"({d.p},{d.q})  full {Tf*1e3:7.3f} ms  phi_D {Td*1e3:6.3f} ms  overhead {100*Td/(Tf-Td-Ta):5.1f}%",
              flush=True)
    slot = pickle.load(open(os.path.join(paths.HERE, "slotcols.pkl"), "rb"))
    sets = {f"slot_set({k})": [pool[i] for i in slot[k]] for k in (4, 6, 10, 12, 24)}
    sets["full pool (64)"] = pool
    summary = {}
    for name, ds in sets.items():
        D = sum(per_dir[f"{d.p}x{d.q}"]["phiD"] for d in ds)
        P = sum(per_dir[f"{d.p}x{d.q}"]["phi_only"] for d in ds)
        summary[name] = round(100 * D / P, 2)
    ov = [v["overhead_D"] for v in per_dir.values()]
    out = dict(host=platform.node(), numba=True, python=platform.python_version(), numpy=np.__version__,
               shapes=N_SHAPES, repeats=REPEATS, per_direction=per_dir, per_set_percent=summary,
               min_percent=round(100 * min(ov), 2), max_percent=round(100 * max(ov), 2),
               argmax=max(per_dir, key=lambda k: per_dir[k]["overhead_D"]))
    json.dump(out, open(os.path.join(paths.HERE, "overhead_phiD.json"), "w"), indent=1)
    print("\nper direction: min %.1f%%  max %.1f%% (at %s)" % (out["min_percent"], out["max_percent"], out["argmax"]))
    for k, v in summary.items():
        print(f"  {k:16s} {v:5.1f}%")


if __name__ == "__main__":
    main()
