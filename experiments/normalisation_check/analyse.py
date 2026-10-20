"""Compare the two normalisations on MPEG-7: validation, bound, zero sets, and
the DGMM 2027 / IWCIA 2025 rotation-free accuracies.

Reads the output of extract.py. Needs numpy only (plus qsig).

  1. The "code" variant must equal the stored DGMM tables
     (results/{all,device}_n130_rows.csv) value for value -- otherwise nothing
     below means anything.
  2. Bound: largest pointwise phi_hat and largest descriptor value per variant.
  3. Zero sets: identical between variants, column by column.
  4. How much the values move, per direction.
  5. 1NN leave-one-out, d_C and L2, on the direction sets the DGMM scripts
     evaluate, both subsets, both variants, with a paired bootstrap on the
     difference (4000 resamples over shapes).

Usage: python analyse.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import qsig.threadguard  # noqa: E402,F401
import numpy as np  # noqa: E402

from qsig import directions as D  # noqa: E402
from qsig.classify import loo_1nn_from_distances  # noqa: E402
from qsig.signature import euclidean, pairwise_d_C  # noqa: E402

DIRS = D.by_angle(D.pool(max_norm2=130))
COL = {(d.p, d.q): i for i, d in enumerate(DIRS)}
SLOT = pickle.load(open(os.path.join(HERE, "..", "pairs_and_rules", "slotcols.pkl"), "rb"))
VARIANTS = ("code", "generator")
FILES = {"mpeg7_full_features.csv": "sig", "mpeg7_disjunctive.csv": "sigd",
         "mpeg7_asymmetry.csv": "siga", "mpeg7_generalpairs.csv": "gp",
         "mpeg7_widepairs.csv": "wp"}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def matrix(rows, prefix, n):
    return np.array([[float(r[f"{prefix}{i}"]) for i in range(n)] for r in rows])


def load_variant(out, v):
    feat = read(os.path.join(out, v, "mpeg7_full_features.csv"))
    ids = [r["shape_id"] for r in feat]
    y = np.array([r["cls"] for r in feat])
    m = {"sig": matrix(feat, "sig", 64)}
    for fn, pre in FILES.items():
        if pre == "sig":
            continue
        rows = {r["shape_id"]: r for r in read(os.path.join(out, v, fn))}
        rr = [rows[i] for i in ids]
        if pre in ("gp", "wp"):
            for suf in ("", "d", "a"):
                m[pre + suf] = matrix(rr, pre + suf, 12)
        else:
            m[pre] = matrix(rr, pre, 64)
    return ids, y, m


def sets():
    """Direction sets, as column indices into the angle-sorted 64-pool."""
    out = {"sint": [COL[(d.p, d.q)] for d in D.S_INT]}
    for g in (12, 10, 8, 6):          # 5 deg is infeasible in the 64-pool
        out[f"maxgap{g}"] = [COL[(d.p, d.q)] for d in D.min_cost_maxgap(float(g))]
    for k in sorted(SLOT):
        out[f"slot{k}"] = list(SLOT[k])
    out["pool64"] = list(range(64))
    return {k: sorted(v) for k, v in out.items()}


def correct_vec(dist, y):
    d = np.array(dist, copy=True)
    np.fill_diagonal(d, np.inf)
    return y[d.argmin(1)] == y


def boot(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = b.astype(int) - a.astype(int)
    N = len(d)
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)]) * 100
    return d.mean() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    data = {v: load_variant(args.out, v) for v in VARIANTS}
    ids, y, _ = data["code"]
    assert data["generator"][0] == ids

    # 1. validation against the stored DGMM tables
    print("== 1. code variant vs stored DGMM tables (E, family S, rows+numba)")
    sig = data["code"][2]["sig"]
    pos = {s: i for i, s in enumerate(ids)}
    for t in ("all_n130_rows.csv", "device_n130_rows.csv"):
        n = bad = 0
        maxdiff = 0.0
        for r in read(os.path.join(REPO, "results", t)):
            a = sig[pos[r["shape_id"]], COL[(int(r["p"]), int(r["q"]))]]
            b = float(r["E"])
            n += 1
            bad += a != b
            maxdiff = max(maxdiff, abs(a - b))
        print(f"   {t}: {n} values, {bad} differ, max |diff| {maxdiff:.3g}")

    # 2. bound
    print("\n== 2. bound: largest pointwise phi_hat and largest descriptor value")
    b = read(os.path.join(args.out, "bounds.csv"))
    for v in VARIANTS:
        h = np.array([float(r[f"phi_hat_max_{v}"]) for r in b])
        print(f"   {v:9s} pointwise phi_hat: max {h.max():.4f}, shapes with a point > 1: "
              f"{int((h > 1).sum())}/{len(h)}")
    for v in VARIANTS:
        m = data[v][2]
        parts = ", ".join(f"{k} {m[k].max():.4f} ({int((m[k] > 1).sum())} >1)" for k in sorted(m))
        print(f"   {v:9s} max value per family: {parts}")

    # 3. zero sets
    print("\n== 3. zero sets (value == 0), code vs generator")
    for k in sorted(data["code"][2]):
        a, g = data["code"][2][k], data["generator"][2][k]
        print(f"   {k:5s} zeros {int((a == 0).sum()):6d} / {a.size}, pattern identical: "
              f"{bool(((a == 0) == (g == 0)).all())}")

    # 4. how much values move
    print("\n== 4. relative change |gen - code| / code, per column (non-zero entries)")
    a, g = data["code"][2]["sig"], data["generator"][2]["sig"]
    nz = a > 0
    rel = np.where(nz, np.abs(g - a) / np.where(nz, a, 1), np.nan)
    med = np.nanmedian(rel, axis=0)
    q95 = np.nanpercentile(rel, 95, axis=0)
    ratio = np.where(nz, g / np.where(nz, a, 1), np.nan)
    unchanged = [f"({d.p},{d.q})" for i, d in enumerate(DIRS) if np.array_equal(a[:, i], g[:, i])]
    print(f"   columns bit-identical: {unchanged}")
    print(f"   median relative change over columns: min {np.nanmin(med):.3f}, "
          f"median {np.nanmedian(med):.3f}, max {np.nanmax(med):.3f}")
    print(f"   95th percentile, worst column: {np.nanmax(q95):.3f}; "
          f"ratio gen/code range [{np.nanmin(ratio):.3f}, {np.nanmax(ratio):.3f}]")
    rho = []
    for i in range(64):
        ra, rg = np.argsort(np.argsort(a[:, i])), np.argsort(np.argsort(g[:, i]))
        rho.append(np.corrcoef(ra, rg)[0, 1])
    rho = np.array(rho)
    print(f"   Spearman rank correlation code vs gen per column: min {rho.min():.4f}, "
          f"median {np.median(rho):.4f}")
    for k in ("sigd", "gp", "wp"):
        a2, g2 = data["code"][2][k], data["generator"][2][k]
        nz2 = a2 > 0
        r2 = np.abs(g2 - a2)[nz2] / a2[nz2]
        print(f"   {k:5s}: median relative change {np.median(r2):.3f}, 95th pct {np.percentile(r2, 95):.3f}")

    # 5. DGMM / IWCIA accuracies
    print("\n== 5. 1NN leave-one-out on rotation-free signatures (phi), code -> generator")
    print("   difference = generator - code, points, with 95% paired bootstrap interval")
    S = sets()
    for subset in ("all", "device"):
        sel = np.ones(len(y), bool) if subset == "all" else np.array([c.startswith("device") for c in y])
        ys = y[sel]
        print(f"\n   subset {subset} ({sel.sum()} shapes)")
        print(f"   {'set':>9s} {'k':>3s} | {'dC code':>8s} {'dC gen':>8s} {'diff [95% CI]':>24s} | "
              f"{'L2 code':>8s} {'L2 gen':>8s} {'diff [95% CI]':>24s}")
        for name, cols in S.items():
            cells = []
            for metric in ("dC", "L2"):
                acc = {}
                for v in VARIANTS:
                    X = data[v][2]["sig"][sel][:, cols]
                    dist = pairwise_d_C(X) if metric == "dC" else euclidean(X)
                    acc[v] = correct_vec(dist, ys)
                    if v == "code":
                        assert abs(acc[v].mean() * 100 - loo_1nn_from_distances(dist, ys)) < 1e-9
                m, lo, hi = boot(acc["code"], acc["generator"])
                star = "*" if not (lo <= 0 <= hi) else " "
                cells.append(f"{acc['code'].mean()*100:8.2f} {acc['generator'].mean()*100:8.2f} "
                             f"{m:+6.2f} [{lo:+.2f},{hi:+.2f}]{star}")
            print(f"   {name:>9s} {len(cols):3d} | {cells[0]} | {cells[1]}")


if __name__ == "__main__":
    main()
