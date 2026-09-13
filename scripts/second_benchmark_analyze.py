#!/usr/bin/env python3
"""Second-benchmark analysis: classical baseline vs the Q-concavity signature,
mirroring classical_baseline_2026-09-13.md's table on a new dataset.

Reads the two tables `classical_features.py` and `run_pool.py` produce and
prints the comparison: classical alone, classical + mu, classical + all 64
raw components, classical + greedy-12, plus the correlation-decomposition
diagnostics (mean pairwise correlation of the 64 components, before and after
subtracting mu; F(mu) vs mean F of the other components).

    python scripts/second_benchmark_analyze.py \
        --data Animal2000-GT.zip --dataset animal2000 \
        --classical results/animal2000_classical.csv \
        --signature results/animal2000_n130_rows.csv

1NN leave-one-out, per-feature z-scored plain L2 throughout -- same protocol
as classical_baseline_2026-09-13.md. Greedy-12 selection here is the same
FIT=EVAL protocol as that doc (selects and scores on the same shapes): a
pre-existing, known-optimistic caveat (paper_a_plan_2026-09-13.md sec 2), not
something this script fixes.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401  -- MUST precede numpy

import numpy as np  # noqa: E402

from qsig import classify, dataset, directions, skeview, store  # noqa: E402
from qsig.classical import FEATURE_NAMES  # noqa: E402

SIGNATURE_DIRS_SPEC = "the 64-orthogonal pool: directions.by_angle(directions.pool(max_norm2=130))"


def load_shapes(data_path: str, ds: str, subset: str, long_side: int):
    if ds == "mpeg7":
        classes = dataset.DEVICE_CLASSES if subset == "device" else None
        return dataset.load_mpeg7(data_path, long_side=long_side, classes=classes)
    return skeview.load_skeview(data_path, ds, long_side=long_side)


def load_classical(path: str, ds: str):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("dataset") != ds:
                continue
            out[row["shape_id"]] = np.array([float(row[f]) for f in FEATURE_NAMES])
    return out


def zscore(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def loo_1nn_acc(X, labels):
    X = zscore(np.asarray(X, dtype=float))
    labels = np.asarray(labels)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(axis=1)
    return float((labels[nn] == labels).mean() * 100.0)


def greedy_select(sig, base, labels, k=12):
    """Forward selection of k signature columns on top of `base`, scored by
    LOO 1NN accuracy on the SAME shapes (fit=eval; see module docstring).
    Distances are additive over z-scored per-feature squared differences, so
    each step reuses the running sum instead of rebuilding it -- O(k * d *
    n^2) instead of O(k * d * n^2 * d)."""
    labels = np.asarray(labels)

    def sq(col):
        col = zscore(col.reshape(-1, 1)).ravel()
        return (col[:, None] - col[None, :]) ** 2

    d2_base = sum((sq(base[:, c]) for c in range(base.shape[1])), np.zeros((len(labels),) * 2))
    col_d2 = [sq(sig[:, j]) for j in range(sig.shape[1])]

    chosen, remaining, accs = [], list(range(sig.shape[1])), []
    cur = d2_base.copy()
    for _ in range(k):
        best = None
        for j in remaining:
            trial = cur + col_d2[j]
            trial2 = trial.copy()
            np.fill_diagonal(trial2, np.inf)
            acc = float((labels[trial2.argmin(axis=1)] == labels).mean() * 100.0)
            if best is None or acc > best[1]:
                best = (j, acc)
        chosen.append(best[0])
        cur = cur + col_d2[best[0]]
        remaining.remove(best[0])
        accs.append(best[1])
    return chosen, accs


def f_ratio(x, y, classes):
    overall = x.mean()
    ssb = sum(((x[y == k].mean() - overall) ** 2) * (y == k).sum() for k in range(len(classes)))
    ssw = sum(((x[y == k] - x[y == k].mean()) ** 2).sum() for k in range(len(classes)))
    return (ssb / (len(classes) - 1)) / (ssw / (len(x) - len(classes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", choices=("mpeg7", "animal2000", "swedishleaves"), default="mpeg7")
    ap.add_argument("--subset", choices=("device", "all"), default="all")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--classical", required=True, help="output of classical_features.py")
    ap.add_argument("--signature", required=True, nargs="+",
                    help="output(s) of run_pool.py --dirs pool --max-norm2 130")
    ap.add_argument("--greedy-k", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N shapes")
    args = ap.parse_args()

    shapes = load_shapes(args.data, args.dataset, args.subset, args.long_side)
    if args.limit:
        shapes = shapes[: args.limit]
    dirs = directions.by_angle(directions.pool(max_norm2=130))
    assert len(dirs) == 64, f"expected 64 directions from max_norm2=130, got {len(dirs)}"

    table = {}
    for t in args.signature:
        table.update(store.load_table(t))
    sig, labels, ids = classify.build_signatures(table, shapes, dirs, family="S")

    cfeat = load_classical(args.classical, args.dataset)
    missing = [sid for sid in ids if sid not in cfeat]
    if missing:
        raise SystemExit(f"{len(missing)} shapes missing from {args.classical}, e.g. {missing[:5]}")
    classical = np.array([cfeat[sid] for sid in ids])

    mu = sig.mean(axis=1)
    print(f"\n=== {args.dataset} ({len(ids)} shapes, {len(set(labels))} classes) ===")
    print(f"signature directions: {SIGNATURE_DIRS_SPEC} ({len(dirs)} dirs)\n")

    feats = {
        "classical (area ratio + circ + Hu1 + Hu2)": classical[:, :4],
        "Hu 1-7": classical[:, 2:],
        "mu alone": mu.reshape(-1, 1),
        "all 64 raw signature components": sig,
        "classical + mu": np.column_stack([classical[:, :4], mu]),
        "classical + all 64 raw components": np.column_stack([classical[:, :4], sig]),
    }
    for name, X in feats.items():
        print(f"  {name:45s} dims={X.shape[1]:3d}  acc={loo_1nn_acc(X, labels):6.2f}%")

    chosen, accs = greedy_select(sig, classical[:, :4], labels, k=args.greedy_k)
    print(f"  classical + greedy {args.greedy_k:<33d} dims={4+args.greedy_k:3d}  acc={accs[-1]:6.2f}%")
    print(f"  greedy path: {[round(a, 2) for a in accs]}")
    print(f"  chosen direction indices (into the 64-pool, angle order): {chosen}")

    corr = np.corrcoef(sig, rowvar=False)
    iu = np.triu_indices(64, k=1)
    print(f"\n  mean pairwise corr of 64 raw components: {corr[iu].mean():+.3f}")
    sig_c = sig - mu[:, None]
    print(f"  mean pairwise corr after removing mu:    {np.corrcoef(sig_c, rowvar=False)[iu].mean():+.3f}")
    print(f"  corr(mu, area_ratio) = {np.corrcoef(mu, classical[:, 0])[0,1]:+.3f}, "
          f"corr(mu, circularity) = {np.corrcoef(mu, classical[:, 1])[0,1]:+.3f}")

    classes = sorted(set(labels))
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[c] for c in labels])
    f_mu = f_ratio(mu, y, classes)
    f_comp = [f_ratio(sig[:, i], y, classes) for i in range(64)]
    print(f"  F(mu) = {f_mu:.2f}, mean F(raw components) = {np.mean(f_comp):.2f}")


if __name__ == "__main__":
    main()
