#!/usr/bin/env python3
"""Accuracy against computation cost -- the paper's headline figure, as a table.

    python scripts/evaluate.py --data ../MPEG7dataset.zip --subset device \
        --table results/device_n130_rows.csv \
        --sets sint maxgap10 maxgap8 maxgap6

COSTS ARE MODELLED, NOT MEASURED FROM THE TABLE. A pool run has 20 workers
contending for memory bandwidth and L3, which inflates per-job times by roughly
3x and unevenly across directions; handoff sec 7.2 separates bulk runs from
measurement runs for exactly this reason. Prices come from
`qsig.directions.COST_MODELS`, fitted on pinned single-process runs. Pass
`--show-bulk` to see the contaminated numbers alongside, for diagnosis only.

Both DISTANCES are reported. d_C is the orbit distance of IWCIA 2025 sec 5.2;
L2 is plain Euclidean with no rotation handling. The control matters: measured
2026-09-06, d_C beats L2 on the Device subset (73.5 vs 68.0) and LOSES to it on
the full 1400-shape set (50.2 vs 53.6). IWCIA 2025 used d_C throughout and never
reported the control.

If the table contains family R rows (rotational signature), they are evaluated
and printed alongside S automatically -- that comparison is handoff sec 6's
test 0, the experiment the paper's framing turns on.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401

from qsig import classify, dataset, directions, store  # noqa: E402
from scripts.run_pool import resolve_dirs  # noqa: E402

# IWCIA 2025 Table 2. R = rotational, S = rotation-free. S_18/S_9/S_6/S_3 were
# obtained by LINEAR INTERPOLATION of Q-concavity values between neighbouring
# S_int angles, so their cost is undefined and their accuracies are
# approximations -- recomputing them exactly moves them by up to 4.5 points
# (handoff sec 3.2.1).
PUBLISHED = {
    "S_int": {"n": 20, "R_all": 42.57, "R_dev": 71.5, "S_all": 46.86, "S_dev": 75.0},
    "S_18":  {"n": 18, "R_all": 41.43, "R_dev": 67.0, "S_all": 48.86, "S_dev": 71.0},
    "S_9":   {"n": 9,  "R_all": 40.00, "R_dev": 61.0, "S_all": 47.21, "S_dev": 70.0},
    "S_6":   {"n": 6,  "R_all": 29.64, "R_dev": 54.0, "S_all": 33.50, "S_dev": 56.5},
    "S_3":   {"n": 3,  "R_all": 14.14, "R_dev": 33.0, "S_all": 15.71, "S_dev": 36.5},
    "S_1":   {"n": 1,  "R_all": 1.43,  "R_dev": 10.0, "S_all": 1.43,  "S_dev": 10.0},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--table", required=True, nargs="+",
                    help="one or more result tables; family R tables may be listed too")
    ap.add_argument("--subset", choices=("device", "all"), default="device")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--sets", nargs="+", required=True,
                    help="direction-set specs: sint | maxgap<G> | slot<N>[:tol] | "
                         "cheapest<K> | pool | list:pxq,...")
    ap.add_argument("--cost-model", default="", help=f"one of {sorted(directions.COST_MODELS)}; "
                                                     "default: taken from the table")
    ap.add_argument("--show-bulk", action="store_true",
                    help="also print the contaminated per-job seconds from the table")
    args = ap.parse_args()

    classes = dataset.DEVICE_CLASSES if args.subset == "device" else None
    shapes = dataset.load_mpeg7(args.data, long_side=args.long_side, classes=classes)

    table = {}
    impls = set()
    for t in args.table:
        part = store.load_table(t)
        table.update(part)
        impls.update(r["impl"] for r in part.values() if r.get("impl"))
    families = sorted({k[4] for k in table})
    model = args.cost_model or next((i for i in impls if i in directions.COST_MODELS),
                                    "rows+numba")

    print(f"{args.subset} subset, {len(shapes)} shapes, {len(table)} rows")
    print(f"implementation(s) in table: {sorted(impls) or ['unknown']}")
    print(f"families present: {families}   cost model used for pricing: {model}\n")

    hdr = f"{'set':>14s} {'fam':>4s} {'k':>3s} {'cost s':>9s} {'maxgap':>7s} {'acc dC':>8s} {'acc L2':>8s}"
    if args.show_bulk:
        hdr += f" {'bulk s':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for spec in args.sets:
        dirs = directions.by_angle(resolve_dirs(spec, 0, 0))
        for fam in families:
            try:
                sigs, labels, _ = classify.build_signatures(table, shapes, dirs, family=fam)
            except KeyError as exc:
                print(f"{spec:>14s} {fam:>4s} {len(dirs):3d}  SKIPPED -- {exc}")
                continue
            # A rotational signature always evaluates at (1,0), so its cost is
            # flat in the number of components, not in the directions' lengths.
            if fam == "R":
                one = directions.cost_of([directions.Direction(1, 0)], model)
                cost = one * len(dirs)
            else:
                cost = directions.cost_of(dirs, model)
            gap = directions.max_angular_gap(dirs)
            line = (f"{spec:>14s} {fam:>4s} {len(dirs):3d} {cost:9.4f} {gap:6.1f}d "
                    f"{classify.loo_1nn(sigs, labels, 'dC'):7.2f}% "
                    f"{classify.loo_1nn(sigs, labels, 'l2'):7.2f}%")
            if args.show_bulk:
                line += f" {classify.measured_cost(table, shapes, dirs, family=fam):9.4f}"
            print(line)

    if "R" not in families:
        print("\nNOTE: no family R rows in these tables, so the rotational baseline is")
        print("      absent and handoff sec 6 test 0 cannot be decided. Produce it with")
        print("      run_pool.py --family R --dirs <same set>.")

    key = "all" if args.subset == "all" else "dev"
    print(f"\npublished IWCIA 2025 Table 2 ({args.subset}), for context only --")
    print("their seconds are 2016-stack reference-implementation timings and are NOT")
    print("comparable to the modelled costs above:")
    print(f"{'set':>10s} {'k':>3s} {'R acc':>7s} {'S acc':>7s}")
    for name, v in PUBLISHED.items():
        print(f"{name:>10s} {v['n']:3d} {v[f'R_{key}']:6.2f}% {v[f'S_{key}']:6.2f}%")
    print("\nS_18/S_9/S_6/S_3 above were linearly interpolated by their authors; the")
    print("exact recomputation differs by up to 4.5 points (handoff sec 3.2.1).")


if __name__ == "__main__":
    main()
