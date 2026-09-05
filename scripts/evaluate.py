#!/usr/bin/env python3
"""Accuracy against computation time -- the paper's headline figure, as a table.

Reads a pool table and reports, for each named direction set, the 1NN
leave-one-out accuracy under d_C (and plain L2 as a control) together with the
measured seconds per signature. The rotational family is printed alongside from
IWCIA 2025's published numbers, because it is the frontier that has to be
beaten (handoff sec 3.3) and plotting only the rotation-free row would compare
against the wrong baseline.

    python scripts/evaluate.py --data MPEG7dataset.zip --subset device \
        --table results/device_n30.csv --sets slot9:5 slot12:3.75 cheapest10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401

from qsig import classify, dataset, directions, store  # noqa: E402
from scripts.run_pool import resolve_dirs  # noqa: E402

# IWCIA 2025 Table 2. R = rotational, S = rotation-free.
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
    ap.add_argument("--table", required=True)
    ap.add_argument("--subset", choices=("device", "all"), default="device")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--sets", nargs="+", required=True,
                    help="direction-set specs, e.g. slot9:5 slot20:2.5 cheapest10 sint")
    args = ap.parse_args()

    classes = dataset.DEVICE_CLASSES if args.subset == "device" else None
    shapes = dataset.load_mpeg7(args.data, long_side=args.long_side, classes=classes)
    table = store.load_table(args.table)

    print(f"{args.subset} subset, {len(shapes)} shapes, table {args.table}\n")
    print(f"{'set':16s} {'k':>3s} {'meas s/shape':>13s} {'pred s':>8s} {'maxgap':>7s} "
          f"{'acc dC':>8s} {'acc L2':>8s}")
    print("-" * 70)
    for spec in args.sets:
        dirs = resolve_dirs(spec, 0, 0)
        try:
            sigs, labels, _ = classify.build_signatures(table, shapes, dirs)
        except KeyError as e:
            print(f"{spec:16s} {len(dirs):3d}  SKIPPED -- {e}")
            continue
        cost = classify.measured_cost(table, shapes, dirs)
        pred = directions.total_cost(dirs)
        gap = directions.max_angular_gap(dirs)
        acc_dc = classify.loo_1nn(sigs, labels, "dC")
        acc_l2 = classify.loo_1nn(sigs, labels, "l2")
        print(f"{spec:16s} {len(dirs):3d} {cost:13.1f} {pred:8.1f} {gap:6.1f}d "
              f"{acc_dc:7.2f}% {acc_l2:7.2f}%")

    key_all = args.subset == "all"
    print("\npublished IWCIA 2025 Table 2 " + ("(All)" if key_all else "(Device)") + ":")
    print(f"{'set':16s} {'k':>3s} {'R s/shape':>10s} {'R acc':>7s} {'S s/shape':>10s} {'S acc':>7s}")
    print("-" * 62)
    for name, v in PUBLISHED.items():
        rcost = v["n"] * directions.ROTATIONAL_SECONDS_PER_COMPONENT
        scost = 803.97 if name == "S_int" else float("nan")
        racc = v["R_all"] if key_all else v["R_dev"]
        sacc = v["S_all"] if key_all else v["S_dev"]
        scost_s = f"{scost:10.1f}" if scost == scost else "  interp. "
        print(f"{name:16s} {v['n']:3d} {rcost:10.1f} {racc:6.2f}% {scost_s} {sacc:6.2f}%")
    print("\nNote: S_18/S_9/S_6/S_3 values above were obtained by LINEAR INTERPOLATION\n"
          "of Q-concavity values between neighbouring S_int angles, so their cost is not\n"
          "well defined and their accuracies are approximations. The sets computed from\n"
          "the cached pool above are exact (handoff sec 7.1).")


if __name__ == "__main__":
    main()
