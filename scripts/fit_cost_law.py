#!/usr/bin/env python3
"""Fit the cost law and compare it against the published complexity bound.

This is a RESULT of the paper, not a sanity check (handoff sec 7.4 item 11).
It produces the coefficients for T(r) = A + B*|r|^2 on this machine, and the
side-by-side fit quality against the O(mn (r1+r2)^2) bound of [IWCIA 2025,
Thm 1].

Two modes:

  --from-table results/device_n30.csv
      Refit on measured timings from a pool run.

  --published
      Refit on IWCIA 2025 Table 1, reproducing the handoff sec 3.1 numbers:
      |r|^2      R^2 = 0.998, MAPE  4.0%
      (r1+r2)^2  R^2 = 0.903, MAPE 25.3%
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401

import numpy as np  # noqa: E402

from qsig import directions, store  # noqa: E402


def fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a + b * x
    ss = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, ss, float(np.abs((pred - y) / y).mean() * 100), float(np.abs(pred - y).max())


def report(label, x, y):
    a, b, r2, mape, mx = fit(x, y)
    print(f"  {label:28s} T = {a:7.4f} + {b:.5f}*x   R2={r2:.5f}  MAPE={mape:5.2f}%  max err={mx:.3f}s")
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-table")
    g.add_argument("--published", action="store_true")
    ap.add_argument("--statistic", choices=("median", "mean"), default="median")
    args = ap.parse_args()

    if args.published:
        items = sorted(directions.IWCIA_TABLE1.items())
        dirs = [directions.Direction(p, q) for (p, q), _ in items]
        secs = [t for _, t in items]
        src = "IWCIA 2025 Table 1"
    else:
        table = store.load_table(args.from_table)
        agg = {}
        for (shape_id, p, q, res), row in table.items():
            agg.setdefault((p, q), []).append(row["seconds"])
        stat = np.median if args.statistic == "median" else np.mean
        items = sorted(agg.items())
        dirs = [directions.Direction(p, q) for (p, q), _ in items]
        secs = [float(stat(v)) for _, v in items]
        src = f"{args.from_table} ({args.statistic} over shapes)"

    print(f"cost-law fit, {len(dirs)} directions, source: {src}\n")
    a, b = report("|r|^2 = |det(r,r_perp)|", [d.norm2 for d in dirs], secs)
    report("(r1+r2)^2  [published bound]", [(d.p + d.q) ** 2 for d in dirs], secs)

    print(f"\nCOST coefficients for this source: ({a:.4f}, {b:.5f})")
    print("Put these in qsig.directions.COST_IWCIA-style constant for cost predictions.\n")

    print("per-direction residuals (worst 6):")
    pred = [a + b * d.norm2 for d in dirs]
    rows = sorted(zip(dirs, secs, pred), key=lambda t: -abs(t[1] - t[2]))[:6]
    for d, y, p_ in rows:
        print(f"  ({d.p:2d},{d.q:2d}) angle {d.angle:5.1f}  |r|^2={d.norm2:4d}  "
              f"measured {y:7.3f}s  predicted {p_:7.3f}s  resid {y-p_:+7.3f}s")

    sig_cost = sum(a + b * d.norm2 for d in directions.S_INT)
    print(f"\npredicted S_int signature cost under this fit: {sig_cost:.1f} s/shape")
    if args.published:
        print("  (IWCIA 2025 Table 1 total: 803.97 s/shape)")
    rot = len(directions.S_INT) * directions.ROTATIONAL_SECONDS_PER_COMPONENT
    print(f"rotational R_int, 20 components at {directions.ROTATIONAL_SECONDS_PER_COMPONENT} s: {rot:.1f} s/shape")


if __name__ == "__main__":
    main()
