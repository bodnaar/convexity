#!/usr/bin/env python3
"""Classical shape descriptors, one row per shape -- the other half of the
second-benchmark replication (the Q-concavity signature half is
`run_pool.py --dirs pool --max-norm2 130`).

    python scripts/classical_features.py --data Animal2000-GT.zip \
        --dataset animal2000 --out results/animal2000_classical.csv

    python scripts/classical_features.py --data MPEG7dataset.zip \
        --dataset mpeg7 --subset all --out results/mpeg7_classical.csv

Append-only and resumable, same spirit as `run_pool.py` / `qsig.store` but a
different schema: one row per SHAPE, not per (shape, direction), since
classical descriptors don't depend on a direction set. See
`journal_paper/second_benchmark_2026-09-13.md` and
`journal_paper/second_benchmark_results_2026-09-13.md` for why this exists.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401  -- MUST precede numpy

from qsig import dataset, skeview  # noqa: E402
from qsig.classical import FEATURE_NAMES, classical_features  # noqa: E402
from qsig.store import _git_commit, library_versions  # noqa: E402

FIELDS = ["shape_id", "cls", "dataset"] + FEATURE_NAMES + [
    "host", "git_commit",
]


def load_shapes(args):
    if args.dataset == "mpeg7":
        classes = dataset.DEVICE_CLASSES if args.subset == "device" else None
        return dataset.load_mpeg7(args.data, long_side=args.long_side, classes=classes)
    if args.subset == "device":
        raise SystemExit(f"--dataset {args.dataset} has no Device subset; pass --subset all")
    return skeview.load_skeview(args.data, args.dataset, long_side=args.long_side)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", choices=("mpeg7", "animal2000", "swedishleaves"), default="mpeg7")
    ap.add_argument("--subset", choices=("device", "all"), default="all")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N shapes")
    args = ap.parse_args()

    shapes = load_shapes(args)
    if args.limit:
        shapes = shapes[: args.limit]
    print(f"{args.dataset}: {len(shapes)} shapes, {len({s.cls for s in shapes})} classes")

    done = set()
    write_header = True
    if os.path.exists(args.out):
        with open(args.out, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("dataset") == args.dataset:
                    done.add(row["shape_id"])
        write_header = os.path.getsize(args.out) == 0

    import platform
    host = platform.node()
    commit = _git_commit(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    todo = [s for s in shapes if s.shape_id not in done]
    print(f"{len(done)} already done, {len(todo)} to compute")
    if not todo:
        return

    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for i, sh in enumerate(todo, 1):
            feats = classical_features(sh.img)
            row = {"shape_id": sh.shape_id, "cls": sh.cls, "dataset": args.dataset,
                   "host": host, "git_commit": commit}
            row.update(zip(FEATURE_NAMES, feats))
            w.writerow(row)
            if i % 100 == 0 or i == len(todo):
                fh.flush()
                print(f"  {i}/{len(todo)}", flush=True)

    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    import json
    meta = {"dataset": args.dataset, "subset": args.subset, "long_side": args.long_side,
             "host": host, "git_commit": commit, "libraries": library_versions()}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"wrote {len(todo)} rows to {args.out}")


if __name__ == "__main__":
    main()
