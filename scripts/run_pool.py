#!/usr/bin/env python3
"""Compute the Q-concavity pool: every (shape, direction) pair, once.

Append-only and resumable -- re-running skips pairs already in the table, so a
multi-hour run survives a crash or a reboot.

Examples
--------
Device subset, cheap end of the pool, all cores:
    python scripts/run_pool.py --data MPEG7dataset.zip --subset device \
        --max-norm2 30 --workers 20 --out results/device_n30.csv

Full MPEG-7, same directions, nice'd:
    python scripts/run_pool.py --data MPEG7dataset.zip --subset all \
        --max-norm2 30 --workers 19 --out results/all_n30.csv

Reproduce IWCIA 2025's own 20 directions (expensive: ~800 s/shape):
    python scripts/run_pool.py --data MPEG7dataset.zip --subset device \
        --dirs sint --workers 20 --out results/device_sint.csv

Measurement runs that produce PUBLISHED timings should be pinned and run under
the `performance` governor (handoff sec 7.2):
    sudo cpupower frequency-set -g performance
    numactl --cpunodebind=0 python scripts/run_pool.py ... --workers 10 --pinned
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401  -- MUST precede numpy

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

from qsig import dataset, directions, store  # noqa: E402
from qsig.descriptor import q_concavity  # noqa: E402

_SHAPES = {}


def _init(shapes):
    for sh in shapes:
        _SHAPES[sh.shape_id] = sh


def _job(args):
    shape_id, p, q, resolution = args
    sh = _SHAPES[shape_id]
    d = directions.Direction(p, q)
    value, secs = q_concavity(sh.img, d)
    return {
        "shape_id": shape_id, "cls": sh.cls, "p": p, "q": q,
        "angle_deg": round(d.angle, 6), "norm2": d.norm2,
        "resolution": resolution, "E": value, "seconds": secs,
    }


def resolve_dirs(spec: str, max_norm2: int, max_component: int):
    if spec == "sint":
        return directions.S_INT
    if spec == "pool":
        if max_component:
            return directions.by_angle(directions.pool(max_component=max_component))
        return directions.by_angle(directions.pool(max_norm2=max_norm2))
    if spec.startswith("slot"):          # e.g. slot20:2.5
        body = spec.split(":", 1)
        n = int(body[0][4:])
        tol = float(body[1]) if len(body) > 1 else 90.0 / n / 2
        return directions.slot_set(n, tol)
    if spec.startswith("cheapest"):      # e.g. cheapest10
        return directions.cheapest_k(int(spec[8:]))
    raise SystemExit(f"unknown --dirs {spec!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="MPEG7dataset.zip or an extracted directory")
    ap.add_argument("--subset", choices=("device", "all"), default="device")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--dirs", default="pool",
                    help="pool | sint | slot<N>[:<tol_deg>] | cheapest<K>")
    ap.add_argument("--max-norm2", type=int, default=30, help="cost budget for --dirs pool")
    ap.add_argument("--max-component", type=int, default=0,
                    help="box budget F_Q instead of a disc; reproduces deg2vec's pool")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    ap.add_argument("--pinned", action="store_true", help="record that workers were CPU-pinned")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N shapes")
    args = ap.parse_args()

    classes = dataset.DEVICE_CLASSES if args.subset == "device" else None
    shapes = dataset.load_mpeg7(args.data, long_side=args.long_side, classes=classes)
    if args.limit:
        shapes = shapes[: args.limit]
    dirs = resolve_dirs(args.dirs, args.max_norm2, args.max_component)

    st = store.ResultStore(args.out, repo_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           pinned=args.pinned)
    st.write_meta({
        **dataset.protocol_metadata(args.long_side),
        "subset": args.subset,
        "dirs_spec": args.dirs,
        "n_shapes": len(shapes),
        "n_directions": len(dirs),
        "directions": [[d.p, d.q] for d in dirs],
        "workers": args.workers,
        "threadguard": qsig.threadguard.report(),
    })

    done = st.done_keys()
    todo = [
        (sh.shape_id, d.p, d.q, args.long_side)
        for sh in shapes for d in dirs
        if (sh.shape_id, d.p, d.q, args.long_side) not in done
    ]
    predicted = len(shapes) * directions.total_cost(dirs) / max(1, args.workers)
    print(f"{len(shapes)} shapes x {len(dirs)} directions = {len(shapes)*len(dirs)} jobs; "
          f"{len(done)} already done, {len(todo)} to run")
    print(f"governor={st.governor} host={st.host} commit={st.commit} workers={args.workers}")
    print(f"predicted wall clock from the IWCIA-fitted cost law: {predicted/60:.1f} min "
          f"(refit on this machine before quoting -- scripts/fit_cost_law.py)")
    if not todo:
        return

    t0 = time.time()
    written = 0
    buf = []
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(args.workers, initializer=_init, initargs=(shapes,)) as pool_:
        for i, row in enumerate(pool_.imap_unordered(_job, todo, chunksize=1), 1):
            buf.append(row)
            if len(buf) >= 200:
                written += st.append(buf)
                buf.clear()
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el/60:.1f} min elapsed, "
                      f"eta {(el/i)*(len(todo)-i)/60:.1f} min", flush=True)
    if buf:
        written += st.append(buf)
    print(f"wrote {written} rows to {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
