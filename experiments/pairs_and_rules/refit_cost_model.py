"""Refit the per-direction cost model of Table 4 from overhead_phiD.json.

overhead_phiD.py stores the median time of one row-prefix (JIT) evaluation for
every direction of the 64-direction pool, measured pinned and single-threaded.
This script fits  t(p,q) = a + b*(p+q)  [ms] on those medians, prints the new
constants for qsig.directions.COST_MODELS["rows+numba"], and the rows of
Table 4 (modeled s/shape and speed-up for the full pool and the slot sets),
plus the cost of the sets relative to the full pool with and without phi_D.

Usage:  python3 refit_cost_model.py [path/to/overhead_phiD.json]
"""
from __future__ import annotations
import paths  # noqa: F401
import json, os, pickle, sys
from qsig import directions

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.HERE, "overhead_phiD.json")
d = json.load(open(src))
per = d["per_direction"]
x = [v["p"] + v["q"] for v in per.values()]
y = [v["full"] * 1e3 for v in per.values()]
n = len(x); mx = sum(x) / n; my = sum(y) / n
b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
a = my - b * mx
pred = [a + b * xi for xi in x]
r2 = 1 - sum((yi - pi) ** 2 for yi, pi in zip(y, pred)) / sum((yi - my) ** 2 for yi in y)
print(f"source {src}  host {d.get('host')}  shapes {d.get('shapes')}  repeats {d.get('repeats')}")
print(f"t(p,q) = {a:.3f} + {b:.4f}*(p+q) ms   R^2 = {r2:.4f}   (n = {n})")
print(f'COST_MODELS["rows+numba"] = ("pq", {a/1e3:.5f}, {b/1e3:.6f})\n')

pool = directions.by_angle(directions.pool(max_norm2=130))
slot = pickle.load(open(os.path.join(paths.HERE, "slotcols.pkl"), "rb"))
sets = {"full pool (64)": pool}
sets.update({f"slot_set({k})": [pool[i] for i in slot[k]] for k in (4, 6, 10, 12, 24)})
cost = lambda ds: sum(a + b * (q.p + q.q) for q in ds) / 1e3
full = cost(pool)
ovh = d.get("per_set_percent", {})
print(f"{'direction set':16s} {'sum(p+q)':>8s} {'s/shape':>9s} {'speed-up':>9s} {'% of full':>9s} {'% of full, +phi_D':>18s}")
for name, ds in sets.items():
    c = cost(ds); o = ovh.get(name, 0.0)
    print(f"{name:16s} {sum(q.p + q.q for q in ds):8d} {c:9.4f} {full / c:8.1f}x {100 * c / full:8.1f}% {100 * c * (1 + o / 100) / full:17.1f}%")
