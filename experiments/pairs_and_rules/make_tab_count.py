"""Direction choice versus direction count (Table 4 of the manuscript).

Feature set: classical (4) + phi at k orthogonal directions, 1NN LOO on
z-scored features. Three structured constructions over the 64-direction pool
(|r|^2 <= 130): slot_set(k, 45/k) (the stored slotcols.pkl), cheapest_k(k), and
the exact min-cost max-gap DP (min_cost_maxgap, gap bound scanned in 0.5 deg
steps; the first bound giving exactly k directions is used). K = 4, 6, 8, 10:
the small set sizes at which all three constructions exist.
  spread  = mean over K of (max - min accuracy across the three constructions)
  best    = maximum over constructions and K
  gain    = best - (classical + phi over all 64 directions)
Writes tab/tab_count.tex and count_results.json.
"""
from __future__ import annotations
import csv, json, os, pickle
import paths
import numpy as np
from qsig import directions as Dm

H = paths.HERE
SLOT = pickle.load(open(os.path.join(H, "slotcols.pkl"), "rb"))
K = [4, 6, 8, 10]
DATASETS = [("MPEG-7", "mpeg7_full_features.csv"), ("Animal2000", "animal2000_features.csv"),
            ("Swedish Leaves", "swedishleaves_features.csv")]


def correct(X, y):
    X = np.asarray(X, float); s = X.std(0, keepdims=True); s[s == 0] = 1
    Z = np.ascontiguousarray(X / s, dtype=np.float64)
    sq = (Z * Z).sum(1); d = sq[None, :] - 2.0 * (Z @ Z.T); np.fill_diagonal(d, np.inf)
    return y[d.argmin(1)] == y


def sets():
    pool = Dm.by_angle(Dm.pool(max_norm2=130))
    idx = {(d.p, d.q): i for i, d in enumerate(pool)}
    out = {}
    for k in K:
        out[("slot_set", k)] = list(SLOT[k])
        out[("cheapest", k)] = sorted(idx[(d.p, d.q)] for d in Dm.cheapest_k(k, pool))
    for g in np.arange(4.0, 46.0, 0.5):
        try:
            s = Dm.min_cost_maxgap(float(g), pool)
        except ValueError:          # no set with this small a gap exists in the pool
            continue
        if len(s) in K and ("min-cost DP", len(s)) not in out:
            out[("min-cost DP", len(s))] = sorted(idx[(d.p, d.q)] for d in s)
    missing = [(c, k) for c in ("slot_set", "cheapest", "min-cost DP") for k in K if (c, k) not in out]
    assert not missing, missing
    return out


def main():
    S_ = sets()
    res, lines = {}, []
    for name, fn in DATASETS:
        rows = list(csv.DictReader(open(os.path.join(H, fn))))
        y = np.array([r["cls"] for r in rows])
        CL = np.array([[float(r[c]) for c in ("area_ratio", "circularity", "hu1", "hu2")] for r in rows])
        SIG = np.array([[float(r[f"sig{i}"]) for i in range(64)] for r in rows])
        acc = {f"{c}@{k}": correct(np.c_[CL, SIG[:, cols]], y).mean() * 100 for (c, k), cols in S_.items()}
        all64 = correct(np.c_[CL, SIG], y).mean() * 100
        spread = np.mean([max(acc[f"{c}@{k}"] for c in ("slot_set", "cheapest", "min-cost DP"))
                          - min(acc[f"{c}@{k}"] for c in ("slot_set", "cheapest", "min-cost DP")) for k in K])
        best = max(acc, key=acc.get)
        res[name] = dict(acc={k: round(v, 2) for k, v in acc.items()}, all64=round(all64, 2),
                         spread=round(float(spread), 2), best=best, best_acc=round(acc[best], 2),
                         gain=round(acc[best] - all64, 2))
        kb = best.split("@")[1]
        lines.append(f"{name} & {spread:.2f} & {acc[best]:.2f} ($k{{=}}{kb}$) & {all64:.2f} & ${acc[best]-all64:+.2f}$ \\\\")
        print(name, res[name]["spread"], best, res[name]["best_acc"], res[name]["all64"], res[name]["gain"])
    json.dump(res, open(os.path.join(H, "count_results.json"), "w"), indent=1)
    body = ["\\begin{tabularx}{.8\\textwidth}{@{}lrrrr@{}}", "\\toprule",
            "dataset & spread (pts) & best small set (\\%) & all 64 (\\%) & gain (pts) \\\\", "\\midrule",
            *lines, "\\bottomrule", "\\end{tabularx}"]
    os.makedirs(paths.TAB_DIR, exist_ok=True)
    open(os.path.join(paths.TAB_DIR, "tab_count.tex"), "w").write("\n".join(body) + "\n")


if __name__ == "__main__":
    main()
