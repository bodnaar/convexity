"""Q-convexity on top of Fourier descriptors (FD), fixed k, orthogonal pairs.

Feature sets: classical, classical+FD, classical+FD+phi (slot_set(12)),
classical+FD+phi+phi_D (slot_set(6), the same 12 columns), plus FD-only arms.
Protocol as everywhere else: 1NN LOO on z-scored features with a paired
bootstrap (4000 resamples, seed 0); LDA (scaled) and random forest (300 trees,
seed 0) on StratifiedKFold(5, shuffle, seed 0). M = 12 FD coefficients is the
primary setting (all three classifiers); M = 8, 16, 32 are 1NN robustness
checks. Needs extract_fourier.py output. Writes/updates fourier_results.json;
optional arguments restrict the run to the named datasets (MPEG-7, Animal2000,
SwedishLeaves), so it can be run one dataset at a time.
"""
from __future__ import annotations
import csv, json, os, pickle, sys
import paths  # noqa: F401
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = paths.HERE
SLOT = pickle.load(open(os.path.join(HERE, "slotcols.pkl"), "rb"))
DATASETS = [("MPEG-7", "mpeg7", "mpeg7_full_features.csv", "mpeg7_disjunctive.csv"),
            ("Animal2000", "animal2000", "animal2000_features.csv", "animal2000_disjunctive.csv"),
            ("SwedishLeaves", "swedishleaves", "swedishleaves_features.csv", "swedishleaves_disjunctive.csv")]
MS = [12, 8, 16, 32]


def load(stem, fn, dfn):
    rd = lambda f: list(csv.DictReader(open(os.path.join(HERE, f))))
    rows, dr = rd(fn), {r["shape_id"]: r for r in rd(dfn)}
    fr = {r["shape_id"]: r for r in rd(f"{stem}_fourier.csv")}
    rows = [r for r in rows if r["shape_id"] in dr and r["shape_id"] in fr]
    assert all(fr[r["shape_id"]]["cls"] == r["cls"] for r in rows)
    y = np.array([r["cls"] for r in rows])
    CL = np.array([[float(r[k]) for k in ("area_ratio", "circularity", "hu1", "hu2")] for r in rows])
    S = np.array([[float(r[f"sig{i}"]) for i in range(64)] for r in rows])
    SD = np.array([[float(dr[r["shape_id"]][f"sigd{i}"]) for i in range(64)] for r in rows])
    FD = np.array([[float(fr[r["shape_id"]][f"fd{m}"]) for m in range(1, 33)] for r in rows])
    return y, CL, S, SD, FD


def load_ids(stem, fn, dfn):
    """Shape ids in the row order of load()."""
    rd = lambda f: list(csv.DictReader(open(os.path.join(HERE, f))))
    dr = {r["shape_id"] for r in rd(dfn)}
    fr = {r["shape_id"] for r in rd(f"{stem}_fourier.csv")}
    return [r["shape_id"] for r in rd(fn) if r["shape_id"] in dr and r["shape_id"] in fr]


def correct(X, y):
    X = np.asarray(X, float); s = X.std(0, keepdims=True); s[s == 0] = 1
    Z = np.ascontiguousarray(X / s, dtype=np.float64)
    sq = (Z * Z).sum(1); d = sq[None, :] - 2.0 * (Z @ Z.T); np.fill_diagonal(d, np.inf)
    return y[d.argmin(1)] == y


def ci(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed); d = b.astype(int) - a.astype(int); N = len(d)
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)]) * 100
    return [round(d.mean() * 100, 2), round(float(np.percentile(bs, 2.5)), 2),
            round(float(np.percentile(bs, 97.5)), 2)]


def main():
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    path = os.path.join(HERE, "fourier_results.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    only = set(sys.argv[1:])
    for name, stem, fn, dfn in DATASETS:
        if only and name not in only:
            continue
        y, CL, S, SD, FDall = load(stem, fn, dfn)
        lda = lambda X: round(cross_val_score(make_pipeline(StandardScaler(), LDA()), X, y, cv=cv).mean() * 100, 2)
        rf = lambda X: round(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                                             X, y, cv=cv).mean() * 100, 2)
        p12, b6 = S[:, SLOT[12]], np.c_[S[:, SLOT[6]], SD[:, SLOT[6]]]
        res = {"n": int(len(y))}
        for M in MS:
            F = FDall[:, :M]
            arms = {"classical": CL, "cl+phi12": np.c_[CL, p12], "cl+both6": np.c_[CL, b6],
                    "FD": F, "cl+FD": np.c_[CL, F], "cl+FD+phi12": np.c_[CL, F, p12],
                    "cl+FD+both6": np.c_[CL, F, b6], "FD+phi12": np.c_[F, p12], "FD+both6": np.c_[F, b6]}
            c = {k: correct(X, y) for k, X in arms.items()}
            acc = {k: {"1nn": round(v.mean() * 100, 2)} for k, v in c.items()}
            if M == MS[0]:
                for k, X in arms.items():
                    acc[k]["lda"], acc[k]["rf"] = lda(X), rf(X)
            tests = {"cl+FD|classical": ci(c["classical"], c["cl+FD"]),
                     "cl+FD+phi12|cl+FD": ci(c["cl+FD"], c["cl+FD+phi12"]),
                     "cl+FD+both6|cl+FD": ci(c["cl+FD"], c["cl+FD+both6"]),
                     "cl+FD+both6|cl+FD+phi12": ci(c["cl+FD+phi12"], c["cl+FD+both6"]),
                     "cl+both6|cl+FD": ci(c["cl+FD"], c["cl+both6"]),
                     "FD+both6|FD": ci(c["FD"], c["FD+both6"])}
            res[f"M={M}"] = {"acc": acc, "tests": tests}
        out[name] = res
        t = res["M=12"]["tests"]
        print(f"{name}: cl+FD {res['M=12']['acc']['cl+FD']['1nn']}  "
              + "  ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for k, v in t.items()), flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
