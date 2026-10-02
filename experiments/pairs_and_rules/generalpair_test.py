"""Does phi_D's complementarity survive NON-ORTHOGONAL direction pairs?

The whole phi_D / phi_A result so far was measured on orthogonal pairs, so
qsig/pairs.py's general-pair machinery was never exercised by the main claim.
This tests the claim on a set of Farey-neighbour pairs (|det| = 1, aperture
10-85 deg, spread over bisector orientation) -- see general_pairs.py.

The comparison that matters is WITHIN the general-pair set: phi alone against
phi + phi_D at matched column count. Comparing the general set to the
orthogonal set would confound non-orthogonality with aperture (Farey pairs run
10-45 deg, orthogonal pairs are 90), so that comparison is reported separately
and read only as context.
"""
from __future__ import annotations
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

K = 12


def load(fn, gfn):
    rows = list(csv.DictReader(open(os.path.join(HERE, fn))))
    gm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, gfn)))}
    rows = [r for r in rows if r["shape_id"] in gm]
    y = np.array([r["cls"] for r in rows])
    CL = np.array([[float(r[k]) for k in ("area_ratio", "circularity", "hu1", "hu2")] for r in rows])
    G = lambda pre: np.array([[float(gm[r["shape_id"]][f"{pre}{i}"]) for i in range(K)] for r in rows])
    return y, CL, G("gp"), G("gpd"), G("gpa")


def correct(X, y):
    X = np.asarray(X, float)
    s = X.std(0, keepdims=True); s[s == 0] = 1
    Z = np.ascontiguousarray(X / s, dtype=np.float64)
    sq = (Z * Z).sum(1)
    d = sq[None, :] - 2.0 * (Z @ Z.T)
    np.fill_diagonal(d, np.inf)
    return (y[d.argmin(1)] == y)


def ci(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = b.astype(int) - a.astype(int); N = len(d)
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)]) * 100
    return d.mean() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def lda(X, y):
    return cross_val_score(make_pipeline(StandardScaler(), LDA()), np.asarray(X, float), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() * 100


def rf(X, y):
    return cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                           np.asarray(X, float), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() * 100


DATA = [("MPEG-7", "mpeg7_full_features.csv", "mpeg7_generalpairs.csv"),
        ("Animal2000", "animal2000_features.csv", "animal2000_generalpairs.csv"),
        ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_generalpairs.csv")]

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn, gfn in DATA:
        if which != "all" and which.lower() not in name.lower():
            continue
        y, CL, P, D, A = load(fn, gfn)
        h = K // 2
        print(f"\n{'='*76}\n{name}: {len(y)} shapes, {len(set(y))} classes, {K} general pairs\n{'='*76}")
        print(f"  phi   range [{P.min():.4f}, {P.max():.4f}]")
        print(f"  phi_D range [{D.min():.4f}, {D.max():.4f}]")
        print(f"  phi_A range [{A.min():.4f}, {A.max():.4f}]")
        print(f"  corr(mu_phi, mu_phiD) = {np.corrcoef(P.mean(1), D.mean(1))[0,1]:+.3f}   "
              f"corr(mu_phi, mu_phiA) = {np.corrcoef(P.mean(1), A.mean(1))[0,1]:+.3f}")

        base = correct(CL, y)
        arms = [("classical", CL),
                (f"+ phi   (all {K})", np.c_[CL, P]),
                (f"+ phi_D (all {K})", np.c_[CL, D]),
                (f"+ phi_A (all {K})", np.c_[CL, A]),
                (f"+ phi (first {h}) ", np.c_[CL, P[:, :h]]),
                (f"+ phi+phi_D ({h}+{h})", np.c_[CL, P[:, :h], D[:, :h]]),
                (f"+ phi+phi_A ({h}+{h})", np.c_[CL, P[:, :h], A[:, :h]])]
        res = {}
        print(f"\n  --- 1NN LOO, paired bootstrap vs classical ---")
        print(f"  {'arm':>22s} {'cols':>4s} {'acc %':>7s} {'gain':>8s} {'95% CI':>18s}")
        for lbl, X in arms:
            c = correct(X, y); res[lbl] = c
            m, lo, hi = ci(base, c)
            star = "  *" if not (lo < 0 < hi) else ""
            print(f"  {lbl:>22s} {X.shape[1]-4:4d} {c.mean()*100:7.2f} {m:+8.2f} "
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>18s}{star}")

        print(f"\n  --- THE TEST: matched column count, {K} signature columns each ---")
        for a_lbl, b_lbl in [(f"+ phi   (all {K})", f"+ phi+phi_D ({h}+{h})"),
                             (f"+ phi   (all {K})", f"+ phi+phi_A ({h}+{h})")]:
            m, lo, hi = ci(res[a_lbl], res[b_lbl])
            star = "  *" if not (lo < 0 < hi) else ""
            print(f"  {b_lbl.strip():>22s} minus {a_lbl.strip():<18s} {m:+7.2f} [{lo:+.2f}, {hi:+.2f}]{star}")

        print(f"\n  --- LDA / RF, 5-fold stratified CV ---")
        print(f"  {'arm':>22s} {'LDA':>7s} {'RF':>7s}")
        for lbl, X in arms:
            print(f"  {lbl:>22s} {lda(X,y):7.2f} {rf(X,y):7.2f}")


if __name__ == "__main__":
    main()
