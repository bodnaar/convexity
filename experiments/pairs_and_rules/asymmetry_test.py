"""Does phi_A reproduce what phi_D adds to phi?

The algebraic account: with u = n0 n3 and v = n1 n2,
    phi = u v,  phi_D = u + v,  phi_A = |u - v| = sqrt(phi_D^2 - 4 phi)
POINTWISE. phi and phi_D are the two elementary symmetric polynomials of the
same pair, so what phi_D adds to phi is exactly the BALANCE between the two
opposing cone-pair products -- and phi_A is that balance, isolated.

The identity is exact before aggregation. The descriptors are sums over
background points of differently normalised per-point quantities, so it does
not automatically survive aggregation. This script tests whether it does:

    if  classical + phi + phi_A  ~=  classical + phi + phi_D
    then the account holds at the aggregate level.

Protocol as in disjmeasure.py: classical 4d baseline,
equiangular slot_set(k) subsets fixed before seeing labels, 1NN LOO with paired
bootstrap CIs over shapes, LDA and random forest with 5-fold stratified CV, and
everything compared at MATCHED COLUMN COUNT.
"""
from __future__ import annotations
import os, sys, pickle, csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: F401,E402
import numpy as np  # BLAS multithreaded: accuracy run, not a timing run
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SB = HERE
SLOT = pickle.load(open(os.path.join(SB, "slotcols.pkl"), "rb"))

DATASETS = [("MPEG-7", "mpeg7_full_features.csv", "mpeg7_disjunctive.csv", "mpeg7_asymmetry.csv"),
            ("Animal2000", "animal2000_features.csv", "animal2000_disjunctive.csv", "animal2000_asymmetry.csv"),
            ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_disjunctive.csv", "swedishleaves_asymmetry.csv")]


def load(fn, dfn, afn):
    rows = list(csv.DictReader(open(os.path.join(SB, fn))))
    dm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(SB, dfn)))}
    am = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(SB, afn)))}
    rows = [r for r in rows if r["shape_id"] in dm and r["shape_id"] in am]
    y = np.array([r["cls"] for r in rows])
    CL = np.array([[float(r[k]) for k in ("area_ratio", "circularity", "hu1", "hu2")] for r in rows])
    SIG = np.array([[float(r[f"sig{i}"]) for i in range(64)] for r in rows])
    SIGD = np.array([[float(dm[r["shape_id"]][f"sigd{i}"]) for i in range(64)] for r in rows])
    SIGA = np.array([[float(am[r["shape_id"]][f"siga{i}"]) for i in range(64)] for r in rows])
    return y, CL, SIG, SIGD, SIGA


def correct(X, y):
    """1NN LOO, z-scored, float64 gram identity (float32 flips near-ties)."""
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


K = 6
for name, fn, dfn, afn in DATASETS:
    y, CL, SIG, SIGD, SIGA = load(fn, dfn, afn)
    c = SLOT[K]
    A, Dm, Am = SIG[:, c], SIGD[:, c], SIGA[:, c]
    print(f"\n{'='*78}\n{name}: {len(y)} shapes, {len(set(y))} classes  (k={K})\n{'='*78}")
    print(f"  phi_A range [{SIGA.min():.4f}, {SIGA.max():.4f}]   "
          f"corr(mu_phiD, mu_phiA) = {np.corrcoef(SIGD.mean(1), SIGA.mean(1))[0,1]:+.3f}   "
          f"corr(mu_phi, mu_phiA) = {np.corrcoef(SIG.mean(1), SIGA.mean(1))[0,1]:+.3f}")

    base = correct(CL, y)
    arms = [("classical", CL),
            (f"classical + phi",            np.c_[CL, A]),
            (f"classical + phi_D",          np.c_[CL, Dm]),
            (f"classical + phi_A",          np.c_[CL, Am]),
            (f"classical + phi + phi_D",    np.c_[CL, A, Dm]),
            (f"classical + phi + phi_A",    np.c_[CL, A, Am]),
            (f"classical + phi_D + phi_A",  np.c_[CL, Dm, Am]),
            (f"classical + all three",      np.c_[CL, A, Dm, Am])]
    res = {}
    print(f"\n  --- 1NN LOO, paired bootstrap CI vs classical ---")
    print(f"  {'arm':>28s} {'cols':>4s} {'acc %':>7s} {'gain':>8s} {'95% CI':>18s}")
    for lbl, X in arms:
        cc = correct(X, y); res[lbl] = cc
        m, lo, hi = ci(base, cc)
        star = "  *" if not (lo < 0 < hi) else ""
        print(f"  {lbl:>28s} {X.shape[1]-4:4d} {cc.mean()*100:7.2f} {m:+8.2f} "
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18s}{star}")

    print("\n  --- THE TEST: phi+phi_A against phi+phi_D (same 12 columns) ---")
    m, lo, hi = ci(res["classical + phi + phi_D"], res["classical + phi + phi_A"])
    verdict = ("INDISTINGUISHABLE -- the algebraic account survives aggregation"
               if lo < 0 < hi else
               ("phi_A is BETTER" if m > 0 else "phi_A is WORSE -- account does not transfer"))
    print(f"  (phi+phi_A) - (phi+phi_D) = {m:+.2f} [{lo:+.2f}, {hi:+.2f}]   {verdict}")

    print("\n  --- other classifiers (5-fold stratified CV) ---")
    print(f"  {'arm':>28s} {'LDA':>7s} {'RF':>7s}")
    for lbl, X in arms:
        print(f"  {lbl:>28s} {lda(X,y):7.2f} {rf(X,y):7.2f}")
