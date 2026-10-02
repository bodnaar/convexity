"""phi_D vs phi: does the disjunctive combination buy anything?

Selection-free protocol:
  - 1NN leave-one-out, paired bootstrap CI over shapes (4000 resamples)
  - LDA, 5-fold stratified CV
  - classical 4d baseline = area_ratio, circularity, hu1, hu2
  - slot_set(k) column subsets, from slotcols.pkl
"""
import sys, os, pickle, csv
import paths  # noqa: F401  -- puts the repository root on sys.path
import qsig.threadguard  # noqa: F401
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

D = os.path.dirname(os.path.abspath(__file__))
SLOT = pickle.load(open(os.path.join(D, "slotcols.pkl"), "rb"))


def load(fn, dfn):
    rows = list(csv.DictReader(open(os.path.join(D, fn))))
    drows = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(D, dfn)))}
    rows = [r for r in rows if r["shape_id"] in drows]
    y = np.array([r["cls"] for r in rows])
    CL = np.array([[float(r[k]) for k in ("area_ratio", "circularity", "hu1", "hu2")] for r in rows])
    SIG = np.array([[float(r[f"sig{i}"]) for i in range(64)] for r in rows])
    SIGD = np.array([[float(drows[r["shape_id"]][f"sigd{i}"]) for i in range(64)] for r in rows])
    return y, CL, SIG, SIGD


def correct(X, y):
    X = np.asarray(X, float); s = X.std(0, keepdims=True); s[s == 0] = 1
    Z = (X / s).astype(np.float32)
    d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
    return (y[d.argmin(1)] == y)


def ci(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed); d = (b.astype(int) - a.astype(int)); N = len(d)
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)]) * 100
    return d.mean() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def lda(X, y):
    return cross_val_score(make_pipeline(StandardScaler(), LDA()), np.asarray(X, float), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() * 100


for name, fn, dfn in [("MPEG-7", "mpeg7_full_features.csv", "mpeg7_disjunctive.csv"),
                      ("Animal2000", "animal2000_features.csv", "animal2000_disjunctive.csv"),
                      ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_disjunctive.csv")]:
    y, CL, SIG, SIGD = load(fn, dfn)
    print(f"\n{'='*78}\n{name}: {len(y)} shapes, {len(set(y))} classes\n{'='*78}")
    print(f"  phi_D range [{SIGD.min():.4f}, {SIGD.max():.4f}]   "
          f"phi range [{SIG.min():.4f}, {SIG.max():.4f}]")
    MU, MUD = SIG.mean(1)[:, None], SIGD.mean(1)[:, None]
    print(f"  corr(mu_phi, mu_phiD) = {np.corrcoef(MU.ravel(), MUD.ravel())[0,1]:+.4f}")
    iu = np.triu_indices(64, k=1)
    cr = np.corrcoef(SIGD, rowvar=False)[iu].mean()
    cc = np.corrcoef(SIGD - MUD, rowvar=False)[iu].mean()
    print(f"  mean pairwise corr of phi_D components: raw {cr:+.3f}   after removing mu {cc:+.3f}")

    K = 12 if 12 in SLOT else sorted(SLOT)[len(SLOT)//2]
    S, SD = SIG[:, SLOT[K]], SIGD[:, SLOT[K]]
    base = correct(CL, y)
    print(f"\n  --- 1NN LOO, paired bootstrap CI vs classical (4d) = {base.mean()*100:.2f}% ---")
    print(f"  {'arm':>40s} {'acc %':>7s} {'gain':>8s} {'95% CI':>18s}")
    arms = [(f"classical + phi   slot_set {K}", np.c_[CL, S]),
            (f"classical + phi_D slot_set {K}", np.c_[CL, SD]),
            (f"classical + BOTH  slot_set {K}", np.c_[CL, S, SD]),
            ("classical + phi   all 64", np.c_[CL, SIG]),
            ("classical + phi_D all 64", np.c_[CL, SIGD]),
            ("classical + mu(phi)", np.c_[CL, MU]),
            ("classical + mu(phi_D)", np.c_[CL, MUD]),
            ("classical + both mus", np.c_[CL, MU, MUD])]
    res = {}
    for lbl, X in arms:
        c = correct(X, y); res[lbl] = c
        m, lo, hi = ci(base, c)
        star = "  *" if not (lo < 0 < hi) else ""
        print(f"  {lbl:>40s} {c.mean()*100:7.2f} {m:+8.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>18s}{star}")

    print("\n  --- head-to-head, paired bootstrap ---")
    for a_lbl, b_lbl in [(f"classical + phi   slot_set {K}", f"classical + phi_D slot_set {K}"),
                         (f"classical + phi   slot_set {K}", f"classical + BOTH  slot_set {K}"),
                         ("classical + phi   all 64", "classical + phi_D all 64"),
                         ("classical + mu(phi)", "classical + mu(phi_D)")]:
        m, lo, hi = ci(res[a_lbl], res[b_lbl])
        star = "  *" if not (lo < 0 < hi) else ""
        print(f"  {b_lbl.strip():>40s} minus {a_lbl.strip():<36s} {m:+7.2f} [{lo:+.2f}, {hi:+.2f}]{star}")

    print("\n  --- k sweep, 1NN LOO (classical + slot_set(k)) ---")
    ks = sorted(SLOT)
    for tag, M in [("phi  ", SIG), ("phi_D", SIGD)]:
        accs = [correct(np.c_[CL, M[:, SLOT[k]]], y).mean() * 100 for k in ks]
        print(f"    {tag}: " + " ".join(f"k={k}:{a:5.1f}" for k, a in zip(ks, accs))
              + f"   best k={ks[int(np.argmax(accs))]} at {max(accs):.2f}")

    print("\n  --- LDA, 5-fold stratified CV ---")
    print(f"    classical {lda(CL,y):5.2f}   +phi{K} {lda(np.c_[CL,S],y):5.2f}   "
          f"+phi_D{K} {lda(np.c_[CL,SD],y):5.2f}   +both{K} {lda(np.c_[CL,S,SD],y):5.2f}   "
          f"| +phi64 {lda(np.c_[CL,SIG],y):5.2f}  +phi_D64 {lda(np.c_[CL,SIGD],y):5.2f}")
