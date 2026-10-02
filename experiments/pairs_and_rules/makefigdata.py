"""Emit the .dat files the pgfplots figures read (whitespace-separated, one
header row of column names) into paths.FIG_DIR.

Every number is recomputed from the feature CSVs or measured on the spot.
"""
from __future__ import annotations
import os, sys, csv, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
import qsig.threadguard  # noqa: F401
import numpy as np
import pickle
FIG = paths.FIG_DIR
os.makedirs(FIG, exist_ok=True)
SLOT = pickle.load(open(os.path.join(HERE, "slotcols.pkl"), "rb"))


def write(name, header, rows):
    with open(os.path.join(FIG, name), "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(str(x) for x in r) + "\n")
    print(f"  {name}: {len(rows)} rows")


def correct(X, y):
    X = np.asarray(X, float); s = X.std(0, keepdims=True); s[s == 0] = 1
    Z = np.ascontiguousarray(X / s, dtype=np.float64)
    sq = (Z * Z).sum(1); d = sq[None, :] - 2.0 * (Z @ Z.T); np.fill_diagonal(d, np.inf)
    return (y[d.argmin(1)] == y)


def ci(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed); d = b.astype(int) - a.astype(int); N = len(d)
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)]) * 100
    return d.mean() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


FEAT = {"mpeg7": "mpeg7_full_features.csv", "animal2000": "animal2000_features.csv",
        "swedishleaves": "swedishleaves_features.csv"}
DS = [("MPEG-7", "mpeg7"), ("Animal2000", "animal2000"), ("SwedishLeaves", "swedishleaves")]


def load(stem, extra=None, pre=None, k=64):
    fm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, FEAT[stem])))}
    ids = list(fm)
    em = None
    if extra:
        em = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, extra)))}
        ids = [s for s in ids if s in em]
    y = np.array([fm[s]["cls"] for s in ids])
    CL = np.array([[float(fm[s][c]) for c in ("area_ratio","circularity","hu1","hu2")] for s in ids])
    SIG = np.array([[float(fm[s][f"sig{i}"]) for i in range(64)] for s in ids])
    EX = None
    if em is not None:
        EX = np.array([[float(em[s][f"{pre}{i}"]) for i in range(k)] for s in ids])
    return y, CL, SIG, EX


# ---------------------------------------------------------------- 1. degeneracy
def deg_rate():
    rows = []
    for name, stem in DS:
        fm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, FEAT[stem])))}
        gm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, f"{stem}_generalpairs.csv")))}
        wm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, f"{stem}_widepairs.csv")))}
        ids = [s for s in fm if s in gm and s in wm]
        G = np.array([[float(gm[s][f"gp{i}"]) for i in range(12)] for s in ids])
        Wd = np.array([[float(wm[s][f"wp{i}"]) for i in range(12)] for s in ids])
        idx = list(range(0, 64, 64 // 12))[:12]
        O = np.array([[float(fm[s][f"sig{i}"]) for i in idx] for s in ids])
        GD = np.array([[float(gm[s][f"gpd{i}"]) for i in range(12)] for s in ids])
        rows.append((name.replace("-", ""), 1, (G == 0).mean()*100, (GD == 0).mean()*100))
        rows.append((name.replace("-", ""), 2, (Wd == 0).mean()*100, 0.0))
        rows.append((name.replace("-", ""), 3, (O == 0).mean()*100, 0.0))
    for nm in {r[0] for r in rows}:
        write(f"deg_{nm}.dat", ["band", "phi", "phiD"],
              [(b, f"{p:.4f}", f"{d:.4f}") for n, b, p, d in rows if n == nm])


# ---------------------------------------------------------------- 2. cost law
def cost_law():
    from math import gcd
    from qsig import pairs, fast
    from qsig.dataset import load_mpeg7, OBJECT, BACKGROUND
    sh = load_mpeg7(paths.data("MPEG7dataset.zip"))[:14]
    imgs = [s.img for s in sh]

    def tm(fn):
        fn(imgs[0])
        t = time.perf_counter()
        for im in imgs:
            fn(im)
        return (time.perf_counter() - t) / len(imgs) * 1000

    orth, gen = [], []
    for v in [(1,0),(2,-1),(3,-1),(3,-2),(4,-1),(5,-2),(5,-3),(7,-2),(7,-4),(7,-6),(9,-5),(11,-3)]:
        s2 = pairs.perp(v)
        rc = min(abs(v[0])+abs(s2[0]), abs(v[1])+abs(s2[1]))
        orth.append((abs(pairs.det(v, s2)), rc, max(max(map(abs,v)), max(map(abs,s2))),
                     f"{tm(lambda im, v=v: fast.compute(im, OBJECT, BACKGROUND, v, 'rows')):.4f}"))
    V = []
    for p in range(0, 9):
        for q in range(-8, 9):
            if (p,q)==(0,0) or p*p+q*q>72 or gcd(abs(p),abs(q))!=1: continue
            if p<0 or (p==0 and q<0): continue
            V.append((p,q))
    seen = set()
    for r in V:
        for s2 in V:
            d = abs(pairs.det(r, s2))
            if d == 0 or d > 60: continue
            k = tuple(sorted([r, s2]))
            if k in seen: continue
            seen.add(k)
            mc = max(max(map(abs,r)), max(map(abs,s2)))
            if mc > 8: continue
            rc = min(abs(r[0])+abs(s2[0]), abs(r[1])+abs(s2[1]))
            gen.append((d, rc, mc,
                        f"{tm(lambda im, r=r, s2=s2: pairs.compute(im, OBJECT, BACKGROUND, r, s2, 'rows')):.4f}"))
            if len(gen) >= 45: break
        if len(gen) >= 45: break
    write("cost_orth.dat", ["det", "rowcost", "maxcoord", "ms"], orth)
    write("cost_gen.dat", ["det", "rowcost", "maxcoord", "ms"], gen)



# ---------------------------------------------------------------- 3. pool
def pool_dat():
    from qsig import directions, pairs as P
    import general_pairs as GP
    POOL = directions.by_angle(directions.pool(max_norm2=130))
    sel = set(SLOT[6])
    rest = [(d.p, d.q) for i, d in enumerate(POOL) if i not in sel]
    chosen = [(POOL[i].p, POOL[i].q) for i in SLOT[6]]
    write("pool_rest.dat", ["p", "q"], rest)
    write("pool_slot6.dat", ["p", "q"], chosen)
    far = GP.spread(12)
    write("pool_farey.dat", ["bisector", "aperture", "maxcoord"],
          [(f"{d['bisector']:.3f}", f"{d['aperture']:.3f}", d["maxcoord"]) for d in far])


# ---------------------------------------------------------------- 4. pareto
def pareto():
    from qsig import directions
    POOL = directions.by_angle(directions.pool(max_norm2=130))
    cost = {k: directions.cost_of([POOL[i] for i in SLOT[k]], "rows+numba") for k in SLOT}
    for name, stem in DS:
        y, CL, SIG, D = load(stem, f"{stem}_disjunctive.csv", "sigd", 64)
        a_rows, b_rows = [], []
        for k in sorted(SLOT):
            a_rows.append((k, f"{cost[k]:.6f}",
                           f"{correct(np.c_[CL, SIG[:, SLOT[k]]], y).mean()*100:.3f}"))
            h = k // 2
            if h in SLOT:
                b_rows.append((k, f"{cost[h]:.6f}",
                               f"{correct(np.c_[CL, SIG[:, SLOT[h]], D[:, SLOT[h]]], y).mean()*100:.3f}"))
        nm = name.replace("-", "")
        write(f"pareto_{nm}_phi.dat", ["k", "cost", "acc"], a_rows)
        write(f"pareto_{nm}_both.dat", ["cols", "cost", "acc"], b_rows)


# ---------------------------------------------------------------- 5. phi_A CIs
def asym_ci():
    rows = []
    for i, (name, stem) in enumerate(DS):
        y, CL, SIG, D = load(stem, f"{stem}_disjunctive.csv", "sigd", 64)
        am = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, f"{stem}_asymmetry.csv")))}
        fm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, FEAT[stem])))}
        dm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, f"{stem}_disjunctive.csv")))}
        ids = [s for s in fm if s in am and s in dm]
        y = np.array([fm[s]["cls"] for s in ids])
        CL = np.array([[float(fm[s][c]) for c in ("area_ratio","circularity","hu1","hu2")] for s in ids])
        S = np.array([[float(fm[s][f"sig{j}"]) for j in SLOT[6]] for s in ids])
        Dm = np.array([[float(dm[s][f"sigd{j}"]) for j in SLOT[6]] for s in ids])
        A = np.array([[float(am[s][f"siga{j}"]) for j in SLOT[6]] for s in ids])
        m, lo, hi = ci(correct(np.c_[CL, S, Dm], y), correct(np.c_[CL, S, A], y))
        rows.append((i + 1, name.replace("-", ""), f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}"))
    write("asym_ci.dat", ["idx", "dataset", "diff", "lo", "hi"], rows)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "deg"): print("degeneracy:"); deg_rate()
    if which in ("all", "cost"): print("cost law:"); cost_law()
    if which in ("all", "pool"): print("pool:"); pool_dat()
    if which in ("all", "pareto"): print("pareto:"); pareto()
    if which in ("all", "ci"): print("phi_A CIs:"); asym_ci()
