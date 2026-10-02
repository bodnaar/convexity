"""Generate the results tables as LaTeX fragments.

Emits .tex fragments into paths.TAB_DIR, each a bare `tabular` inside a `table`
environment with a \caption and \label, ready for \input. Everything is
computed from the CSVs, except the 1NN and LDA columns of tab_matched, which
are transcribed from the output of disjmeasure.py, disjdim.py and disjhead.py.
"""
from __future__ import annotations
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import paths  # noqa: E402
OUT = paths.TAB_DIR
os.makedirs(OUT, exist_ok=True)

DS = [("MPEG-7", "mpeg7"), ("Animal2000", "animal2000"), ("SwedishLeaves", "swedishleaves")]
FEAT = {"mpeg7": "mpeg7_full_features.csv", "animal2000": "animal2000_features.csv",
        "swedishleaves": "swedishleaves_features.csv"}


def correct(X, y):
    X = np.asarray(X, float); s = X.std(0, keepdims=True); s[s == 0] = 1
    Z = np.ascontiguousarray(X / s, dtype=np.float64)
    sq = (Z * Z).sum(1); d = sq[None, :] - 2.0 * (Z @ Z.T); np.fill_diagonal(d, np.inf)
    return (y[d.argmin(1)] == y)


def ci(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed); d = b.astype(int) - a.astype(int); N = len(d)
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)]) * 100
    return d.mean() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def base(stem):
    fm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, FEAT[stem])))}
    return fm


def band(stem, which):
    if which == "narrow":
        f, pre = f"{stem}_generalpairs.csv", "gp"
    elif which == "wide":
        f, pre = f"{stem}_widepairs.csv", "wp"
    else:
        return None
    m = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, f)))}
    return m, pre


def wrap(body, caption, label, note=None):
    s = ["\\begin{table}[t]", "\\centering",
         f"\\caption{{{caption}}}\\label{{{label}}}", body]
    if note:
        # plain footnote line: \begin{tablenotes} needs threeparttable, which
        # the sn-jnl template does not load.
        s.append("\\vspace{2pt}")
        s.append("\\begin{minipage}{\\textwidth}\\footnotesize " + note + "\\end{minipage}")
    s.append("\\end{table}")
    return "\n".join(s) + "\n"


# ---- Table: degeneracy by aperture band -----------------------------------
def tab_degeneracy():
    rows = []
    for name, stem in DS:
        fm = base(stem)
        gm, gp = band(stem, "narrow"); wm, wp = band(stem, "wide")
        dm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, f"{stem}_disjunctive.csv")))}
        ids = [s for s in fm if s in gm and s in wm and s in dm]
        G = np.array([[float(gm[s][f"gp{i}"]) for i in range(12)] for s in ids])
        W = np.array([[float(wm[s][f"wp{i}"]) for i in range(12)] for s in ids])
        idx = list(range(0, 64, 64 // 12))[:12]
        O = np.array([[float(fm[s][f"sig{i}"]) for i in idx] for s in ids])
        GD = np.array([[float(gm[s][f"gpd{i}"]) for i in range(12)] for s in ids])
        WD = np.array([[float(wm[s][f"wpd{i}"]) for i in range(12)] for s in ids])
        OD = np.array([[float(dm[s][f"sigd{i}"]) for i in idx] for s in ids])
        y = np.array([fm[s]["cls"] for s in ids])
        CL = np.array([[float(fm[s][c]) for c in ("area_ratio","circularity","hu1","hu2")] for s in ids])
        cells = []
        for P, D in ((G, GD), (W, WD), (O, OD)):
            m, lo, hi = ci(correct(np.c_[CL, P], y), correct(np.c_[CL, D], y))
            star = "$^{*}$" if not (lo < 0 < hi) else ""
            cells.append((f"{(P==0).mean()*100:.2f}", f"{(D==0).mean()*100:.2f}",
                          f"${m:+.2f}${star}"))
        rows.append((name, cells))
    b = ["\\begin{tabular}{@{}l" + "ccc" * 3 + "@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{narrow, $11$--$27^\\circ$} & "
         "\\multicolumn{3}{c}{wide, $53$--$79^\\circ$} & "
         "\\multicolumn{3}{c}{orthogonal, $90^\\circ$} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "dataset & $\\varphi{=}0$ & $\\varphi_D{=}0$ & $\\Delta$ & "
         "$\\varphi{=}0$ & $\\varphi_D{=}0$ & $\\Delta$ & "
         "$\\varphi{=}0$ & $\\varphi_D{=}0$ & $\\Delta$ \\\\", "\\midrule"]
    for name, cells in rows:
        b.append(name + " & " + " & ".join(x for c in cells for x in c) + " \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    return wrap("\n".join(b),
                "Degeneracy of the conjunctive product by aperture band, and the "
                "resulting accuracy difference. $\\varphi{=}0$ and $\\varphi_D{=}0$ are "
                "percentages of shape\\,$\\times$\\,pair descriptor values that vanish; "
                "$\\Delta$ is (classical\\,$+\\,\\varphi_D$) minus (classical\\,$+\\,\\varphi$), "
                "1NN leave-one-out, $12$ signature columns each.",
                "tab:degeneracy",
                "$^{*}$ outside the $95\\%$ paired bootstrap interval "
                "($4000$ resamples over shapes).")


# ---- Table: matched-dimension, three classifiers (orthogonal set) ---------
def tab_matched():
    rf = {}
    with open(os.path.join(HERE, "rf_results.csv")) as f:
        for r in csv.DictReader(f):
            rf[(r["dataset"], r["arm"])] = float(r["acc"])
    # 1NN and LDA: transcribed from disjmeasure.py / disjdim.py / disjhead.py
    # output (1NN there uses a float32 distance). RF: rf_results.csv (rfcheck.py).
    lda = {"MPEG-7": dict(classical=56.43, phi12=64.07, phiD12=65.93, both6=69.79, both12=69.57, phi24=64.86),
           "Animal2000": dict(classical=33.15, phi12=38.00, phiD12=37.25, both6=41.95, both12=41.35, phi24=37.35),
           "SwedishLeaves": dict(classical=74.67, phi12=73.69, phiD12=78.58, both6=79.64, both12=80.44, phi24=74.67)}
    nn = {"MPEG-7": dict(classical=73.07, phi12=79.21, phiD12=79.14, both6=82.57, both12=82.07, phi24=78.36),
          "Animal2000": dict(classical=31.90, phi12=37.90, phiD12=34.80, both6=41.85, both12=41.30, phi24=36.40),
          "SwedishLeaves": dict(classical=78.31, phi12=81.16, phiD12=81.07, both6=82.31, both12=81.69, phi24=80.71)}
    armmap = [("classical", "classical", 0), ("phi12", "phi@12", 12), ("phiD12", "phi_D@12", 12),
              ("both6", "both@6", 12), ("phi24", "phi@24", 24), ("both12", "both@12", 24)]
    show = {"classical": "classical only", "phi12": "$+\\,\\varphi$ ($k{=}12$)",
            "phiD12": "$+\\,\\varphi_D$ ($k{=}12$)", "both6": "$+\\,\\varphi,\\varphi_D$ ($k{=}6$)",
            "phi24": "$+\\,\\varphi$ ($k{=}24$)", "both12": "$+\\,\\varphi,\\varphi_D$ ($k{=}12$)"}
    b = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
         "dataset & feature set & cols & 1NN & LDA & RF \\\\", "\\midrule"]
    for name, _ in DS:
        first = True
        for key, rfkey, cols in armmap:
            lbl = show[key]
            b.append(f"{name if first else ''} & {lbl} & {cols} & "
                     f"{nn[name][key]:.2f} & {lda[name][key]:.2f} & {rf[(name, rfkey)]:.2f} \\\\")
            first = False
        b.append("\\midrule" if name != DS[-1][0] else "")
    b = [x for x in b if x != ""]
    b += ["\\bottomrule", "\\end{tabular}"]
    return wrap("\n".join(b),
                "Matched-dimension comparison on orthogonal pairs. Arms with the same "
                "number of signature columns are directly comparable; the two "
                "combination rules at $k{=}6$ use the same $12$ columns as one rule at "
                "$k{=}12$. 1NN is leave-one-out; LDA and the random forest ($300$ trees) are $5$-fold stratified CV.",
                "tab:matched")


# ---- Table: cost ----------------------------------------------------------
def tab_cost():
    import pickle
    from qsig import directions
    pool = directions.by_angle(directions.pool(max_norm2=130))
    slot = pickle.load(open(os.path.join(HERE, "slotcols.pkl"), "rb"))
    full = round(directions.cost_of(pool, "rows+numba"), 6)
    rows = [("full pool, 64 directions", full, 1.0)]
    for k in (4, 6, 10, 12, 24):
        c = round(directions.cost_of([pool[i] for i in slot[k]], "rows+numba"), 6)
        rows.append(("$\\mathrm{slot\\_set}(%d)$" % k, c, full / c))
    b = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule",
         "direction set & modelled s/shape & speed-up \\\\", "\\midrule"]
    TIMES = "$\\times$"
    for n, c, sp in rows:
        spd = "---" if sp == 1.0 else ("%.1f" % sp) + TIMES
        b.append(n + " & " + ("%.6f" % c) + " & " + spd + " \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    return wrap("\n".join(b),
                "Modelled cost of the direction sets, row-prefix kernel with JIT "
                "compilation. Computing $\\varphi_D$ alongside $\\varphi$ adds "
                "$1.9$--$17.4\\%$, worst at the cheapest direction, since both come "
                "from the same four cone counts.",
                "tab:cost")



# ---- Table: datasets ------------------------------------------------------
def tab_datasets():
    rows = []
    meta = [("MPEG-7 CE Shape-1 Part B", "mpeg7", "latecki2000shape",
             "no overlap by construction"),
            ("Animal2000", "animal2000", "bai2009integrating",
             "8 shared class \\emph{names} with MPEG-7; closest cross-dataset "
             "pairs in the two closest classes inspected, clearly different "
             "shapes. Spot check, not exhaustive"),
            ("Swedish Leaves", "swedishleaves", "soderkvist2001computer",
             "disjoint domain")]
    for name, stem, key, prov in meta:
        rows_csv = list(csv.DictReader(open(os.path.join(HERE, FEAT[stem]))))
        n = len(rows_csv); c = len({r["cls"] for r in rows_csv})
        rows.append((name, key, n, c, prov))
    b = ["\\begin{tabular}{@{}llrrp{52mm}@{}}", "\\toprule",
         "dataset & source & shapes & classes & provenance check \\\\", "\\midrule"]
    for name, key, n, c, prov in rows:
        b.append(f"{name} & \\cite{{{key}}} & {n} & {c} & {prov} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    return wrap("\n".join(b),
                "The three benchmarks. Ground-truth masks for Animal2000 and "
                "Swedish Leaves are taken from the skeleton ground-truth "
                "collection \\cite{yang2024skeleton}. All shapes are rescaled to a "
                "long side of 128 pixels and padded square, identically for every "
                "descriptor and every baseline.",
                "tab:datasets",
                "Note: \\cite{bai2009integrating} does not itself use the name "
                "``Animal2000''; it reports 20 classes of 100 shapes. The name is "
                "the community's.")


# ---- Table: class-disjoint selection --------------------------------------
def tab_classdisjoint():
    """Recomputed from the feature CSVs via classdisjoint_general.py."""
    import numpy as np
    import classdisjoint_general as CD
    lines = []
    for name, feat, bands in CD.SPECS:
        first = True
        for bandname, bf, pre in bands:
            y, CL, P, D = CD.load(feat, bf, pre)
            classes = np.array(sorted(set(y)))
            nsel = max(2, len(classes) // 3)
            rng = np.random.default_rng(20260926)
            picks, gains = [], []
            for _ in range(CD.NREP):
                selc = set(rng.choice(classes, nsel, replace=False))
                m = np.array([c in selc for c in y])
                ch, ev, phi, _ = CD.run(y, CL, P, D, np.where(m)[0], np.where(~m)[0])
                picks.append(ch.split("@")[0]); gains.append(ev - phi)
            g = np.array(gains)
            from collections import Counter
            mode = Counter(picks).most_common(1)[0]
            lab = {"narrow": "narrow, $11$--$27^\\circ$",
                   "wide": "wide, $53$--$79^\\circ$",
                   "orthog": "orthogonal, $90^\\circ$"}[bandname]
            EOL = " \\\\"
            arm = mode[0].replace("_", "\\_")
            rng_s = "$[" + ("%+.2f" % g.min()) + ",\\," + ("%+.2f" % g.max()) + "]$"
            lines.append((name if first else "") + " & " + lab + " & "
                         + arm + " (" + str(mode[1]) + "/" + str(CD.NREP) + ") & $"
                         + ("%+.2f" % g.mean()) + "$ & " + rng_s + " & "
                         + str(int((g > 0).sum())) + "/" + str(CD.NREP) + EOL)
            first = False
        if name != CD.SPECS[-1][0]:
            lines.append("\\midrule")
    b = ["\\begin{tabular}{@{}llcrcc@{}}", "\\toprule",
         "dataset & aperture band & arm selected & mean & range & wins \\\\",
         "\\midrule"] + lines + ["\\bottomrule", "\\end{tabular}"]
    return wrap("\n".join(b),
                "Class-disjoint selection at matched column budget. The arm and "
                "the number of directions are chosen on classes that are never "
                "scored; the reported gain is the selected arm's held-out accuracy "
                "minus that of the phi-only arm the same procedure would have "
                "selected. 25 random class splits per cell.",
                "tab:classdisjoint",
                "The effect holds on MPEG-7, is neutral on Swedish Leaves and is "
                "reversed on Animal2000 --- which is what the degeneracy rates of "
                "Table~\\ref{tab:degeneracy} predict for MPEG-7 and Animal2000, "
                "and not what they predict for Swedish Leaves.")

if __name__ == "__main__":
    for fn, name in ((tab_degeneracy, "tab_degeneracy"), (tab_matched, "tab_matched"),
                     (tab_cost, "tab_cost"), (tab_datasets, "tab_datasets"),
                     (tab_classdisjoint, "tab_classdisjoint")):
        t = fn()
        open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8").write(t)
        print(f"wrote {name}.tex  ({len(t.splitlines())} lines)")
