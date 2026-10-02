"""Table: phi_D as a drop-in replacement for phi (phi_D - phi, same direction set).

1NN leave-one-out, 4000-resample paired bootstrap over shapes, classical 4d + 12
(slot_set(12)) or all 64 signature columns. Same protocol as disjmeasure.py.
Writes tab_replacement.tex into paths.TAB_DIR and prints the numbers.
"""
import os, sys, csv, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import maketables as mt

SLOT = pickle.load(open(os.path.join(HERE, "slotcols.pkl"), "rb"))
DISJ = {"mpeg7": "mpeg7_disjunctive.csv", "animal2000": "animal2000_disjunctive.csv",
        "swedishleaves": "swedishleaves_disjunctive.csv"}

def load(stem):
    rows = list(csv.DictReader(open(os.path.join(HERE, mt.FEAT[stem]))))
    dr = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, DISJ[stem])))}
    rows = [r for r in rows if r["shape_id"] in dr]
    y = np.array([r["cls"] for r in rows])
    CL = np.array([[float(r[k]) for k in ("area_ratio", "circularity", "hu1", "hu2")] for r in rows])
    S = np.array([[float(r[f"sig{i}"]) for i in range(64)] for r in rows])
    SD = np.array([[float(dr[r["shape_id"]][f"sigd{i}"]) for i in range(64)] for r in rows])
    return y, CL, S, SD

def cell(m, lo, hi):
    sig = not (lo < 0 < hi)
    s = f"${m:+.2f}$" + ("$^{*}$" if sig else "")
    return s + f" $[{lo:+.2f},{hi:+.2f}]$", sig

res = {}
for name, stem in mt.DS:
    y, CL, S, SD = load(stem)
    idx = SLOT[12]
    for lbl, a, b in (("slot12", np.c_[CL, S[:, idx]], np.c_[CL, SD[:, idx]]),
                      ("all64", np.c_[CL, S], np.c_[CL, SD])):
        ca, cb = mt.correct(a, y), mt.correct(b, y)
        m, lo, hi = mt.ci(ca, cb)
        res[(name, lbl)] = (m, lo, hi, ca.mean()*100, cb.mean()*100, len(y))
        print(f"{name:14s} {lbl:7s} n={len(y)} phi={ca.mean()*100:6.2f} phiD={cb.mean()*100:6.2f} "
              f"diff {m:+.2f} [{lo:+.2f},{hi:+.2f}]")

b = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule",
     "dataset & $\\texttt{slot\\_set}(12)$ & all $64$ directions \\\\", "\\midrule"]
for name, _ in mt.DS:
    c1, _ = cell(*res[(name, "slot12")][:3]); c2, _ = cell(*res[(name, "all64")][:3])
    b.append(f"{name} & {c1} & {c2} \\\\")
b += ["\\bottomrule", "\\end{tabular}"]
t = mt.wrap("\n".join(b),
            "The disjunctive rule as a replacement for the conjunctive one: 1NN leave-one-out "
            "accuracy of (classical\\,$+\\,\\varphi_D$) minus (classical\\,$+\\,\\varphi$), in "
            "percentage points, with the $95\\%$ paired-bootstrap interval in brackets, on the "
            "same direction set. $\\texttt{slot\\_set}(12)$ uses $12$ signature columns per arm, "
            "the full pool $64$.",
            "tab:replacement",
            "$^{*}$ the $95\\%$ paired bootstrap interval ($4000$ resamples over shapes) of the "
            "difference excludes zero.")
open(os.path.join(mt.OUT, "tab_replacement.tex"), "w", encoding="utf-8").write(t)
print("wrote tab_replacement.tex")
