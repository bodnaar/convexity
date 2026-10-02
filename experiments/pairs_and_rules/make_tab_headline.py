"""Table: two rules over 6 directions vs one rule over all 64 (the cost/accuracy headline).

classical + phi at all 64 pool directions (64 columns) against classical + [phi, phi_D]
at slot_set(6) (12 columns). 1NN LOO with 4000-resample paired bootstrap; LDA 5-fold
stratified CV with in-pipeline scaling. Same protocol as disjmeasure.py / disjhead.py.
Writes tab_headline.tex into paths.TAB_DIR.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import maketables as mt
import make_tab_replacement as _r   # reuses load() and SLOT; also rewrites tab_replacement.tex
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def lda(X, y):
    return cross_val_score(make_pipeline(StandardScaler(), LDA()), np.asarray(X, float), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() * 100

rows = []
for name, stem in mt.DS:
    y, CL, S, SD = _r.load(stem)
    idx = _r.SLOT[6]
    A = np.c_[CL, S]
    B = np.c_[CL, S[:, idx], SD[:, idx]]
    ca, cb = mt.correct(A, y), mt.correct(B, y)
    m, lo, hi = mt.ci(ca, cb)
    la, lb = lda(A, y), lda(B, y)
    sig = not (lo < 0 < hi)
    print(f"{name:14s} 1NN {ca.mean()*100:.2f} -> {cb.mean()*100:.2f}  {m:+.2f} [{lo:+.2f},{hi:+.2f}]{' *' if sig else ''}   LDA {la:.2f} -> {lb:.2f}")
    rows.append(f"{name} & {ca.mean()*100:.2f} & {cb.mean()*100:.2f} & ${m:+.2f}$" + ("$^{*}$" if sig else "")
                + f" $[{lo:+.2f},{hi:+.2f}]$ & {la:.2f} & {lb:.2f} \\\\")

b = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{1NN} & \\multicolumn{2}{c}{LDA} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
     "dataset & $\\varphi$, all $64$ & $\\varphi,\\varphi_D$, $k{=}6$ & difference & $\\varphi$, all $64$ & $\\varphi,\\varphi_D$, $k{=}6$ \\\\",
     "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
t = mt.wrap("\n".join(b),
            "Two rules over $6$ directions against one rule over the full $64$-direction pool, "
            "accuracy in \\%. The pool arm uses $64$ signature columns and $0.504$\\,s per shape; "
            "the two-rule arm uses $12$ columns and $0.0201$\\,s (Table~\\ref{tab:cost}). "
            "1NN is leave-one-out, with the difference in points and its $95\\%$ paired-bootstrap "
            "interval; LDA is $5$-fold stratified CV.",
            "tab:headline",
            "$^{*}$ the $95\\%$ paired bootstrap interval ($4000$ resamples over shapes) of the "
            "difference excludes zero.")
open(os.path.join(mt.OUT, "tab_headline.tex"), "w", encoding="utf-8").write(t)
print("wrote tab_headline.tex")
