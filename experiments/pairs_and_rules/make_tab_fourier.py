"""LaTeX table bodies for the Fourier-descriptor experiments, from
fourier_results.json, fourier_variants.json and classdisjoint_fourier.json.
Writes tab/tab_fourier.tex, tab/tab_fourier_variants.tex and
tab/tab_classdisjoint_fd.tex: bare tabular environments plus their footnote
line. Captions are left to the manuscript.
"""
from __future__ import annotations
import json, os
import paths
OUT = paths.TAB_DIR
os.makedirs(OUT, exist_ok=True)
H = paths.HERE
DS = [("MPEG-7", "MPEG-7"), ("Animal2000", "Animal2000"), ("SwedishLeaves", "Swedish Leaves")]
STAR = "$^{*}$"
NOTE = ("\\vspace{2pt}\n\\par\\noindent\\begin{minipage}{\\linewidth}\\footnotesize "
        "%s\\end{minipage}")


def iv(t, star_if_excl=True):
    m, lo, hi = t
    s = STAR if star_if_excl and not (lo <= 0 <= hi) else ""
    return f"${m:+.2f}${s} $[{lo:+.2f},{hi:+.2f}]$"


def tab_fourier():
    R = json.load(open(os.path.join(H, "fourier_results.json")))
    rows = [("FD", "FD only", 0, None),
            ("cl+FD", "classical $+$ FD", 0, None),
            ("cl+FD+phi12", "$+\\,\\varphi$ ($k{=}12$)", 12, "cl+FD+phi12|cl+FD"),
            ("cl+FD+both6", "$+\\,\\varphi,\\varphi_D$ ($k{=}6$)", 12, "cl+FD+both6|cl+FD")]
    b = ["\\begin{tabular}{@{}llrrcrr@{}}", "\\toprule",
         "dataset & feature set & cols & 1NN & $\\Delta$1NN $[95\\%]$ & LDA & RF \\\\", "\\midrule"]
    for i, (key, show) in enumerate(DS):
        r = R[key]["M=12"]
        for j, (arm, lbl, cols, test) in enumerate(rows):
            a = r["acc"][arm]
            d = iv(r["tests"][test]) if test else ""
            b.append(f"{show if j == 0 else ''} & {lbl} & {cols} & {a['1nn']:.2f} & {d} & {a['lda']:.2f} & {a['rf']:.2f} \\\\")
        if i < len(DS) - 1:
            b.append("\\midrule")
    b += ["\\bottomrule", "\\end{tabular}",
          NOTE % ("FD: $12$ centroid-distance Fourier descriptors. $\\Delta$1NN: gain over classical $+$ FD, "
                  "in points, with the $95\\%$ paired bootstrap interval ($4000$ resamples over shapes); "
                  "$^{*}$ the interval excludes zero.")]
    return "\n".join(b) + "\n"


def tab_variants():
    R = json.load(open(os.path.join(H, "fourier_variants.json")))
    show_v = {"area/resampled": "area centroid, resampled",
              "boundary/resampled": "boundary centroid, resampled",
              "boundary/raw": "boundary centroid, raw contour"}
    b = ["\\begin{tabular}{@{}llrcrr@{}}", "\\toprule",
         "dataset & FD variant & cl$+$FD & $\\Delta$1NN $[95\\%]$ & $\\Delta$LDA & $\\Delta$RF \\\\", "\\midrule"]
    for i, (key, show) in enumerate(DS):
        for j, (v, lbl) in enumerate(show_v.items()):
            r = R[key][v]
            b.append(f"{show if j == 0 else ''} & {lbl} & {r['cl+FD_1nn']:.2f} & {iv(r['both6'])} & "
                     f"${r['lda_gain_both6']:+.2f}$ & ${r['rf_gain_both6']:+.2f}$ \\\\")
        if i < len(DS) - 1:
            b.append("\\midrule")
    b += ["\\bottomrule", "\\end{tabular}",
          NOTE % ("Gain of classical $+$ FD $+\\,\\varphi,\\varphi_D$ ($k{=}6$) over classical $+$ FD; "
                  "$^{*}$ the $95\\%$ paired bootstrap interval excludes zero.")]
    return "\n".join(b) + "\n"


def tab_classdisjoint_fd():
    R = json.load(open(os.path.join(H, "classdisjoint_fourier.json")))
    band = {"narrow": "narrow", "wide": "wide", "orthog": "orthogonal"}
    b = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llcrcccc@{}}", "\\toprule",
         " & & \\multicolumn{4}{c}{classical baseline} & \\multicolumn{2}{c}{classical $+$ FD baseline} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}",
         "dataset & band & selected & mean & range & wins & mean & wins \\\\", "\\midrule"]
    for i, (key, show) in enumerate(DS):
        for j, bd in enumerate(("narrow", "wide", "orthog")):
            c, f = R[f"{key}|{bd}"]["classical"], R[f"{key}|{bd}"]["classical+FD"]
            arm = {"phi": "$\\varphi$", "phi_D": "$\\varphi_D$", "both": "both"}[c["mode"][0]]
            b.append(f"{show if j == 0 else ''} & {band[bd]} & {arm} ({c['mode'][1]}/25) & ${c['mean']:+.2f}$ & "
                     f"$[{c['lo']:+.2f},\\,{c['hi']:+.2f}]$ & {c['wins']}/25 & ${f['mean']:+.2f}$ & {f['wins']}/25 \\\\")
        if i < len(DS) - 1:
            b.append("\\midrule")
    b += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(b) + "\n"


if __name__ == "__main__":
    for fn, name in ((tab_fourier, "tab_fourier"), (tab_variants, "tab_fourier_variants"),
                     (tab_classdisjoint_fd, "tab_classdisjoint_fd")):
        open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8").write(fn())
        print("wrote", name)
