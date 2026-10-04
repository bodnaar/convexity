"""Robustness of the Fourier-descriptor result to the FD variant (M = 12).

  area/resampled     -- primary (extract_fourier.py): area centroid, 128 points
                        equally spaced in arc length
  boundary/resampled -- centroid = mean of the resampled boundary points
  boundary/raw       -- as written in Zhang & Lu (2003): the traced boundary
                        points as they are, centroid = their mean

For each: 1NN gain of classical+FD+phi(12), and of classical+FD+phi,phi_D(6),
over classical+FD, with paired bootstrap intervals; LDA and RF for the two-rule
gain. Writes fourier_variants.json; optional dataset-name arguments as in
fourier_test.py.
"""
from __future__ import annotations
import json, os, sys
import paths  # noqa: F401
import numpy as np
import cv2
import extract_fourier as EF
import fourier_test as FT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

M = 12


def fd(bin_img, variant):
    if variant == "area/resampled":
        return EF.fourier_descriptor(bin_img)[:M]
    img8 = (bin_img * 255).astype(np.uint8)
    cs, _ = cv2.findContours(img8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    if variant == "boundary/resampled":
        c2 = np.vstack([c, c[:1]])
        s = np.r_[0.0, np.cumsum(np.sqrt((np.diff(c2, axis=0) ** 2).sum(1)))]
        t = np.linspace(0.0, s[-1], EF.N, endpoint=False)
        x, y = np.interp(t, s, c2[:, 0]), np.interp(t, s, c2[:, 1])
    else:
        x, y = c[:, 0], c[:, 1]
    F = np.abs(np.fft.fft(np.hypot(x - x.mean(), y - y.mean())))
    return F[1:M + 1] / F[0]


VARIANTS = ["area/resampled", "boundary/resampled", "boundary/raw"]


def main():
    imgs = {}
    for ds, sid, cls, img in EF.shapes():
        imgs.setdefault(ds, {})[sid] = img
    path = os.path.join(paths.HERE, "fourier_variants.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    only = set(sys.argv[1:])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, stem, fn, dfn in FT.DATASETS:
        if only and name not in only:
            continue
        y, CL, S, SD, _ = FT.load(stem, fn, dfn)
        ids = FT.load_ids(stem, fn, dfn)
        p12, b6 = S[:, FT.SLOT[12]], np.c_[S[:, FT.SLOT[6]], SD[:, FT.SLOT[6]]]
        res = {}
        for v in VARIANTS:
            F = np.array([fd(imgs[stem][i], v) for i in ids])
            A, B, C = np.c_[CL, F], np.c_[CL, F, p12], np.c_[CL, F, b6]
            a, b, c = FT.correct(A, y), FT.correct(B, y), FT.correct(C, y)
            lda = lambda X: cross_val_score(make_pipeline(StandardScaler(), LDA()), X, y, cv=cv).mean() * 100
            rf = lambda X: cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                                           X, y, cv=cv).mean() * 100
            res[v] = {"cl+FD_1nn": round(a.mean() * 100, 2), "phi12": FT.ci(a, b), "both6": FT.ci(a, c),
                      "both6_vs_phi12": FT.ci(b, c),
                      "lda_gain_both6": round(lda(C) - lda(A), 2), "rf_gain_both6": round(rf(C) - rf(A), 2)}
            print(name, v, res[v], flush=True)
        out[name] = res
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
