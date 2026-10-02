"""The general (non-orthogonal) direction-pair set used for the phi_D test.

Why Farey neighbours. |det(r,s)| = 1 means (r,s) is a BASIS of Z^2, which is
both the cheapest case for the kernel and the natural "general pair" analogue
of an orthogonal frame.

Why the full orientation range. directions.pool() restricts to p>=1, q<=0 -- a
90-degree window. That is sufficient for ORTHOGONAL pairs, because {r, r_perp}
is determined by the angle mod 90, but it loses most general pairs: inside the
pool the usable-aperture Farey pairs all have bisectors in 112-176 deg, a
116-degree hole. Over the full range [0,180) the same construction gives 29
pairs at aperture >= 10 with a maximum bisector gap of 13.3 deg.

Aperture floor. Narrow apertures drive values toward the degenerate floor
(sliver saturation), so pairs below APERTURE_MIN are excluded. The axis pair
(1,0),(0,1) has aperture 90 and IS the orthogonal case, so it is excluded too --
the set must be genuinely non-orthogonal for the test to mean anything.
"""
from __future__ import annotations
import sys, os
from math import atan2, degrees, gcd

import paths  # noqa: F401  -- puts the repository root on sys.path

APERTURE_MIN = 10.0
APERTURE_MAX = 85.0
MAX_NORM2 = 130


def _angle(w):
    return degrees(atan2(w[1], w[0])) % 180.0


def geometry(r, s):
    d = abs(_angle(r) - _angle(s))
    ap = min(d, 180.0 - d)
    bis = (_angle(r) + _angle(s)) / 2 if d <= 90 else ((_angle(r) + _angle(s)) / 2 + 90) % 180
    return bis, ap


def all_directions(max_norm2=MAX_NORM2):
    """One primitive representative per line orientation in [0,180)."""
    out = []
    lim = int(max_norm2 ** 0.5) + 1
    for p in range(0, lim + 1):
        for q in range(-lim, lim + 1):
            if (p, q) == (0, 0) or p * p + q * q > max_norm2:
                continue
            if gcd(abs(p), abs(q)) != 1:
                continue
            if p < 0 or (p == 0 and q < 0):
                continue
            out.append((p, q))
    return sorted(set(out), key=_angle)


def farey_pairs():
    from qsig import pairs as _p
    V = all_directions()
    seen, out = set(), []
    for r in V:
        for s in V:
            if abs(_p.det(r, s)) != 1:
                continue
            k = tuple(sorted([r, s]))
            if k in seen:
                continue
            seen.add(k)
            bis, ap = geometry(r, s)
            if not (APERTURE_MIN <= ap <= APERTURE_MAX):
                continue
            out.append({"bisector": bis, "aperture": ap,
                        "maxcoord": max(max(map(abs, r)), max(map(abs, s))),
                        "r": r, "s": s})
    return sorted(out, key=lambda d: d["bisector"])


def spread(k):
    """k pairs spread as evenly as possible over bisector orientation.
    Ties broken by the largest coordinate magnitude, which is what drives the
    general-pair kernel cost (measured 2026-09-26; the row-cost model does NOT
    predict it)."""
    fp = farey_pairs()
    chosen = []
    for i in range(k):
        target = 180.0 * i / k
        cand = sorted(fp, key=lambda d: (min(abs(d["bisector"] - target),
                                             180 - abs(d["bisector"] - target)),
                                         d["maxcoord"]))
        for c in cand:
            if c not in chosen:
                chosen.append(c)
                break
    return sorted(chosen, key=lambda d: d["bisector"])


if __name__ == "__main__":
    fp = farey_pairs()
    print(f"{len(fp)} Farey pairs with aperture in "
          f"[{APERTURE_MIN}, {APERTURE_MAX}] over the full orientation range")
    for k in (6, 12):
        sel = spread(k)
        bs = [d["bisector"] for d in sel]
        gaps = [b - a for a, b in zip(bs, bs[1:])] + [180 - bs[-1] + bs[0]]
        print(f"\nspread({k}):  max bisector gap {max(gaps):.2f} deg")
        print(f"  {'bisector':>9s} {'apert':>7s} {'maxc':>5s} {'r':>9s} {'s':>9s}")
        for d in sel:
            print(f"  {d['bisector']:9.2f} {d['aperture']:7.2f} {d['maxcoord']:5d} "
                  f"{str(d['r']):>9s} {str(d['s']):>9s}")
