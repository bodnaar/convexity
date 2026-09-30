#!/usr/bin/env python3
"""Digital rotation does not preserve Q-convexity -- figure and counts.

    python scripts/fig_rotloss.py --out fig4_rotloss

Four convex synthetic shapes, whose Q-convexity is exactly 1 in every direction.
The rotation-free signature confirms this: over all 64 directions of the pool it
returns 1 exactly on every shape, integer arithmetic on an unresampled image.
The rotational descriptor does not: after digital rotation the re-binarised set
is no longer Q-convex, and the figure paints the background pixels that acquire
object points in all four of their quadrants and so contribute to the measure.

The shapes are drawn at low resolution on purpose, so that individual pixels are
legible without a magnified inset.

NOTE on the quadrant convention: the quadrants at P are CLOSED -- they include
the row and the column through P -- so a single boundary pixel detached by
resampling on the far side of that row is enough to occupy a quadrant.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qsig import directions as D, fast, rotational  # noqa: E402

N = 28                  # low resolution: pixels must be legible in print
ANG = 40.601295         # the angle of the lattice direction (7, 6)
OBJ, BG, ACC, GRID = "#D3D8DE", "#FFFFFF", "#B2182B", "#EDEFF2"


def build(n=N):
    """Four convex shapes of comparable area."""
    out = {}
    a = np.zeros((n, n), np.uint8)
    h, w = int(n * 0.70), int(n * 0.44)
    a[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = 1
    out["rectangle"] = a

    b = np.zeros((n, n), np.uint8)
    s = int(n * 0.80)
    o = (n - s) // 2
    for i in range(s):
        b[o + i, o:o + (s - i)] = 1
    out["triangle"] = b

    y, x = np.mgrid[0:n, 0:n]
    c, R = (n - 1) / 2, n * 0.36
    m = np.ones((n, n), bool)
    for k in range(8):
        t = 2 * math.pi * k / 8
        m &= ((x - c) * math.cos(t) + (y - c) * math.sin(t)) <= R * math.cos(math.pi / 8)
    out["octagon"] = m.astype(np.uint8)
    out["disc"] = (((y - c) ** 2 + (x - c) ** 2) <= R * R).astype(np.uint8)
    return out


def rotational_value(img, ang):
    """(rotated set, phi map, Q-convexity) of the rotational descriptor."""
    rot = rotational.rotate_binary(img, ang, expand=True, protocol="exact")
    d = fast.compute(rot, vec=(1, 0), method="rows")
    return rot > 0, d["_phi"], 1.0 - d["q1"]


def survey(pool):
    """Per-shape counts over the whole pool -- the numbers quoted in the caption."""
    angles = sorted({round(math.degrees(math.atan2(d.q, d.p)), 6) for d in pool})
    rows = []
    for name, img in build().items():
        qcx = [1.0 - fast.compute(img, vec=(d.p, -d.q), method="rows")["q1"] for d in pool]
        worst, nz = 1.0, 0
        for a in angles:
            v = rotational_value(img, a)[2]
            if v < 1.0:
                nz += 1
            worst = min(worst, v)
        rows.append((name, min(qcx), nz, len(angles), worst))
    return rows


def _crop(A, phi, half):
    """Square window of a fixed half-size, centred on the object; padded if needed."""
    A = np.pad(A, half, constant_values=False)
    phi = np.pad(phi, half)
    ys, xs = np.nonzero(A)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    sy = slice(cy - half, cy + half + 1)
    sx = slice(cx - half, cx + half + 1)
    return A[sy, sx], phi[sy, sx]


def figure(out):
    shapes = build()
    panels = {k: rotational_value(v, ANG) for k, v in shapes.items()}
    half = 2 + max(max(np.ptp(np.nonzero(A)[0]), np.ptp(np.nonzero(A)[1])) // 2
                   for A, _, _ in panels.values())
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.2, 1.62))
    for ax, (name, (A, phi, qcx)) in zip(axes, panels.items()):
        A, phi = _crop(A, phi, half)
        rgb = np.ones(A.shape + (3,))
        rgb[A] = matplotlib.colors.to_rgb(OBJ)
        rgb[phi > 0] = matplotlib.colors.to_rgb(ACC)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks(np.arange(-.5, A.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, A.shape[0], 1), minor=True)
        ax.grid(which="minor", color=GRID, lw=0.25)
        ax.tick_params(which="both", length=0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#9AA3AE")
        ax.set_title(f"{name}\n{qcx:.4f}", fontsize=7.4, pad=3, color="#1F2933")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.80, bottom=0.02, wspace=0.10)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig4_rotloss")
    ap.add_argument("--max-norm2", type=int, default=130)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    pool = D.pool(args.max_norm2)
    print(f"resolution {N} px, figure angle {ANG}deg, pool |P_{args.max_norm2}| = {len(pool)}\n")
    print(f"{'shape':11s} {'S: min Q-cx':>12s} {'R: below 1':>12s} {'R: min Q-cx':>12s}")
    for name, s_min, nz, n, worst in survey(pool):
        print(f"{name:11s} {s_min:>12.6f} {nz:>7d}/{n:<4d} {worst:>12.4f}")
    if not args.no_figure:
        figure(args.out)
        print(f"\nfigure written to {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
