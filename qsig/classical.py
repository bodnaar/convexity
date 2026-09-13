"""Classical shape descriptors -- the baseline that reframed the paper.

Specified in journal_paper/paper_a_plan_2026-09-13.md sec 3: convex-hull area
ratio, circularity, Hu moment invariants 1-7. Computed on the same
preprocessed binary image (`qsig.dataset.Shape.img`) as the Q-concavity
signature, so both feature families see identical shapes.

OPEN QUESTION, flagged not resolved: the original `~/work/classical.py` /
`mu.py` scripts that produced classical_baseline_2026-09-13.md's numbers are
outside this repository and were not available when this module was written.
This is a from-the-spec reimplementation. Recomputing full MPEG-7 through it
reproduces the two signature-only rows of that table EXACTLY (mu alone
20.64%, all-64-raw 51.43%) and the classical-descriptor rows to within ~1.3
points (73.07 vs 73.71 for the 4-dim classical baseline) -- close enough to
trust, not identical. The Hu log-transform convention below is the leading
suspect for the gap; it is a free design choice the original scripts may have
made differently.
"""

from __future__ import annotations

import cv2
import numpy as np

from .dataset import OBJECT

HU_DIMS = 7


def classical_features(img: np.ndarray) -> np.ndarray:
    """Return [area_ratio, circularity, hu1, ..., hu7] for one binary image.

    - area_ratio = contour area / convex-hull area (Q-convexity's classical
      analogue; 1.0 for an already-convex shape).
    - circularity = 4*pi*area / perimeter^2 (1.0 for a perfect disc).
    - Hu 1-7: cv2.HuMoments, log-transformed as
      `-sign(h) * log10(|h| + 1e-30)` to compress their natural range (raw Hu
      values span many orders of magnitude). This transform is an assumption
      -- see the module docstring.
    """
    img8 = (np.asarray(img) == OBJECT).astype(np.uint8) * 255
    contours, _ = cv2.findContours(img8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no contour found in classical_features")
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    if area <= 0:
        area = float((img8 > 0).sum())
    hull_area = cv2.contourArea(cv2.convexHull(c))
    area_ratio = area / hull_area if hull_area > 0 else np.nan

    perim = cv2.arcLength(c, True)
    circularity = (4.0 * np.pi * area / (perim ** 2)) if perim > 0 else np.nan

    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    return np.concatenate([[area_ratio, circularity], hu_log]).astype(float)


FEATURE_NAMES = ["area_ratio", "circularity"] + [f"hu{i}" for i in range(1, HU_DIMS + 1)]
