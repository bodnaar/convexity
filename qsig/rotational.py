"""The ROTATIONAL Q-concavity signature R_F^D -- handoff sec 3.3, sec 7.4 item 12.

IWCIA 2025 defines two descriptors, and until now only one of them existed in
this codebase:

  S_F^D  rotation-free.  Evaluate a genuinely slanted lattice direction pair
         (d, d_perp). Accurate, and the cost grows with the direction --
         Theta(mn|r|^2) for the point kernel, Theta(mn(p+q)) for the rows one.

  R_F^D  rotational.     ROTATE THE IMAGE by theta and evaluate at the plain
         (1,0),(0,1) pair. The cost is the same for every angle, because the
         cheapest possible direction pair is always used. The price is that
         rotating a digital image is not a bijection on Z^2: it resamples, and
         information is lost.

R is the cheap competitor the paper's accuracy-versus-cost curve has to beat
(handoff sec 3.3). Their Table 2 gives R's accuracies but its timings are 2016
stack seconds, so R must be re-measured in the same kernel as everything else
before any comparison is meaningful. That is what this module is for.

Rotating the canvas -- a decision worth stating in the paper
------------------------------------------------------------
IWCIA 2025 sec 5.3 says only that they "calculated the rotation matrix and then
performed the affine transformation with linear interpolation". It does not say
what happens to the corners.

MPEG-7 silhouettes touch the frame edge, so rotating inside a fixed 128x128
canvas CLIPS the shape and destroys object pixels -- which changes the shape
being described, not merely its orientation. `expand=True` (the default here)
grows the canvas to the rotated bounding box so nothing is lost.

The consequence is that R's cost is then mildly angle-dependent, since the array
grows by up to sqrt(2) in each dimension at 45 degrees. Their reported flat
2.54 s per component implies they did NOT expand, i.e. they clipped. Both are
implemented; the default is the one that preserves the shape, and `expand` is
recorded with the results so a run can be attributed.

Re-binarisation
---------------
Linear interpolation of a binary image produces grey values, which must be
thresholded back. That is the same hazard as the pyramid re-binarisation of
handoff sec 6 test 1, and it is isolated in one function here so the choice can
be varied without touching anything else.
"""

from __future__ import annotations

import time

import numpy as np

from .dataset import BACKGROUND, OBJECT
from .directions import Direction

try:  # pragma: no cover - cv2 is in requirements.txt but keep the import soft
    import cv2

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


def rotate_binary(img: np.ndarray, angle_deg: float, expand: bool = True,
                  threshold: int = 127) -> np.ndarray:
    """Rotate a {0,1} image by `angle_deg` and re-binarise.

    Rotation is counter-clockwise about the image centre, matching the sense in
    which `Direction.angle` increases, so that rotating by theta_d and
    evaluating at (1,0) corresponds to evaluating direction d.
    """
    if not HAVE_CV2:  # pragma: no cover
        raise RuntimeError("opencv is required for the rotational signature")
    a = (np.asarray(img) == OBJECT).astype(np.uint8) * 255
    h, w = a.shape
    if angle_deg % 360 == 0:
        return (a > threshold).astype(np.uint8)
    m = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), angle_deg, 1.0)
    if expand:
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
    else:
        nw, nh = w, h
    out = cv2.warpAffine(a, m, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (out > threshold).astype(np.uint8)


def rotational_value(img: np.ndarray, direction: Direction, impl: str = "rows",
                     expand: bool = True) -> tuple[float, float]:
    """One component of R_F^D: rotate by the direction's angle, evaluate at (1,0).

    Returns (value, elapsed_seconds). The timing INCLUDES the rotation, because
    that is a real cost of this descriptor and excluding it would flatter R.
    """
    from .descriptor import q_concavity          # local: avoids a circular import

    axis = Direction(1, 0)
    t0 = time.perf_counter()
    rot = rotate_binary(img, direction.angle, expand=expand)
    if rot.sum() == 0 or rot.sum() == rot.size:
        raise ValueError(f"rotation by {direction.angle:.1f} deg emptied the image")
    value, _ = q_concavity(rot, axis, impl)
    return float(value), time.perf_counter() - t0


def rotation_loss(img: np.ndarray, angles=(15.0, 30.0, 45.0), expand: bool = True) -> dict:
    """How much of the shape a rotation destroys -- the honest caveat, quantified.

    Rotating by theta and back should be the identity but is not: resampling and
    re-binarisation move pixels. Reports the symmetric difference as a fraction
    of the object, per angle. Useful for handoff sec 3.5 (measured rotation
    robustness) and for stating R's limitation with a number rather than a
    hand-wave.
    """
    base = (np.asarray(img) == OBJECT).astype(np.uint8)
    out = {}
    for a in angles:
        there = rotate_binary(base, a, expand=expand)
        back = rotate_binary(there, -a, expand=expand)
        h = min(base.shape[0], back.shape[0])
        w = min(base.shape[1], back.shape[1])
        b0 = base[:h, :w].astype(bool)
        b1 = back[:h, :w].astype(bool)
        out[a] = float((b0 ^ b1).sum() / max(b0.sum(), 1))
    return out
