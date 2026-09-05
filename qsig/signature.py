"""Q-concavity signatures and the orbit distance d_C.

d_C is defined in IWCIA 2025 sec 5.2:

    rev(u)        = (u_k, ..., u_1)
    shift(u, i)   = (u_{k-i+1}, ..., u_k, u_1, ..., u_{k-i})
    C_u           = {u} u rev(u) u {shift(u,i)} u {rev(shift(u,i))},  i=1..k-1
    d(u, v)       = min { ||u' - v||_2 : u' in C_u }
    d_C(u, v)     = d(u - mean(u), v - mean(v))

WARNING -- the equiangularity assumption
----------------------------------------
The cyclic-shift orbit models rotation only when the direction set is
EQUIANGULAR: rotating the shape by one angular step then permutes the signature
cyclically. IWCIA 2025's `S_n` are exactly equiangular and `S_int` is
approximately so (nearest lattice direction to each integer angle), which is
what justifies d_C there.

A "cheapest-k" set (`qsig.directions.cheapest_k`) is NOT equiangular -- its gaps
are irregular by construction -- so a cyclic shift of its signature does not
correspond to any rotation, and d_C has no justification on it. Comparing a
cheapest-k set against IWCIA's published Table 2 numbers under d_C would not be
an honest comparison.

Use `qsig.directions.slot_set` instead: it keeps the equiangular schedule and
chooses the cheapest lattice representative of each slot, so d_C remains as
well-founded as it is for `S_int`, and the comparison is like for like. Both
constructions are implemented so the difference can be measured rather than
asserted, but the paper's headline curve should use `slot_set`.
"""

from __future__ import annotations

import numpy as np


def orbit(u: np.ndarray) -> np.ndarray:
    """All 2k elements of C_u, as a (2k, k) array. Rows are the k cyclic shifts
    of u followed by the k cyclic shifts of rev(u)."""
    u = np.asarray(u, dtype=float).ravel()
    k = u.size
    idx = (np.arange(k)[None, :] - np.arange(k)[:, None]) % k
    shifts = u[idx]                      # (k, k); row 0 is u itself
    r = u[::-1]
    rshifts = r[idx]
    return np.vstack([shifts, rshifts])


def centred_orbit(u: np.ndarray) -> np.ndarray:
    """Mean-centred orbit, ready for repeated distance queries."""
    u = np.asarray(u, dtype=float).ravel()
    return orbit(u - u.mean())


def d_C(u: np.ndarray, v: np.ndarray) -> float:
    """Orbit distance between two signatures."""
    v = np.asarray(v, dtype=float).ravel()
    return float(np.linalg.norm(centred_orbit(u) - (v - v.mean()), axis=1).min())


def pairwise_d_C(sigs: np.ndarray, block: int = 64) -> np.ndarray:
    """Full (N, N) d_C matrix for a stack of signatures, shape (N, k).

    Not symmetric in general by construction (the min is taken over the orbit of
    the first argument only), but it *is* symmetric here because C_u is closed
    under shift and reversal, so d(u,v) = d(v,u). We compute the upper triangle
    and mirror it; `test_dc.py` asserts the symmetry on random input.
    """
    sigs = np.asarray(sigs, dtype=float)
    n, k = sigs.shape
    cent = sigs - sigs.mean(axis=1, keepdims=True)
    out = np.zeros((n, n), dtype=float)
    for start in range(0, n, block):
        stop = min(start + block, n)
        orbits = np.stack([orbit(cent[i]) for i in range(start, stop)])   # (b, 2k, k)
        # (b, 2k, 1, k) - (1, 1, n, k) -> (b, 2k, n)
        d = np.linalg.norm(orbits[:, :, None, :] - cent[None, None, :, :], axis=-1)
        out[start:stop, :] = d.min(axis=1)
    np.fill_diagonal(out, 0.0)
    return np.minimum(out, out.T)


def euclidean(sigs: np.ndarray) -> np.ndarray:
    """Plain L2 between signatures, no orbit. Reported alongside d_C as the
    'no rotation handling' control."""
    sigs = np.asarray(sigs, dtype=float)
    d = sigs[:, None, :] - sigs[None, :, :]
    return np.linalg.norm(d, axis=-1)
