"""Lattice directions: enumeration, cost model, and selection constructions.

Conventions
-----------
A direction is a primitive lattice vector (p, q) with p >= 1, q >= 0,
gcd(p, q) = 1, whose angle is

    theta = atan2(q, p)  in  [0, 90) degrees.

`convexity.Convexity.compute` takes its vector in the *lower-right Cartesian
quadrant*, i.e. with the second component non-positive. So the vector handed to
the existing code is

    vec = (p, -q)

and NOT (p, q). Passing (p, q) with q > 0 raises ValueError inside `rotsat`
(empty `lower_triangle_pts`), and (0, +-1) raises ZeroDivisionError. Both were
verified against the repository code on 2026-09-05. `Direction.vec` is the only
thing that should ever be passed to `Convexity`.

`compute(vec=r)` evaluates the Q-concavity descriptor for the *pair* (r, r_perp)
-- the perpendicular is implicit in `rotsat`'s corner construction -- so one
Direction is one component of the rotation-free signature of [IWCIA 2025, Def 3].

Cost model
----------
With s = r_perp, the mask parallelogram spanned by r and s contains
|det(r, s)| - 1 = p^2 + q^2 - 1 interior lattice points (Pick's theorem), and
`rotsat` iterates exactly those points per pixel. Hence

    T(r)  ~  A + B * (p^2 + q^2)                     [handoff sec. 3.1]

which fits IWCIA 2025 Table 1 with R^2 = 0.998 (MAPE 4.0%), against R^2 = 0.903
(MAPE 25%) for the published O(mn (r1+s1)(|r2|+|s2|)) = O(mn (p+q)^2) bound.

`COST_IWCIA` holds the coefficients fitted to their published table. Refit
`A`/`B` on the target machine before using cost numbers in the paper -- see
`scripts/fit_cost_law.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, gcd
from typing import Iterable, Sequence

# Fitted to IWCIA 2025 Table 1 (MPEG-7, 128 px long side, Xeon E5-2670 v2).
# seconds = A + B * norm2
COST_IWCIA = (1.419, 0.5724)

# Measured cost models, nirgnode200, 2026-09-06: 24 shapes, subset=all, 128 px,
# performance governor, turbo on, pinned, quiet machine, Python 3.11.9 /
# numpy 1.26.4. Each entry is (predictor, intercept, slope) in seconds.
#
# USE THESE, NOT THE `seconds` COLUMN OF A POOL RUN. A pool run has 20 workers
# contending for memory bandwidth and L3, which inflates per-job times by ~3x
# and unevenly. Handoff sec 7.2: bulk runs optimise throughput, published
# timings come from pinned single-process measurement runs. Pricing a direction
# set from the fitted law is the only defensible number.
#
# The predictor differs by kernel because the cost model is a property of the
# ALGORITHM (handoff sec 3.1.2): the point kernel is Theta(mn|r|^2), the rows
# kernel Theta(mn(p+q)).
COST_MODELS = {
    "published":   ("norm2", 1.41900, 0.572400),
    "reference":   ("norm2", 1.71876, 0.074227),
    "fast+numba":  ("norm2", 0.00139, 0.000251),
    "rows+numba":  ("pq",    0.00150, 0.000655),
}


def cost_of(dirs, model: str = "rows+numba") -> float:
    """Modelled seconds to compute a whole signature with this direction set."""
    if model not in COST_MODELS:
        raise ValueError(f"unknown cost model {model!r}; have {sorted(COST_MODELS)}")
    kind, a, b = COST_MODELS[model]
    return sum(a + b * (d.norm2 if kind == "norm2" else d.p + d.q) for d in dirs)


def _cost1(d, model: str) -> float:
    kind, a, b = COST_MODELS[model]
    return a + b * (d.norm2 if kind == "norm2" else d.p + d.q)


@dataclass(frozen=True, order=False)
class Direction:
    p: int
    q: int

    def __post_init__(self):
        if self.p < 1 or self.q < 0:
            raise ValueError(f"direction must have p >= 1, q >= 0; got {(self.p, self.q)}")
        if gcd(self.p, self.q) != 1:
            raise ValueError(f"direction must be primitive; got {(self.p, self.q)}")

    @property
    def norm2(self) -> int:
        """p^2 + q^2 = |det(r, r_perp)|. The cost variable."""
        return self.p * self.p + self.q * self.q

    @property
    def angle(self) -> float:
        """Direction angle in degrees, in [0, 90)."""
        return degrees(atan2(self.q, self.p))

    @property
    def vec(self) -> tuple[int, int]:
        """The vector to hand to `Convexity.compute` (lower-right quadrant)."""
        return (self.p, -self.q)

    def cost(self, coeffs=COST_IWCIA) -> float:
        a, b = coeffs
        return a + b * self.norm2

    def __repr__(self) -> str:
        return f"Dir({self.p},{self.q} @{self.angle:.1f}deg n2={self.norm2})"


def pool(max_norm2: int | None = None, max_component: int | None = None) -> list[Direction]:
    """All primitive directions in [0, 90) degrees, cheapest first.

    Exactly one of `max_norm2` (a *cost* budget -- a disc) or `max_component`
    (a *box*, i.e. the Farey set F_Q that `deg2vec` produces) must be given.
    The disc is the right object under the cost model; the box is provided only
    to reproduce IWCIA 2025's `S_int`.

    Ties in cost are broken by angle, so the ordering is deterministic.
    """
    if (max_norm2 is None) == (max_component is None):
        raise ValueError("give exactly one of max_norm2 or max_component")
    out: list[Direction] = []
    if max_norm2 is not None:
        lim = int(max_norm2**0.5) + 1
        for p in range(1, lim + 1):
            for q in range(0, lim + 1):
                if gcd(p, q) == 1 and p * p + q * q <= max_norm2:
                    out.append(Direction(p, q))
    else:
        for p in range(1, max_component + 1):
            for q in range(0, max_component + 1):
                if gcd(p, q) == 1:
                    out.append(Direction(p, q))
    out.sort(key=lambda d: (d.norm2, d.angle))
    return out


def by_angle(dirs: Iterable[Direction]) -> list[Direction]:
    """Signature order: increasing slope, per [IWCIA 2025, Def 3]."""
    return sorted(dirs, key=lambda d: d.angle)


def max_angular_gap(dirs: Sequence[Direction]) -> float:
    """Largest gap in degrees, treating [0, 90) as cyclic (Q-concavity has
    period 90 degrees under rotation)."""
    a = sorted(d.angle for d in dirs)
    if len(a) == 1:
        return 90.0
    gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    gaps.append(90.0 - a[-1] + a[0])
    return max(gaps)


def total_cost(dirs: Iterable[Direction], coeffs=COST_IWCIA) -> float:
    return sum(d.cost(coeffs) for d in dirs)


# --------------------------------------------------------------------------
# Selection constructions
# --------------------------------------------------------------------------

def cheapest_k(k: int, candidates: Sequence[Direction] | None = None) -> list[Direction]:
    """The k cheapest primitive directions, in signature order.

    NOTE: this set is NOT equiangular, which matters for d_C -- see
    `qsig.signature.d_C` and the warning in its docstring. Use `slot_set` for
    anything compared against IWCIA 2025's published accuracies.
    """
    cand = list(candidates) if candidates is not None else pool(max_norm2=400)
    cand.sort(key=lambda d: (d.norm2, d.angle))
    return by_angle(cand[:k])


def slot_set(
    n: int,
    tol_deg: float,
    candidates: Sequence[Direction] | None = None,
    distinct: bool = True,
) -> list[Direction]:
    """Cheapest lattice representative of each of `n` equiangular slots.

    Slots are k * 90/n degrees for k = 0..n-1. For each slot, take the
    admissible direction of least cost whose angle is within `tol_deg` of the
    slot centre, breaking ties by closeness to the centre.

    This is the construction to prefer over `cheapest_k`. It keeps the
    equiangular schedule that makes the cyclic-shift orbit distance d_C
    meaningful (see `qsig.signature.d_C`), while choosing the representative by
    price rather than by proximity. It is directly comparable to IWCIA 2025's
    `S_int`, which uses the same slot idea with the *nearest* representative
    under a components<=10 box constraint.

    With `distinct=True` a direction already used by an earlier slot is not
    reused, and the next-cheapest admissible candidate is taken instead; with a
    tight slot spacing and a loose tolerance two adjacent slots would otherwise
    select the same vector.

    Raises ValueError if any slot has no admissible candidate.
    """
    cand = list(candidates) if candidates is not None else pool(max_norm2=4096)
    slots = [k * 90.0 / n for k in range(n)]
    used: set[tuple[int, int]] = set()
    chosen: list[Direction] = []
    for s in slots:
        adm = [d for d in cand if abs(d.angle - s) <= tol_deg]
        adm.sort(key=lambda d: (d.norm2, abs(d.angle - s)))
        pick = None
        for d in adm:
            if not distinct or (d.p, d.q) not in used:
                pick = d
                break
        if pick is None:
            raise ValueError(
                f"no admissible direction for slot {s:.2f} deg at tol {tol_deg} deg"
                + (" (all candidates already used)" if adm else "")
            )
        used.add((pick.p, pick.q))
        chosen.append(pick)
    return by_angle(chosen)


def min_cost_maxgap(max_gap_deg: float, candidates: Sequence[Direction] | None = None,
                    model: str = "rows+numba") -> list[Direction]:
    """Cheapest direction set whose maximum angular gap is at most `max_gap_deg`.

    THIS IS THE CONSTRUCTION THE PAPER USES. It supersedes `slot_set`.

    `slot_set` stated the problem the wrong way round: it fixed n equiangular
    slots and minimised cost within a tolerance of each. That lets neighbouring
    picks drift TOWARDS each other and open a hole elsewhere -- `slot9:±5°`
    produced an 18.4 degree maximum gap where equiangular S_9 has 10, and scored
    64.0% where the same budget spent on coverage scores 73.0% (handoff 3.2.1).
    Accuracy is governed by the maximum gap, so the gap is the CONSTRAINT and
    cost is the OBJECTIVE:

        minimise  sum c(r)   subject to   max angular gap <= G

    Solved EXACTLY, not greedily. Sort the candidates by angle; a valid set is a
    path through them whose consecutive angular steps are all <= G and which
    closes the cycle back to 0 degrees (Q-concavity has period 90). Shortest
    path on a DAG, O(n^2) in the pool size, optimal.

    (1,0) is always included: it is the cheapest direction in every cost model
    and anchors the cyclic wrap-around at 0/90 degrees.

    `model` names the cost model to minimise against -- see COST_MODELS. In
    practice the CHOICE OF MODEL DOES NOT CHANGE THE SELECTED SET (handoff
    3.1.3), so this argument matters for the reported price, not the answer.
    """
    cand = sorted(candidates if candidates is not None else pool(max_norm2=130),
                  key=lambda d: d.angle)
    if not cand or cand[0].angle != 0.0:
        raise ValueError("candidate pool must contain (1,0) at 0 degrees")
    n = len(cand)
    INF = float("inf")
    best = [INF] * n
    prev: list[int | None] = [None] * n
    best[0] = _cost1(cand[0], model)
    for i in range(n):
        if best[i] == INF:
            continue
        for j in range(i + 1, n):
            if cand[j].angle - cand[i].angle > max_gap_deg:
                break                      # sorted by angle, so no later j fits
            c = best[i] + _cost1(cand[j], model)
            if c < best[j]:
                best[j] = c
                prev[j] = i
    # close the cycle: the last direction must be within G of 90 degrees
    end, endcost = None, INF
    for i in range(n):
        if best[i] < INF and 90.0 - cand[i].angle <= max_gap_deg and best[i] < endcost:
            end, endcost = i, best[i]
    if end is None:
        raise ValueError(
            f"no set with maximum gap <= {max_gap_deg} deg exists in this pool "
            f"({n} candidates, largest angle {cand[-1].angle:.1f} deg)"
        )
    out: list[Direction] = []
    k: int | None = end
    while k is not None:
        out.append(cand[k])
        k = prev[k]
    return list(reversed(out))


def deg2vec_set(scale: int = 10, n_angles: int = 90) -> list[Direction]:
    """IWCIA 2025's own construction, as implemented rather than as described.

    Their sec 5.1 describes `S_int` as "for all integer angles in [0,90) the
    closest (in terms of direction angle) lattice direction d = (d1,d2) with
    0 <= d1,d2 <= 10". The repository's `Convexity.deg2vec` does something
    slightly different: it rounds the unit vector scaled by `scale` and reduces
    the resulting fraction,

        Fraction(round(scale*sin theta), round(scale*cos theta))

    The two differ. A true nearest search over the components<=10 box picks
    (5,4) for 38 degrees (0.66 deg away) where the round-and-reduce rule picks
    (4,3) (1.13 deg away), and yields more than 20 distinct directions overall.
    The published `S_int` is the round-and-reduce output, so that is what is
    implemented here -- verified to reproduce all 20 directions exactly.

    This matters when reproducing their protocol: the prose is a loose
    description of the code, and following the prose gives a different set.
    """
    from fractions import Fraction
    from math import cos, radians, sin

    seen: set[tuple[int, int]] = set()
    chosen: list[Direction] = []
    for k in range(n_angles):
        theta = radians(k * 90.0 / n_angles)
        num = int(round(sin(theta) * scale))
        den = int(round(cos(theta) * scale))
        if den == 0:
            continue                       # 90 degrees: (0,1) is not computable
        f = Fraction(num, den)
        p, q = f.denominator, f.numerator
        if p < 1 or q < 0 or (p, q) in seen:
            continue
        seen.add((p, q))
        chosen.append(Direction(p, q))
    return by_angle(chosen)


# The 20 directions of IWCIA 2025 Table 1, in signature order.
S_INT: list[Direction] = by_angle(
    Direction(p, q)
    for p, q in [
        (1, 0), (10, 1), (5, 1), (10, 3), (3, 1), (9, 4), (9, 5), (8, 5),
        (4, 3), (8, 7), (1, 1), (7, 8), (3, 4), (5, 8), (5, 9), (4, 9),
        (1, 3), (3, 10), (1, 5), (1, 10),
    ]
)

# Mean seconds per component reported in IWCIA 2025 Table 1, keyed by (p, q).
# Used by scripts/fit_cost_law.py and as a calibration target (handoff sec 1.3).
IWCIA_TABLE1 = {
    (1, 0): 2.54, (10, 1): 60.90, (5, 1): 15.96, (10, 3): 65.21, (3, 1): 7.38,
    (9, 4): 56.92, (9, 5): 62.07, (8, 5): 51.25, (4, 3): 15.05, (8, 7): 64.07,
    (1, 1): 3.10, (7, 8): 64.10, (3, 4): 15.06, (5, 8): 51.37, (5, 9): 62.13,
    (4, 9): 57.13, (1, 3): 7.38, (3, 10): 65.35, (1, 5): 15.99, (1, 10): 61.01,
}

# Flat cost per component of the *rotational* signature, seconds, from
# IWCIA 2025 sec 5.4: "the average time for calculating an element of any
# rotational signature is 2.54s". Independent of direction -- this is the
# frontier the cost-aware rotation-free sets must beat (handoff sec 3.3).
ROTATIONAL_SECONDS_PER_COMPONENT = 2.54
