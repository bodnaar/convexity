import os
import sys
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from qsig import directions as D


def test_vec_convention_is_lower_right():
    """Convexity.rotsat needs the second component non-positive."""
    assert D.Direction(1, 0).vec == (1, 0)
    assert D.Direction(3, 1).vec == (3, -1)
    assert D.Direction(1, 10).vec == (1, -10)


def test_angles():
    assert D.Direction(1, 0).angle == pytest.approx(0.0)
    assert D.Direction(1, 1).angle == pytest.approx(45.0)
    assert D.Direction(10, 3).angle == pytest.approx(16.699, abs=1e-3)
    assert D.Direction(1, 10).angle == pytest.approx(84.289, abs=1e-3)


def test_norm2_is_det_of_the_orthogonal_pair():
    for p, q in [(1, 0), (1, 1), (3, 1), (10, 3), (8, 5)]:
        d = D.Direction(p, q)
        # s = r_perp = (q, -p);  |det(r,s)| = |p*(-p) - q*q| = p^2 + q^2
        assert abs(p * (-p) - q * q) == d.norm2


def test_pool_is_primitive_sorted_and_in_range():
    pl = D.pool(max_norm2=50)
    assert all(gcd(d.p, d.q) == 1 for d in pl)
    assert all(0 <= d.angle < 90 for d in pl)
    assert all(d.norm2 <= 50 for d in pl)
    assert [d.norm2 for d in pl] == sorted(d.norm2 for d in pl)
    assert len({(d.p, d.q) for d in pl}) == len(pl)


def test_pool_requires_exactly_one_budget():
    with pytest.raises(ValueError):
        D.pool()
    with pytest.raises(ValueError):
        D.pool(max_norm2=10, max_component=10)


def test_rejects_non_primitive_and_out_of_quadrant():
    for bad in [(2, 2), (0, 1), (-1, 1), (1, -1)]:
        with pytest.raises(ValueError):
            D.Direction(*bad)


def test_s_int_matches_the_published_table():
    assert len(D.S_INT) == 20
    assert {(d.p, d.q) for d in D.S_INT} == set(D.IWCIA_TABLE1)
    assert [round(d.angle) for d in D.S_INT] == [
        0, 6, 11, 17, 18, 24, 29, 32, 37, 41, 45, 49, 53, 58, 61, 66, 72, 73, 79, 84
    ]


def test_deg2vec_set_reproduces_s_int_exactly():
    """The repo's round-and-reduce rule, not the prose's nearest search."""
    got = D.deg2vec_set(scale=10)
    assert len(got) == 20
    assert {(d.p, d.q) for d in got} == {(d.p, d.q) for d in D.S_INT}


def test_prose_nearest_search_does_NOT_reproduce_s_int():
    """Guards the discrepancy documented in deg2vec_set: following IWCIA sec
    5.1's wording literally gives a different, larger set -- e.g. (5,4) rather
    than (4,3) at 38 degrees. If this ever starts passing, the docstring is
    wrong and the protocol note in the paper needs revisiting."""
    box = D.pool(max_component=10)
    nearest = {min(box, key=lambda d: (abs(d.angle - a), d.norm2)) for a in range(90)}
    assert {(d.p, d.q) for d in nearest} != {(d.p, d.q) for d in D.S_INT}
    assert D.Direction(5, 4) in nearest and D.Direction(5, 4) not in D.S_INT


def test_cost_law_beats_the_published_bound_on_their_own_table():
    """The central empirical claim of the paper, as a regression test."""
    import numpy as np

    items = sorted(D.IWCIA_TABLE1.items())
    y = np.array([t for _, t in items])

    def r2(x):
        x = np.asarray(x, float)
        A = np.vstack([x, np.ones_like(x)]).T
        (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
        p = a + b * x
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    r2_norm = r2([p * p + q * q for (p, q), _ in items])
    r2_box = r2([(p + q) ** 2 for (p, q), _ in items])
    assert r2_norm > 0.99
    assert r2_box < 0.95
    assert r2_norm > r2_box


def test_s_int_total_matches_published_803_97():
    assert D.total_cost(D.S_INT) == pytest.approx(803.97, abs=2.0)


def test_slot_set_is_equiangular_and_cheaper_than_s_int():
    s = D.slot_set(20, 2.5)
    assert len(s) == 20
    assert len({(d.p, d.q) for d in s}) == 20          # distinct
    for k, d in enumerate(s):
        assert abs(d.angle - k * 4.5) <= 2.5
    assert D.total_cost(s) < D.total_cost(D.S_INT)


def test_slot_set_raises_when_infeasible():
    with pytest.raises(ValueError):
        D.slot_set(20, 0.001, candidates=D.pool(max_norm2=10))


def test_cheapest_k_is_not_equiangular():
    """Guards the warning in qsig.signature: cheapest-k has irregular gaps, so
    a cyclic shift is not a rotation and d_C is unjustified on it."""
    c = D.cheapest_k(10)
    gaps = sorted(c[i + 1].angle - c[i].angle for i in range(len(c) - 1))
    assert gaps[-1] - gaps[0] > 5.0


def test_max_angular_gap_is_cyclic_on_0_90():
    assert D.max_angular_gap([D.Direction(1, 0)]) == 90.0
    two = [D.Direction(1, 0), D.Direction(1, 1)]
    assert D.max_angular_gap(two) == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# min_cost_maxgap -- the construction the paper uses (handoff sec 3.2.1)
# ---------------------------------------------------------------------------

def test_maxgap_respects_the_constraint():
    for G in (6.0, 8.0, 10.0, 15.0, 20.0):
        S = D.min_cost_maxgap(G)
        assert D.max_angular_gap(S) <= G + 1e-9, f"G={G}: got {D.max_angular_gap(S)}"


def test_maxgap_is_optimal_not_greedy():
    """Brute-force the optimum on a small pool and require the DP to match it.

    A greedy sweep is not optimal here -- taking the cheapest admissible next
    direction can force an expensive one later -- so this is a real check.
    """
    import itertools

    cand = D.pool(max_norm2=26)
    G = 20.0
    best = None
    for k in range(2, 7):
        for combo in itertools.combinations(cand, k):
            if D.Direction(1, 0) not in combo:
                continue
            if D.max_angular_gap(combo) <= G + 1e-9:
                c = D.cost_of(combo)
                if best is None or c < best:
                    best = c
    got = D.cost_of(D.min_cost_maxgap(G, cand))
    assert got == pytest.approx(best, rel=1e-12), f"DP {got} vs brute force {best}"


def test_maxgap_beats_slot_set_at_equal_coverage():
    """The reason slot_set was superseded: at the same coverage it costs more.

    slot9:+-5 deg yields an 18.4 deg gap for 0.0358 s; the DP reaches the same
    coverage for less, with fewer directions (handoff sec 3.2.1).
    """
    slot = D.slot_set(9, 5.0)
    g = D.max_angular_gap(slot)
    dp = D.min_cost_maxgap(g)
    assert D.max_angular_gap(dp) <= g + 1e-9
    assert D.cost_of(dp) < D.cost_of(slot)
    assert len(dp) <= len(slot)


def test_maxgap_always_includes_the_axis_direction():
    for G in (8.0, 12.0, 20.0):
        assert D.Direction(1, 0) in D.min_cost_maxgap(G)


def test_maxgap_cost_and_size_are_monotone_in_G():
    prev_cost, prev_k = 0.0, 0
    for G in (25.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0):
        S = D.min_cost_maxgap(G)
        assert D.cost_of(S) >= prev_cost - 1e-12, f"cost not monotone at G={G}"
        assert len(S) >= prev_k, f"size not monotone at G={G}"
        prev_cost, prev_k = D.cost_of(S), len(S)


def test_maxgap_raises_when_the_pool_cannot_cover():
    with pytest.raises(ValueError):
        D.min_cost_maxgap(1.0, D.pool(max_norm2=10))


def test_cost_models_are_all_usable():
    for model in D.COST_MODELS:
        S = D.min_cost_maxgap(10.0, model=model)
        assert D.cost_of(S, model) > 0
        assert D.max_angular_gap(S) <= 10.0 + 1e-9


def test_the_slot_construction_was_model_invariant_but_the_DP_IS_NOT():
    """Corrects an over-claim. Handoff rev 9 sec 3.1.3 reported that the choice
    of cost model never changes the selected set. That was measured on
    `slot_set`, where it is true -- each slot picks its own cheapest
    representative and, within a narrow angular window, every norm orders the
    candidates identically.

    `min_cost_maxgap` optimises GLOBALLY: it trades a cheap direction now
    against a larger step later, and that trade depends on relative prices. So
    its optimum genuinely depends on the cost model. The invariance was a
    property of the weaker construction, not of the problem.
    """
    slot_a = [(d.p, d.q) for d in D.slot_set(12, 3.75)]
    assert slot_a == [(d.p, d.q) for d in D.slot_set(12, 3.75)]      # deterministic

    differs = False
    for G in (6.0, 10.0, 12.0):
        a = [(d.p, d.q) for d in D.min_cost_maxgap(G, model="fast+numba")]
        b = [(d.p, d.q) for d in D.min_cost_maxgap(G, model="rows+numba")]
        differs = differs or a != b
    assert differs, "if this now passes, re-examine the claim in handoff sec 3.1.3"


def test_selecting_with_the_wrong_cost_model_stays_FEASIBLE():
    """The robustness that survives: the CONSTRAINT is model-independent.

    Optimising against the wrong cost model costs more, but never violates the
    coverage requirement -- so a set chosen for one implementation is always
    usable on another, just not optimal there. That is what lets the paper state
    one construction and report its price per implementation.
    """
    for G in (6.0, 8.0, 10.0, 12.0, 15.0, 20.0):
        for model in ("fast+numba", "rows+numba"):
            other = "rows+numba" if model == "fast+numba" else "fast+numba"
            wrong = D.min_cost_maxgap(G, model=other)
            assert D.max_angular_gap(wrong) <= G + 1e-9


def test_the_wrong_model_penalty_is_bounded():
    """Measured 2026-09-06: 0% at G in {8,15,20}, 5.9-24.4% at G in {6,10,12}.
    A regression guard, and the number the paper should quote when it says the
    selection is robust-but-not-invariant to the cost model."""
    worst = 0.0
    for G in (6.0, 8.0, 10.0, 12.0, 15.0, 20.0):
        for model in ("fast+numba", "rows+numba"):
            other = "rows+numba" if model == "fast+numba" else "fast+numba"
            own = D.cost_of(D.min_cost_maxgap(G, model=model), model)
            oth = D.cost_of(D.min_cost_maxgap(G, model=other), model)
            worst = max(worst, (oth - own) / own)
    assert worst <= 0.30, f"wrong-model penalty grew to {worst:.1%}"
    assert worst > 0.0, "if this is now zero the sets have become invariant again"


def test_cost_of_rejects_an_unknown_model():
    with pytest.raises(ValueError):
        D.cost_of(D.S_INT, "nonexistent")
