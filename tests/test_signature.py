import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsig import classify
from qsig.signature import centred_orbit, d_C, euclidean, orbit, pairwise_d_C


def test_orbit_size_and_membership():
    u = np.arange(5.0)
    o = orbit(u)
    assert o.shape == (10, 5)
    rows = {tuple(r) for r in o}
    assert tuple(u) in rows
    assert tuple(u[::-1]) in rows
    # every cyclic shift is present
    for i in range(5):
        assert tuple(np.roll(u, i)) in rows


def test_d_C_is_zero_on_shifts_and_reversals():
    """IWCIA 2025 Fig 6: 'bat-1' is a rotated variant of 'bat-16' so its
    signature is SHIFTED; 'dog-12' is a reflected variant of 'dog-13' so its
    signature is REVERSED. d_C must be ~0 in both cases -- the two free sanity
    checks of handoff sec 7.4 item 8."""
    rng = np.random.default_rng(0)
    u = rng.random(20)
    for i in range(20):
        assert d_C(np.roll(u, i), u) == pytest.approx(0.0, abs=1e-12)
    assert d_C(u[::-1], u) == pytest.approx(0.0, abs=1e-12)
    assert d_C(np.roll(u, 7)[::-1], u) == pytest.approx(0.0, abs=1e-12)


def test_d_C_is_invariant_to_an_additive_offset():
    """Mean-centring means a uniform shift in descriptor level is free."""
    rng = np.random.default_rng(1)
    u, v = rng.random(12), rng.random(12)
    assert d_C(u + 3.5, v) == pytest.approx(d_C(u, v), rel=1e-12)


def test_d_C_is_symmetric():
    rng = np.random.default_rng(2)
    for _ in range(20):
        u, v = rng.random(9), rng.random(9)
        assert d_C(u, v) == pytest.approx(d_C(v, u), rel=1e-12)


def test_d_C_is_a_lower_bound_on_plain_l2():
    rng = np.random.default_rng(3)
    u, v = rng.random(10), rng.random(10)
    l2 = np.linalg.norm((u - u.mean()) - (v - v.mean()))
    assert d_C(u, v) <= l2 + 1e-12


def test_pairwise_matches_the_scalar_version():
    rng = np.random.default_rng(4)
    sigs = rng.random((13, 7))
    M = pairwise_d_C(sigs, block=5)
    assert M.shape == (13, 13)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)
    for i in range(13):
        for j in range(13):
            assert M[i, j] == pytest.approx(d_C(sigs[i], sigs[j]), abs=1e-10)


def test_centred_orbit_rows_have_zero_mean():
    rng = np.random.default_rng(5)
    o = centred_orbit(rng.random(8))
    assert np.allclose(o.mean(axis=1), 0.0, atol=1e-12)


def test_loo_1nn_is_perfect_on_separated_clusters():
    a = np.tile([0.0, 1.0, 0.0, 1.0], (10, 1)) + 1e-3
    b = np.tile([5.0, 9.0, 2.0, 0.0], (10, 1)) + 1e-3
    sigs = np.vstack([a, b])
    labels = ["a"] * 10 + ["b"] * 10
    assert classify.loo_1nn(sigs, labels, "l2") == pytest.approx(100.0)


def test_loo_1nn_excludes_the_query_itself():
    """Without leave-one-out every point is its own nearest neighbour and the
    accuracy would be a meaningless 100%."""
    rng = np.random.default_rng(6)
    sigs = rng.random((40, 6))
    labels = list(rng.integers(0, 4, 40))
    acc = classify.loo_1nn(sigs, labels, "l2")
    assert acc < 100.0


def test_euclidean_matches_numpy():
    rng = np.random.default_rng(7)
    s = rng.random((6, 4))
    E = euclidean(s)
    assert E[2, 5] == pytest.approx(np.linalg.norm(s[2] - s[5]))
