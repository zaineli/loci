"""The reimplementation against the paper's guarantees and our own theory. Small scaffolds, seconds."""

import numpy as np
import pytest

from loci.memory import Memory, flip, mmse_alpha, overlap, theory_overlap
from loci.scaffold import Scaffold

# 4 x 9 = 36 addresses. Enough place cells matters: at the paper's periods (3, 4, 5), 100 place
# cells leave only 62% of addresses as fixed points, 200 leave 95%, 400 all but at most one.
SMALL = {"periods": (2, 3), "place_cells": 400}


def _patterns(dim: int, count: int, seed: int = 0) -> np.ndarray:
    return np.sign(np.random.default_rng(seed).standard_normal((dim, count)))


def test_every_address_is_a_fixed_point():
    scaffold = Scaffold(**SMALL)
    phases, _ = scaffold.settle(scaffold.H)
    assert (scaffold.index(phases) == np.arange(scaffold.addresses)).all()


def test_the_address_index_inverts_the_phases():
    scaffold = Scaffold()
    x = np.arange(0, scaffold.addresses, 37)
    assert (scaffold.index(scaffold.phases[x]) == x).all()


def test_nearness_is_shared_module_phases_only():
    scaffold = Scaffold()
    assert scaffold.shared(0, 1) == 0, "adjacent positions share no module phase"
    assert scaffold.shared(0, 9 * 16) == 2, "a shift by l_1 * l_2 keeps two phases"


def test_recall_is_exact_up_to_the_knee():
    scaffold = Scaffold(**SMALL)
    patterns = _patterns(400, scaffold.addresses)
    memory = Memory(scaffold)
    memory.store(patterns, np.arange(scaffold.addresses))
    got = memory.recall(patterns)
    assert (got.address == np.arange(scaffold.addresses)).all()
    assert overlap(got.content, patterns).mean() == 1.0


@pytest.mark.parametrize("stored", [600, 900])
def test_past_the_knee_the_overlap_follows_the_erf_law(stored):
    scaffold = Scaffold()
    patterns = _patterns(1_000, stored)
    memory = Memory(scaffold)
    memory.store(patterns, np.arange(stored))
    measured = overlap(memory.recall(patterns).content, patterns).mean()
    assert abs(measured - theory_overlap(scaffold.place_cells, stored)) < 0.02


def test_too_few_place_cells_break_the_scaffold():
    """Not a property of storage: the address space itself stops being a set of fixed points."""
    thin = Scaffold(place_cells=100)
    phases, _ = thin.settle(thin.H)
    assert (thin.index(phases) == np.arange(thin.addresses)).mean() < 0.8


def test_ridge_at_zero_is_the_pseudo_inverse():
    scaffold = Scaffold(**SMALL)
    patterns = _patterns(200, 30)
    pinv, ridge = Memory(scaffold), Memory(scaffold, rule="ridge", alpha=1e-9)
    pinv.store(patterns, np.arange(30))
    ridge.store(patterns, np.arange(30))
    assert pinv.W_hs is not None and ridge.W_hs is not None
    assert np.allclose(pinv.W_hs, ridge.W_hs, atol=1e-6)


def test_flip_rate_is_the_fraction_flipped():
    patterns = _patterns(1_000, 200)
    flipped = flip(patterns, 0.1, np.random.default_rng(1))
    assert abs((flipped != patterns).mean() - 0.1) < 0.005


def test_the_mmse_alpha_is_zero_for_clean_cues_and_grows_with_noise():
    assert mmse_alpha(1_000) == 0.0
    assert mmse_alpha(1_000, flip_rate=0.2) > mmse_alpha(1_000, flip_rate=0.1) > 0
    assert mmse_alpha(1_000, mask_rate=0.5) == 1_000, "r / (1 - r) per stored item at r = 0.5"
