"""Placements and the laws they minimise. Small instances, seconds."""

import numpy as np

from loci import place
from loci.content import factored
from loci.memory import flip, mmse_alpha
from loci.scaffold import Scaffold


def _nmi(x: np.ndarray, y: np.ndarray) -> float:
    joint = np.zeros((x.max() + 1, y.max() + 1))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum()
    entropy = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(2 * mi / (entropy(px) + entropy(py)))


def test_swaps_keep_group_sizes_and_never_raise_the_law():
    content = factored(300, 200, np.random.default_rng(0))
    kernel = place.gamma(content.patterns, 50.0)
    off = kernel - np.diag(np.diag(kernel))
    rng = np.random.default_rng(1)
    start = rng.permutation(np.arange(200) % 9)
    end = place._swaps(off, start, rng)
    assert (np.bincount(end, minlength=9) == np.bincount(start, minlength=9)).all()
    assert place._within(off, end) <= place._within(off, start)


def test_the_learned_placement_finds_the_hidden_factors():
    """Nothing tells it the factors; with k groups the law's minimum is the factor with ~k values."""
    content = factored(600, 400, np.random.default_rng(0))
    scaffold = Scaffold()
    where, labels = place.learned(content.patterns, scaffold, np.random.default_rng(1), mmse_alpha(400, 0.1))
    assert len(np.unique(where)) == 400
    for module in range(3):
        assert _nmi(labels[:, module], content.factors[:, module]) > 0.8


def test_the_oracle_puts_each_factor_on_its_module():
    content = factored(300, 400, np.random.default_rng(0))
    scaffold = Scaffold()
    where = place.oracle(scaffold, content.factors)
    assert len(np.unique(where)) == 400
    phases = scaffold.phases[where]
    kept = (phases[:, :2] == content.factors[:, :2]).all(axis=1)
    assert kept.mean() > 0.95, "open addressing moves an item off its factors only when its group is full"
    assert (phases[kept, 2] // place.SLOTS == content.factors[kept, 2]).all()


def test_without_ridge_the_two_laws_agree():
    patterns = factored(300, 200, np.random.default_rng(0)).patterns
    assert np.allclose(place.gamma(patterns, 0.0), place.precision(patterns, 0.0))


def test_the_error_law_is_the_mean_squared_error_of_the_cue_coefficients():
    """At the noise-matched ridge, bias plus variance of the coefficients c = (G + a I)^-1 S^T s~
    against the target (1 - 2f) e_mu is Var(xi) (G + a I)^-1 - checked by sampling flipped cues."""
    rate, dim, count, draws = 0.1, 200, 40, 400
    patterns = np.sign(np.random.default_rng(0).standard_normal((dim, count)))
    alpha = mmse_alpha(count, flip_rate=rate)
    solve = np.linalg.inv(patterns.T @ patterns + alpha * np.eye(count)) @ patterns.T
    rng, scale, total = np.random.default_rng(1), 1 - 2 * rate, np.zeros((count, count))
    for _ in range(draws):
        error = solve @ flip(patterns, rate, rng) - scale * np.eye(count)
        total += error @ error.T / count
    measured = total / draws
    predicted = 4 * rate * (1 - rate) * place.precision(patterns, alpha)
    assert np.abs(measured - predicted).max() < 0.1 * np.abs(predicted).max()
