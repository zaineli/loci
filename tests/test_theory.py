"""The theory's moments are exact, and its predictions match simulated recall. Small cases, seconds."""

import numpy as np

from loci import place, theory
from loci.content import factored
from loci.memory import CueMaps, cue_moments, flip, mmse_alpha
from loci.scaffold import Scaffold, relu


def test_the_cue_coefficients_have_the_stated_mean_and_covariance():
    """c~ = K S^T s~ for a flipped cue of item i: mean a (e_i - alpha K e_i), covariance
    sigma^2 (K - alpha K^2). Checked by sampling cues."""
    rate, dim, count, draws = 0.1, 200, 30, 4000
    patterns = np.sign(np.random.default_rng(0).standard_normal((dim, count)))
    alpha = mmse_alpha(count, flip_rate=rate)
    precision = np.linalg.inv(patterns.T @ patterns + alpha * np.eye(count))
    gain, noise = cue_moments(rate)
    item = 3
    cue = np.repeat(patterns[:, [item]], draws, axis=1)
    coefficients = precision @ patterns.T @ flip(cue, rate, np.random.default_rng(1))
    mean = gain * (np.eye(count)[item] - alpha * precision[item])
    cov = noise * (precision - alpha * precision @ precision)
    assert np.abs(coefficients.mean(axis=1) - mean).max() < 4 * np.sqrt(cov.diagonal().max() / draws)
    assert np.abs(np.cov(coefficients) - cov).max() < 0.1 * np.abs(cov).max()


def test_the_prediction_matches_simulated_recall():
    """The relu level against the snap's recovery, averaged over several cue draws."""
    content = factored(1_000, 600, np.random.default_rng(0))
    scaffold = Scaffold(seed=0)
    alpha = mmse_alpha(600, flip_rate=0.1)
    maps = CueMaps(content.patterns)
    for where in (place.scattered(600, scaffold, np.random.default_rng(1)), place.oracle(scaffold, content.factors)):
        w_hs = maps.map(scaffold.H[:, where], alpha)
        measured = np.mean([
            np.mean(scaffold.index(scaffold.settle(relu(w_hs @ flip(content.patterns, 0.1, np.random.default_rng(r))))[0])
                    == where)
            for r in range(4)])
        predicted = theory.predict(scaffold, content.patterns, where, alpha, flip_rate=0.1).address
        assert abs(predicted - measured) < 0.03, (predicted, measured)


def test_the_error_level_overstates_recall_under_random_placement():
    """The error law alone has no template crosstalk, so it overstates random placement's recall."""
    content = factored(1_000, 800, np.random.default_rng(0))
    scaffold = Scaffold(seed=0)
    alpha = mmse_alpha(800, flip_rate=0.1)
    where = place.scattered(800, scaffold, np.random.default_rng(1))
    error = theory.predict(scaffold, content.patterns, where, alpha, flip_rate=0.1, level="error").address
    relu_level = theory.predict(scaffold, content.patterns, where, alpha, flip_rate=0.1).address
    assert error > relu_level + 0.15
