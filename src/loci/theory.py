"""Predicting recall without running it: a Gaussian theory of the cue path, with no fitted parameters.

Given the stored patterns S, the scaffold and a placement, this predicts each module's accuracy and
the address recovery for noisy cues. It never draws a cue.

The chain, from the cue to the snap:
  c~ = K S^T s~,  K = (S^T S + alpha I)^-1    the cue's coefficients over the stored items. For a cue
                                             a s_i + xi, with Cov(xi) = sigma^2 I, they have exactly
      E c~   = a (e_i - alpha K e_i)            (the bias: a clean cue spreads onto similar items)
      Cov c~ = sigma^2 (K - alpha K^2)          (the variance)
  u  = H_a c~                                 place-cell input; Gaussian by the CLT over Ns bits
  h0 = ReLU(u)                                moment-matched per cell (exact mean and variance, and
                                             the covariance to first order in the Hermite expansion)
  x  = W_gh h0                                the grid input, Gaussian over all grid cells jointly
  snap = per-module argmax of x               accuracy by sampling that Gaussian

Three levels, kept separate because what each one misses is the finding:
  "error"   the error law alone: x = G_a c~ through the phase indicators, no templates, no ReLU
  "linear"  through the scaffold's templates W_gh H_a, no ReLU
  "relu"    through the templates and the ReLU (the default)

At the bench's conditions (180 of them: 6 placements x 5 loads x 2 flip rates x 3 seeds), "relu"
predicts address recovery to a mean absolute error of 0.010, which is the sampling noise of one cue
per item. "linear" gets 0.012. "error" gets 0.132: it orders placements, but it cannot give their
magnitudes. The error law is the right object (K); the phase indicators are the wrong map. What
recall actually sees is K projected through the scaffold's own templates.

What it is not: a fit. Nothing here reads a recall result. It is also not a closed form: the argmax
probabilities are Monte Carlo over a 50-dimensional Gaussian (`draws` per item), because the joint
argmax over three modules has no convenient closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from loci.memory import cue_moments
from loci.scaffold import Scaffold

LEVELS = ("error", "linear", "relu")


@dataclass(frozen=True)
class Prediction:
    modules: np.ndarray  # (M,) predicted per-module accuracy, averaged over items
    address: float  # predicted address recovery: every module right at once


def _root(cov: np.ndarray) -> np.ndarray:
    """A square root of a covariance, robust to tiny negative eigenvalues."""
    w, v = np.linalg.eigh(cov)
    return v * np.sqrt(np.clip(w, 0, None))


def _argmax_accuracy(scaffold: Scaffold, mean: np.ndarray, root: np.ndarray, truth: np.ndarray,
                     rng: np.random.Generator, draws: int) -> Prediction:
    """Every item shares one noise law N(0, root root^T) around its own mean (n, grid cells)."""
    count = mean.shape[0]
    noise = rng.standard_normal((draws, root.shape[0])) @ root.T
    modules, address = np.zeros(len(scaffold.sizes)), 0.0
    for start in range(0, count, 100):
        x = mean[start:start + 100, None, :] + noise[None]
        right = np.ones(x.shape[:2], dtype=bool)
        for m, (offset, size) in enumerate(zip(scaffold.offsets, scaffold.sizes, strict=True)):
            hit = x[..., offset:offset + size].argmax(-1) == truth[start:start + 100, m][:, None]
            modules[m] += hit.mean(1).sum()
            right &= hit
        address += right.mean(1).sum()
    return Prediction(modules / count, float(address / count))


def predict(scaffold: Scaffold, patterns: np.ndarray, where: np.ndarray, alpha: float,
            flip_rate: float = 0.0, mask_rate: float = 0.0, level: str = "relu",
            rng: np.random.Generator | None = None, draws: int = 400) -> Prediction:
    """Predicted snap accuracy for patterns (Ns x P) stored at `where`, cued with bits flipped at
    `flip_rate` and/or unknown at `mask_rate`, read through the cue map at ridge strength `alpha`."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    rng = rng if rng is not None else np.random.default_rng(0)
    gain, noise = cue_moments(flip_rate, mask_rate)
    count = patterns.shape[1]
    precision = np.linalg.inv(patterns.T @ patterns + alpha * np.eye(count))
    mean_c = gain * (np.eye(count) - alpha * precision)  # column i: E c~ when item i is cued
    cov_c = noise * (precision - alpha * precision @ precision)
    truth = scaffold.phases[where]
    if level == "error":
        indicators = scaffold.G[:, where]  # (grid cells, P): each module's phase indicators, stacked
        return _argmax_accuracy(scaffold, (indicators @ mean_c).T,
                                _root(indicators @ cov_c @ indicators.T), truth, rng, draws)
    codes = scaffold.H[:, where]
    mean_u = (codes @ mean_c).T  # (P, Nh)
    cov_u = codes @ cov_c @ codes.T  # (Nh, Nh), the same for every cued item
    if level == "linear":
        return _argmax_accuracy(scaffold, mean_u @ scaffold.W_gh.T,
                                _root(scaffold.W_gh @ cov_u @ scaffold.W_gh.T), truth, rng, draws)
    sd = np.sqrt(np.diag(cov_u))
    z = mean_u / sd
    mean_h = mean_u * norm.cdf(z) + sd * norm.pdf(z)
    var_h = (mean_u**2 + sd**2) * norm.cdf(z) + mean_u * sd * norm.pdf(z) - mean_h**2
    slope = norm.cdf(z)  # d E[ReLU] / d mean: the first Hermite coefficient, per item and cell
    shared = rng.standard_normal((draws, scaffold.grid_cells))
    modules, address = np.zeros(len(scaffold.sizes)), 0.0
    for i in range(count):
        through = scaffold.W_gh * slope[i]
        cov = through @ cov_u @ through.T + (scaffold.W_gh * (var_h[i] - slope[i] ** 2 * sd**2)) @ scaffold.W_gh.T
        x = (scaffold.W_gh @ mean_h[i])[None] + shared @ _root(cov).T
        right = np.ones(draws, dtype=bool)
        for m, (offset, size) in enumerate(zip(scaffold.offsets, scaffold.sizes, strict=True)):
            hit = x[:, offset:offset + size].argmax(1) == truth[i, m]
            modules[m] += hit.mean()
            right &= hit
        address += right.mean()
    return Prediction(modules / count, float(address / count))
