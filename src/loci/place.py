"""Where each memory goes: the placements E2 compares, and the one that is learned.

A placement is a map from P stored items to P distinct scaffold addresses. It enters recall only
through which place codes the items get, and with a linear cue map the noise on module m's phase k
is sigma^2 q^T Gamma q, q marking the items on that phase and Gamma = (G + a I)^-1 G (G + a I)^-1
for the Gram matrix G = S^T S (the pseudo-inverse is a = 0, Gamma = G^-1). Correlated items have
negative off-diagonal entries in Gamma, so putting them on one phase cancels noise - the best
placement groups items by partial correlation.

The content-driven placements (k-means, learned, oracle) choose an item's phase in the first two
modules and a group of five phases in the last; the phase inside that group is a free slot, found
by open addressing - so similar items share phases without ever sharing an address (two items at
one address read out as their mean).

What it is not: a read path. Recall stays the paper's linear cue map (`memory.py`); a learned map
from cues to per-module phases would be product quantisation, and the scaffold would be decoration.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from loci.scaffold import Scaffold

# Per-module group counts for the content-driven modules; the last module's 25 phases are 5 groups
# of 5 slots. Matches the factor structure of E2's content, and is shared by every placement.
GROUPS = (9, 16, 5)
SLOTS = 5


def sequential(count: int) -> np.ndarray:
    """The paper's: item j at address j."""
    return np.arange(count)


def scattered(count: int, scaffold: Scaffold, rng: np.random.Generator) -> np.ndarray:
    """Every item at a uniformly random free address."""
    return rng.permutation(scaffold.addresses)[:count]


# ---- from group labels to addresses ----------------------------------------------------------------


def _candidates(scaffold: Scaffold, want: tuple[int, int, int]):
    """Phase tuples to try for an item whose content chose (phase 0, phase 1, last-module group):
    the group's own slots first, then the same first two phases anywhere, then one phase, then any."""
    p0, p1, group = want
    last = scaffold.sizes[2]
    yield from ((p0, p1, group * SLOTS + s) for s in range(SLOTS))
    yield from ((p0, p1, k) for k in range(last))
    yield from ((p0, k1, k2) for k1, k2 in product(range(scaffold.sizes[1]), range(last)))
    yield from product(*(range(size) for size in scaffold.sizes))


def addresses(scaffold: Scaffold, groups: np.ndarray) -> np.ndarray:
    """(P, 3) labels - phase in module 0, phase in module 1, group in the last - to distinct
    addresses, by open addressing: the nearest free address in shared-residue terms."""
    taken = np.zeros(scaffold.sizes, dtype=bool)
    out = np.empty(len(groups), dtype=np.int64)
    for i, want in enumerate(groups):
        phases = next(c for c in _candidates(scaffold, tuple(int(x) for x in want)) if not taken[c])
        taken[phases] = True
        out[i] = scaffold.index(np.array([phases]))[0]
    return out


def oracle(scaffold: Scaffold, factors: np.ndarray) -> np.ndarray:
    """The true factors as the address: A -> module 0, B -> module 1, C -> the last module's group."""
    return addresses(scaffold, factors)


# ---- clustering, and the noise law --------------------------------------------------------------


def _pinv_gram(patterns: np.ndarray, power: int) -> np.ndarray:
    """(S^T S)^+ ** power by the SVD, with `memory.CueMaps`' cut-off: dense content (sign
    projections of 384-d embeddings) leaves S^T S too ill-conditioned for a plain inverse."""
    _, sigma, vt = np.linalg.svd(patterns, full_matrices=False)
    keep = sigma > 1e-10 * sigma.max()
    v = vt[keep].T
    return (v / sigma[keep] ** (2 * power)) @ v.T


def gamma(patterns: np.ndarray, alpha: float) -> np.ndarray:
    """The noise law: the P x P matrix whose quadratic form is the variance cue noise puts on each
    phase, (G + a I)^-1 G (G + a I)^-1 for G = S^T S (the pseudo-inverse's G^-1 at a = 0)."""
    if alpha == 0:
        return _pinv_gram(patterns, 1)
    gram = patterns.T @ patterns
    inverse = np.linalg.inv(gram + alpha * np.eye(len(gram)))
    return inverse @ gram @ inverse


def precision(patterns: np.ndarray, alpha: float) -> np.ndarray:
    """The error law: (G + a I)^-1, the regularised precision of the stored items.

    The noise law leaves out the ridge's bias - the part of a clean cue it spreads onto similar
    items - and on dense content that is most of the error. Bias and variance together, at the
    noise-matched ridge a = Var(xi) P / a_s^2, sum to Var(xi) q^T (G + a I)^-1 q on every phase:
    the posterior covariance of the cue's coefficients over the stored items (the Bayesian
    linear-regression identity, Bishop 2006, section 3.3). At a = 0 it is the noise law.
    """
    if alpha == 0:
        return _pinv_gram(patterns, 1)
    return np.linalg.inv(patterns.T @ patterns + alpha * np.eye(patterns.shape[1]))


def law(kernel: np.ndarray, labels: np.ndarray) -> float:
    """Mean over occupied groups of q^T K q for one module's labels."""
    return float(np.mean([kernel[np.ix_(m, m)].sum() for k in np.unique(labels) if (m := labels == k).any()]))


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, steps: int = 30) -> np.ndarray:
    """Plain Lloyd's on the columns of x; the non-learned control."""
    centres = x[:, rng.choice(x.shape[1], k, replace=False)]
    labels = np.zeros(x.shape[1], dtype=np.int64)
    for _ in range(steps):
        distance = (centres**2).sum(0)[:, None] - 2 * centres.T @ x
        labels = distance.argmin(axis=0)
        for j in range(k):
            if (labels == j).any():
                centres[:, j] = x[:, labels == j].mean(axis=1)
    return labels


def _residual(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """What a module's groups leave unexplained: each item minus its group's mean."""
    out = x.copy()
    for k in np.unique(labels):
        out[:, labels == k] -= x[:, labels == k].mean(axis=1, keepdims=True)
    return out


def _within(off: np.ndarray, labels: np.ndarray) -> float:
    """sum_k q_k^T off q_k: the part of the noise law a placement can change (for groups of fixed
    sizes the diagonal is the same for every placement)."""
    onehot = np.eye(labels.max() + 1)[labels]
    return float((onehot * (off @ onehot)).sum())


def _swaps(off: np.ndarray, labels: np.ndarray, rng: np.random.Generator, sweeps: int = 50) -> np.ndarray:
    """Local search on `_within`: take items in random order, make the best swap with an item of
    another group if it lowers the law, stop when a sweep changes nothing. A swap keeps every
    group's size, so the groups stay as balanced as they started."""
    labels = labels.copy()
    rows = np.arange(len(labels))
    coupling = off @ np.eye(labels.max() + 1)[labels]  # coupling[i, k] = (off q_k)_i
    tolerance = 1e-9 * np.abs(off).max()
    for _ in range(sweeps):
        moved = False
        for i in rng.permutation(len(labels)):
            a = labels[i]
            change = 2 * (coupling[i, labels] - coupling[i, a] + coupling[rows, a] - coupling[rows, labels])
            change -= 4 * off[i]
            change[labels == a] = np.inf
            j = int(change.argmin())
            if change[j] < -tolerance:
                b = labels[j]
                coupling[:, a] += off[:, j] - off[:, i]
                coupling[:, b] += off[:, i] - off[:, j]
                labels[i], labels[j] = b, a
                moved = True
        if not moved:
            break
    return labels


def kmeans(
    patterns: np.ndarray, scaffold: Scaffold, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """(addresses, labels) by residual k-means - a residual quantiser, the non-learned control:
    one module at a time, the next on what the last left unexplained, by raw similarity."""
    x, labels = patterns.astype(float), []
    for k in GROUPS:
        labels.append(_kmeans(x, k, rng))
        x = _residual(x, labels[-1])
    stacked = np.stack(labels, axis=1)
    return addresses(scaffold, stacked), stacked


def learned(
    patterns: np.ndarray,
    scaffold: Scaffold,
    rng: np.random.Generator,
    alpha: float,
    error: bool = False,
    starts: int = 20,
    warm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(addresses, labels) minimising a law at ridge strength `alpha`, module by module: the noise
    law (`gamma`), or with `error` the error law (`precision`).

    Each module's groups are the lowest-law result of pairwise-swap descent from `starts` random
    balanced labellings - and from `warm`'s column for that module, if given - on the stored
    patterns' own kernel. Nothing ties one module to another: with k groups, the partition that
    cancels the most noise is the one along a factor with about k values, so which module takes
    which factor falls out of the law. (A residual chain - module m + 1 on what module m left -
    re-found module m's partition, because Gamma amplifies the low-variance directions a residual
    leaves; soft Sinkhorn relaxations stalled in mixtures of factors. Both are in docs/PREREG.md.)
    """
    kernel = precision(patterns, alpha) if error else gamma(patterns, alpha)
    off = kernel - np.diag(np.diag(kernel))
    count, labels = patterns.shape[1], []
    for m, k in enumerate(GROUPS):
        runs = [_swaps(off, rng.permutation(np.arange(count) % k), rng) for _ in range(starts)]
        if warm is not None:
            runs.append(_swaps(off, warm[:, m], rng))
        labels.append(min(runs, key=lambda run: _within(off, run)))
    stacked = np.stack(labels, axis=1)
    return addresses(scaffold, stacked), stacked
