"""Shared set-up for the theorist's checks: the bench's exact conditions, rebuilt from the repo."""
import os
for _t in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_t, "2")
import sys
sys.path.insert(0, "/Users/zain/work/systems/loci/src")
import numpy as np
from loci import place
from loci.content import factored
from loci.memory import Memory, flip, mmse_alpha, CueMaps
from loci.scaffold import Scaffold, relu

DIM = 1000


def bench_condition(seed, load, rate=0.1):
    """Content, scaffold, alpha and cues exactly as bench/placement.py builds them."""
    count = round(load * DIM)
    content = factored(DIM, count, np.random.default_rng(10_000 * seed + count))
    grid = Scaffold(seed=seed)
    alpha = mmse_alpha(count, flip_rate=rate)
    cues = flip(content.patterns, rate, np.random.default_rng(20_000 * seed + count + int(1_000 * rate)))
    return content, grid, alpha, cues


def placements(content, grid, seed, which=("sequential", "random", "kmeans", "oracle"), rate=0.1):
    S, count = content.patterns, content.patterns.shape[1]
    out = {}
    if "sequential" in which:
        out["sequential"] = place.sequential(count)
    if "random" in which:
        out["random"] = place.scattered(count, grid, np.random.default_rng(seed + 1))
    if "kmeans" in which:
        out["kmeans"] = place.kmeans(S, grid, np.random.default_rng(seed + 2))[0]
    if "oracle" in which:
        out["oracle"] = place.oracle(grid, content.factors)
    if "learned" in which or "error" in which:
        alpha = mmse_alpha(count, flip_rate=rate)
        learned, labels = place.learned(S, grid, np.random.default_rng(seed + 3), alpha)
        if "learned" in which:
            out["learned"] = learned
        if "error" in which:
            out["error"] = place.learned(S, grid, np.random.default_rng(seed + 4), alpha, error=True, warm=labels)[0]
    return out


def ridge_recall(grid, S, where, cues, alpha):
    """The bench's ridge read: per-module accuracy and address recovery."""
    maps = CueMaps(S)
    h0 = relu(maps.map(grid.H[:, where], alpha) @ cues)
    phases, _ = grid.settle(h0)
    got = grid.index(phases)
    return float(np.mean(got == where)), (grid.phases[got] == grid.phases[where]).mean(0)


def nmi(x, y):
    j = np.zeros((x.max() + 1, y.max() + 1)); np.add.at(j, (x, y), 1); j /= j.sum()
    px, py = j.sum(1), j.sum(0); nz = j > 0
    mi = (j[nz] * np.log(j[nz] / np.outer(px, py)[nz])).sum()
    H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return 2 * mi / (H(px) + H(py))


def J_of(K, labels):
    """Error law as a SUM over groups and modules: sum_m sum_k q_mk^T K q_mk."""
    labels = np.atleast_2d(labels.T).T
    tot = 0.0
    for m in range(labels.shape[1]):
        Q = np.eye(labels[:, m].max() + 1)[labels[:, m]]
        tot += float(np.einsum("ik,ij,jk->", Q, K, Q))
    return tot
