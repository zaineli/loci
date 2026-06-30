"""Encoding and consolidation driven by recall: no labels, no group counts, one pass plus replay.

**Encoding.** Items arrive one at a time. The stored items explain a new item s with the ridge
coefficients c = K S^T s, i.e. its own recall. Putting s on phase k of module m changes the error law
J = sum_m sum_k q_mk^T K q_mk by exactly

    Delta J_m(k) = || Q_m^T c - e_k ||^2 / r,    r = s^T s + alpha - s^T S c   (s's prediction error),

(Schur complement; the theory note derives it). Only -2 (Q_m^T c)_k / r depends on k. So the best
address for s is the free one that maximises the sum, over modules, of s's summed recall coefficients
on that address's phases. **Store where your own recall points.** This works with every module at its
full phase count (9, 16, 25), so nothing is told how many values any content factor has.

**Replay.** For a stored item i, the recall coefficients are c = e_i - alpha K e_i. With the item's own
trace set aside, the rest, -alpha K e_i, is exactly -alpha/2 times the gradient of J with respect to
moving i. So replaying an item and moving it to the free address its trace-free recall prefers is
exact coordinate descent on J over realized addresses (Besag's ICM). With a temperature it is Gibbs
sampling, i.e. annealing.

What is exact here and what is not:
- the coefficients come from K, the ridge's own posterior precision, maintained by recursive least
  squares during encoding;
- the network's Hebbian snap delivers the same choice for 78-86% of new items in pilots
  (docs/THEORY.md, C1), not all of them;
- by default (`capacity="growing"`) the caps grow with the count so far, so the final number of
  items is never used; `capacity="final"` caps by it instead.
"""

from __future__ import annotations

import numpy as np

from loci.scaffold import Scaffold

CAPACITIES = ("final", "growing")


def _broadcast(values: np.ndarray, module: int, sizes: tuple[int, ...]) -> np.ndarray:
    shape = [1] * len(sizes)
    shape[module] = sizes[module]
    return values.reshape(shape)


def encode(patterns: np.ndarray, scaffold: Scaffold, alpha: float, capacity: str = "growing",
           slack: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(addresses, precision K) for patterns (Ns x P) stored in arrival order, each at the free address
    its own recall prefers. No module phase takes more than ceil(slack * P / size) items, unless
    every free address is capped."""
    if capacity not in CAPACITIES:
        raise ValueError(f"capacity must be one of {CAPACITIES}")
    sizes, count = scaffold.sizes, patterns.shape[1]
    phases = np.zeros((count, len(sizes)), dtype=np.int64)
    taken = np.zeros(sizes, dtype=bool)
    taken[(0,) * len(sizes)] = True
    precision = np.array([[1.0 / (patterns[:, 0] @ patterns[:, 0] + alpha)]])
    for n in range(1, count):
        item = patterns[:, n]
        overlaps = patterns[:, :n].T @ item
        recall = precision @ overlaps
        residual = item @ item + alpha - overlaps @ recall
        total = count if capacity == "final" else n + 1
        score, capped = np.zeros(sizes), np.zeros(sizes)
        for m, size in enumerate(sizes):
            load = np.bincount(phases[:n, m], minlength=size)
            summed = np.bincount(phases[:n, m], weights=recall, minlength=size)
            score = score + _broadcast(summed, m, sizes)
            capped = capped + _broadcast((load >= np.ceil(slack * total / size)).astype(float), m, sizes)
        score[capped > 0] -= 1e6  # soft: a capped phase is used only if nothing uncapped is free
        score[taken] = -np.inf
        best = np.unravel_index(int(score.argmax()), sizes)
        taken[best] = True
        phases[n] = best
        precision = np.block([[precision + np.outer(recall, recall) / residual, -recall[:, None] / residual],
                              [-recall[None, :] / residual, np.array([[1.0 / residual]])]])
    return scaffold.index(phases), precision


def replay(precision: np.ndarray, scaffold: Scaffold, where: np.ndarray, rng: np.random.Generator,
           sweeps: int = 100, temperature: tuple[float, float] | None = None, settle: int = 20,
           slack: float = 1.1) -> np.ndarray:
    """New addresses after replay. Each sweep visits every item in random order and moves it to the
    free address that most lowers J (a Gibbs draw when `temperature` = (start, end), annealed
    geometrically over `sweeps`, then `settle` sweeps at zero). Phase loads are capped at
    ceil(slack * P / size). Stops early once a zero-temperature sweep moves nothing."""
    sizes, count = scaffold.sizes, len(where)
    phases = scaffold.phases[where].copy()
    caps = [int(np.ceil(slack * count / size)) for size in sizes]
    coupling = [precision @ np.eye(size)[phases[:, m]] for m, size in enumerate(sizes)]  # K q_mk
    loads = [np.bincount(phases[:, m], minlength=size) for m, size in enumerate(sizes)]
    taken = np.zeros(sizes, dtype=bool)
    taken[tuple(phases.T)] = True
    diagonal = np.diag(precision)
    schedule = list(np.geomspace(*temperature, sweeps)) if temperature else [0.0] * sweeps
    for heat in schedule + [0.0] * (settle if temperature else 0):
        moved = 0
        for i in rng.permutation(count):
            here = tuple(phases[i])
            change = np.zeros(sizes)
            for m, size in enumerate(sizes):
                step = 2 * (coupling[m][i] - (coupling[m][i, here[m]] - diagonal[i]))
                step[here[m]] = 0.0
                step[(loads[m] >= caps[m]) & (np.arange(size) != here[m])] = np.inf
                change = change + _broadcast(step, m, sizes)
            change[taken] = np.inf
            change[here] = 0.0
            if heat > 0:
                flat = change.ravel()
                finite = np.isfinite(flat)
                weight = np.zeros_like(flat)
                weight[finite] = np.exp(-(flat[finite] - flat[finite].min()) / heat)
                there = np.unravel_index(int(rng.choice(flat.size, p=weight / weight.sum())), sizes)
            else:
                there = np.unravel_index(int(change.argmin()), sizes)
                if not change[there] < -1e-15:
                    there = here
            if there != here:
                taken[here], taken[there] = False, True
                for m in range(len(sizes)):
                    if there[m] != here[m]:
                        coupling[m][:, here[m]] -= precision[:, i]
                        coupling[m][:, there[m]] += precision[:, i]
                        loads[m][here[m]] -= 1
                        loads[m][there[m]] += 1
                phases[i] = there
                moved += 1
        if heat == 0 and moved == 0:
            break
    return scaffold.index(phases)


def typical_change(precision: np.ndarray, scaffold: Scaffold, where: np.ndarray, rng: np.random.Generator,
                   items: int = 50) -> float:
    """The median |Delta J| of moving an item to another free address: the natural temperature at
    which replay starts to discriminate between addresses (a scale for annealed replay)."""
    sizes = scaffold.sizes
    phases = scaffold.phases[where]
    coupling = [precision @ np.eye(size)[phases[:, m]] for m, size in enumerate(sizes)]
    taken = np.zeros(sizes, dtype=bool)
    taken[tuple(phases.T)] = True
    diagonal = np.diag(precision)
    changes = []
    for i in rng.choice(len(where), min(items, len(where)), replace=False):
        here = tuple(phases[i])
        change = np.zeros(sizes)
        for m, size in enumerate(sizes):
            step = 2 * (coupling[m][i] - (coupling[m][i, here[m]] - diagonal[i]))
            step[here[m]] = 0.0
            change = change + _broadcast(step, m, sizes)
        changes.append(np.abs(change[~taken]))
    return float(np.median(np.concatenate(changes)))


def law(precision: np.ndarray, scaffold: Scaffold, where: np.ndarray) -> float:
    """The error law on realized phases: sum over modules and phases of q^T K q."""
    phases = scaffold.phases[where]
    total = 0.0
    for m, size in enumerate(scaffold.sizes):
        onehot = np.eye(size)[phases[:, m]]
        total += float((onehot * (precision @ onehot)).sum())
    return total
