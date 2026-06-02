"""The scaffold: a content-free space of addresses that corrects its own errors.

Vector-HaSH's "where" (Chandra, Sharma, Chaudhuri & Fiete, Nature 2025). M grid modules, module m a
one-hot code over l_m = lambda_m^2 cells; an address is one phase per module, so there are
prod(l_m) addresses from sum(l_m) grid cells (the Chinese remainder theorem makes every phase tuple
reachable when the l_m are coprime). A fixed random projection turns an address into a sparse
place code h = ReLU(W_hg g - theta); a Hebbian map W_gh back, and a per-module winner-take-all,
make every address an exact fixed point with a maximal basin.

What it is not: content. Nothing here depends on what is stored - `memory.py` binds content to
these addresses. Two addresses are "near" exactly when they share module phases: the place-code
correlation is set by that count alone (0.00 / 0.26 / 0.58 / 1.00 for 0-3 shared modules at the
defaults), and spatially adjacent positions share none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import numpy as np

# The paper's item-memory configuration (Fig 3): periods 3, 4, 5 -> 50 grid cells, 3,600 addresses.
PERIODS = (3, 4, 5)
PLACE_CELLS = 400
THRESHOLD = 0.5
# Nominal grid->place connectivity. The upstream mask zeroes indices drawn *with* replacement, so
# the effective connectivity is 0.67, not 0.6 (measured 0.6699); `faithful=True` reproduces that.
CONNECTIVITY = 0.6


def relu(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.maximum(x - threshold, 0.0)


@dataclass
class Scaffold:
    """Grid modules, the place code of every address, and the clean-up that snaps to one."""

    periods: tuple[int, ...] = PERIODS
    place_cells: int = PLACE_CELLS
    threshold: float = THRESHOLD
    connectivity: float = CONNECTIVITY
    faithful: bool = True
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.sizes = tuple(p * p for p in self.periods)
        self.offsets = tuple(np.cumsum((0, *self.sizes[:-1])).tolist())
        self.grid_cells = sum(self.sizes)
        self.addresses = prod(self.sizes)
        self.phases = self._phases(np.arange(self.addresses))
        self.G = self.grid(self.phases)  # (Ng, Npos)
        self.W_hg = self._projection()  # (Nh, Ng), fixed
        self.H = relu(self.W_hg @ self.G, self.threshold)  # (Nh, Npos)
        # Hebbian over every address; the 1/Npos scale is irrelevant under winner-take-all.
        self.W_gh = (self.G @ self.H.T) / self.addresses  # (Ng, Nh)

    # ---- addresses ----------------------------------------------------------------------------

    def _phases(self, x: np.ndarray) -> np.ndarray:
        """Address index -> one phase per module (x mod l_m), shape (n, M)."""
        return np.stack([np.mod(x, size) for size in self.sizes], axis=1)

    def index(self, phases: np.ndarray) -> np.ndarray:
        """Phase tuples -> address index, by the Chinese remainder theorem (brute-force table)."""
        table = getattr(self, "_table", None)
        if table is None:
            table = np.full(self.sizes, -1, dtype=np.int64)
            table[tuple(self.phases.T)] = np.arange(self.addresses)
            self._table = table
        return table[tuple(np.asarray(phases).T)]

    def grid(self, phases: np.ndarray) -> np.ndarray:
        """Phase tuples (n, M) -> grid vectors (Ng, n), one-hot per module."""
        phases = np.atleast_2d(phases)
        g = np.zeros((self.grid_cells, len(phases)))
        for m, offset in enumerate(self.offsets):
            g[offset + phases[:, m], np.arange(len(phases))] = 1.0
        return g

    def shared(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """How many module phases two addresses share: the scaffold's only notion of nearness."""
        return (self.phases[np.asarray(a)] == self.phases[np.asarray(b)]).sum(axis=-1)

    # ---- dynamics -----------------------------------------------------------------------------

    def _projection(self) -> np.ndarray:
        weights = self.rng.standard_normal((self.place_cells, self.grid_cells))
        if self.faithful:
            cut = int((1 - self.connectivity) * self.place_cells * self.grid_cells)
            rows = self.rng.integers(0, self.place_cells, size=cut)
            cols = self.rng.integers(0, self.grid_cells, size=cut)
            mask = np.ones_like(weights)
            mask[rows, cols] = 0.0
        else:
            mask = (self.rng.random(weights.shape) < self.connectivity).astype(float)
        return weights * mask

    def snap(self, grid_input: np.ndarray) -> np.ndarray:
        """Per-module winner-take-all: raw grid input (Ng, n) -> phase tuples (n, M)."""
        return np.stack(
            [
                grid_input[offset : offset + size].argmax(axis=0)
                for offset, size in zip(self.offsets, self.sizes, strict=True)
            ],
            axis=1,
        )

    def settle(self, h: np.ndarray, steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Place activity (Nh, n) -> (phase tuples, clean place code) after `steps` loops."""
        phases = self.snap(self.W_gh @ h)
        for _ in range(steps - 1):
            phases = self.snap(self.W_gh @ self.place(phases))
        return phases, self.place(phases)

    def place(self, phases: np.ndarray) -> np.ndarray:
        return relu(self.W_hg @ self.grid(phases), self.threshold)
