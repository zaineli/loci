"""Content with known structure, for measuring what a placement keeps.

Each item is sign(A[a] + B[b] + C[c] + w z): three factors - think person, project, activity -
with 9, 16 and 5 values, plus detail z of its own, so two items are as alike as the factors they
share. The factor codes and each item's detail are kept, because E2 decodes factors back out of a
recall (did the gist survive?) and builds cues that leave a factor out (can memory fill it in?).

What it is not: text. `bench/placement.py` also runs a life-log of embedded sentences; this is the
version where the ground truth is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CARDS = (9, 16, 5)


@dataclass(frozen=True)
class Factored:
    patterns: np.ndarray  # (Ns, P) +/-1
    factors: np.ndarray  # (P, 3) the value of each factor
    codes: tuple[np.ndarray, ...]  # per factor, (Ns, card) Gaussian codes
    detail: np.ndarray  # (Ns, P) each item's own part, already weighted

    def cue(self, without: int) -> np.ndarray:
        """Every item's cue with one factor left out entirely: the rest of what it was, signed."""
        parts = sum(c[:, self.factors[:, i]] for i, c in enumerate(self.codes) if i != without)
        return np.sign(parts + self.detail)

    def decode(self, recalled: np.ndarray, factor: int) -> np.ndarray:
        """Which value of `factor` each recalled pattern is closest to."""
        return (self.codes[factor].T @ recalled).argmax(axis=0)


def factored(dim: int, count: int, rng: np.random.Generator, detail: float = 1.0,
             cards: tuple[int, ...] = CARDS) -> Factored:
    """`count` items in `dim` bits; `detail` is the weight of each item's own part against the
    unit-variance factors (1.0 with three factors: an item is a quarter itself, three quarters its
    factors). `cards` sets how many factors there are and how many values each takes; the default
    matches the scaffold's module group counts, other values un-rig that match."""
    codes = tuple(rng.standard_normal((dim, card)) for card in cards)
    factors = np.stack([rng.integers(0, card, count) for card in cards], axis=1)
    own = detail * rng.standard_normal((dim, count))
    signal = sum(c[:, factors[:, i]] for i, c in enumerate(codes))
    return Factored(np.sign(signal + own), factors, codes, own)


def hierarchical(dim: int, count: int, rng: np.random.Generator, branches: tuple[int, int] = (5, 4),
                 detail: float = 1.0) -> Factored:
    """Content with no product structure: `branches[0]` classes, each with its own `branches[1]`
    subclasses, s = sign(class + subclass + detail). A subclass code belongs to one class only, so
    no pair of independent factors exists for a product scaffold to align with. `factors` holds
    (class, subclass), the subclass numbered globally (0 .. classes x subclasses - 1), so `cue` and
    `decode` work as for factored content."""
    top, sub = branches
    classes = rng.standard_normal((dim, top))
    subclasses = rng.standard_normal((dim, top * sub))
    kind = rng.integers(0, top, count)
    factors = np.stack([kind, kind * sub + rng.integers(0, sub, count)], axis=1)
    own = detail * rng.standard_normal((dim, count))
    return Factored(np.sign(classes[:, factors[:, 0]] + subclasses[:, factors[:, 1]] + own), factors,
                    (classes, subclasses), own)
