"""Content bound to scaffold addresses, and recall from a cue. Vector-HaSH's "what".

Storing P patterns S (Ns x P, +/-1) at addresses a_1..a_P writes two heteroassociations: sensory to
place, W_hs, and place to sensory, W_sh = S H_a^+. Recall from a cue s~: h = ReLU(W_hs s~), the
scaffold snaps it to an address, and s = sign(W_sh h). Past P = Nh the read-out degrades smoothly,
m = erf(sqrt(Nh / (2 (P - Nh)))) - the paper's "no memory cliff", for clean cues.

The write rule for W_hs is where a noisy cue is won or lost, and it is a parameter here:
  pinv    H_a S^+, the paper's. Its noise gain grows as P / (Ns - P - 1), so near Ns a 10%-flipped
          cue lands on a valid but wrong address and nothing downstream corrects it.
  ridge   H_a (S^T S + alpha I)^-1 S^T, the MMSE version at alpha ~ 4 f (1 - f) P for flip rate f.
  hebb    H_a S^T / Ns: no noise gain, but crosstalk even for clean cues.
What it is not: an address chooser. Where each pattern goes is `addresses`, given by the caller -
the paper's fixed path, a random draw, or a learned placement (`place.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from loci.scaffold import Scaffold, relu

RULES = ("pinv", "ridge", "hebb")


def mmse_alpha(stored: int, flip_rate: float = 0.0, mask_rate: float = 0.0) -> float:
    """The ridge strength that makes the cue map the MMSE estimate for this kind of cue.

    A cue is a s + xi per bit: flipped with probability f, a = 1 - 2f and Var(xi) = 4 f (1 - f);
    masked (set to 0) with probability r, a = 1 - r and Var(xi) = r (1 - r). The least-squares map
    from such cues to place codes regularises by Var(xi) / a^2 per stored item. Zero for a clean
    cue, where it is the paper's pseudo-inverse.
    """
    a, noise = 1.0, 0.0
    if flip_rate > 0:
        a, noise = a * (1 - 2 * flip_rate), noise + 4 * flip_rate * (1 - flip_rate)
    if mask_rate > 0:
        noise = a * a * mask_rate * (1 - mask_rate) + (1 - mask_rate) * noise
        a *= 1 - mask_rate
    return noise * stored / (a * a) if noise else 0.0


class CueMaps:
    """Every ridge strength's cue map from one SVD of the stored patterns.

    (S^T S + alpha I)^-1 S^T = V diag(sigma / (sigma^2 + alpha)) U^T, so a sweep over alphas - the
    pseudo-inverse is alpha = 0 - costs one decomposition, not one solve per alpha.
    """

    def __init__(self, patterns: np.ndarray) -> None:
        self.U, self.sigma, Vt = np.linalg.svd(patterns, full_matrices=False)
        self.V = Vt.T

    def map(self, place: np.ndarray, alpha: float) -> np.ndarray:
        """W_hs = H_a (S^T S + alpha I)^-1 S^T, for place codes H_a (Nh x P)."""
        keep = self.sigma > 1e-10 * self.sigma.max()
        gain = np.where(keep, self.sigma / (self.sigma**2 + alpha), 0.0)
        return (place @ self.V) * gain @ self.U.T


def theory_overlap(place_cells: int, stored: int) -> float:
    """Clean-cue overlap past the knee: erf(sqrt(Nh / (2 (P - Nh)))), and 1 at or below it."""
    if stored <= place_cells:
        return 1.0
    return erf(sqrt(place_cells / (2 * (stored - place_cells))))


def flip(patterns: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Each +/-1 entry flipped independently with probability `rate` - the paper's cue noise."""
    if rate <= 0:
        return patterns.copy()
    return patterns * np.where(rng.random(patterns.shape) < rate, -1.0, 1.0)


@dataclass
class Recall:
    """What one recall pass returned for each cue."""

    phases: np.ndarray  # (n, M) the address the scaffold settled on
    address: np.ndarray  # (n,) its index
    content: np.ndarray  # (Ns, n) sign read-out


class Memory:
    """Patterns bound to addresses on one scaffold."""

    def __init__(self, scaffold: Scaffold, rule: str = "pinv", alpha: float = 0.0) -> None:
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        self.scaffold, self.rule, self.alpha = scaffold, rule, alpha
        self.W_hs: np.ndarray | None = None
        self.W_sh: np.ndarray | None = None
        self.addresses = np.zeros(0, dtype=np.int64)

    def store(self, patterns: np.ndarray, addresses: np.ndarray) -> None:
        """Bind column j of `patterns` (Ns x P) to address `addresses[j]`. Offline, like the paper:
        a new call replaces what was stored."""
        H = self.scaffold.H[:, addresses]
        self.addresses = np.asarray(addresses)
        self.W_sh = patterns @ np.linalg.pinv(H)
        if self.rule == "pinv":
            self.W_hs = H @ np.linalg.pinv(patterns)
        elif self.rule == "ridge":
            gram = patterns.T @ patterns + self.alpha * np.eye(patterns.shape[1])
            self.W_hs = H @ np.linalg.solve(gram, patterns.T)
        else:
            self.W_hs = (H @ patterns.T) / patterns.shape[0]

    def recall(self, cues: np.ndarray, steps: int = 1) -> Recall:
        """Cues (Ns x n) -> the address each settles on and the content read out there."""
        assert self.W_hs is not None and self.W_sh is not None, "store before recall"
        phases, h = self.scaffold.settle(relu(self.W_hs @ cues), steps)
        return Recall(phases, self.scaffold.index(phases), np.sign(self.W_sh @ h))


def overlap(recalled: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-item overlap m = 1 - 2 * bit error rate (1 = exact, 0 = chance)."""
    return (recalled * true).mean(axis=0)
