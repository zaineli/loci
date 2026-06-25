"""Imagination and false memory in a scaffold memory: what a cue for something never experienced
recalls, and what that costs.

Every phase tuple of the grid scaffold is a valid state, stored or not. That is the paper's
zero-shot generalisation. If memories sit on phases that carry the content's factors (aligned
placement), an empty address whose phases were never stored *together* holds a combination of
factors never experienced together, and a recall can land on it. Four probes measure what follows:
  studied      each stored item's own cue                         -> recall
  gist         the factors of a never-experienced combination     -> construction
  recombined   a new event made of those factors, with new detail  -> false recall
  unrelated    an event that shares nothing with memory            -> recognition baseline

Decoders, all from the same place activity h0 = ReLU(W_hs cue):
  snap         the paper's per-module argmax: any of the 3,600 tuples
  nearest      the place code, of all 3,600, closest to h0 by cosine: any tuple is valid, and a
               stored one wins only by being nearer. It needs one unit per address; the paper's
               scaffold does not have that decoder natively
  stored       the stored place code closest to h0 by cosine: never-stored states are not valid

Measured per decoder:
  recall         a studied cue lands on its own address
  construction   a gist cue lands on an EMPTY address whose read-out decodes every factor of it
  false recall   a recombined event lands on an EMPTY address whose read-out decodes its factors
  familiarity    d' of cos(h0, the decoded place code) between studied cues and each lure type
  recollection   d' of the overlap between the cue and what is read out (recall-to-reject)

What it is not: a generative model. Nothing here is trained to imagine. The construct is whatever
the unmodified heteroassociation reads out at an empty address.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from loci.memory import CueMaps, flip
from loci.scaffold import Scaffold, relu

DECODERS = ("snap", "nearest", "stored")


@dataclass(frozen=True)
class Probes:
    """Clean probe patterns (Ns, n) and, for gist and recombined, the factor values they carry."""

    gist: np.ndarray
    gist_factors: np.ndarray  # (n, 3)
    recombined: np.ndarray
    recombined_factors: np.ndarray  # (n, 3)
    unrelated: np.ndarray


def never_experienced(held: frozenset[tuple[int, int]], cards: tuple[int, ...]) -> np.ndarray:
    """Every (a, b, c) with (a, b) held out: the combinations the memory has never experienced."""
    return np.array([(a, b, c) for a, b in sorted(held) for c in range(cards[2])])


def decode(scaffold: Scaffold, where: np.ndarray, h0: np.ndarray, decoder: str) -> np.ndarray:
    """The address each column of h0 is decoded to."""
    if decoder == "snap":
        return scaffold.index(scaffold.snap(scaffold.W_gh @ h0))
    unit = h0 / (np.linalg.norm(h0, axis=0, keepdims=True) + 1e-12)
    codes = scaffold.H / np.linalg.norm(scaffold.H, axis=0, keepdims=True)
    if decoder == "nearest":
        return (codes.T @ unit).argmax(axis=0)
    if decoder == "stored":
        return where[(codes[:, where].T @ unit).argmax(axis=0)]
    raise ValueError(f"decoder must be one of {DECODERS}")


def dprime(studied: np.ndarray, lure: np.ndarray) -> float:
    return float((studied.mean() - lure.mean()) / np.sqrt(0.5 * (studied.var() + lure.var()) + 1e-12))


def measure(scaffold: Scaffold, patterns: np.ndarray, where: np.ndarray, alpha: float, probes: Probes,
            factor_of: Callable[[np.ndarray, int], np.ndarray], flip_rate: float,
            rng: np.random.Generator, decoders: tuple[str, ...] = DECODERS,
            readout_ridge: float = 0.0) -> dict[str, dict[str, float]]:
    """Recall, construction, false recall, familiarity and recollection d', per decoder. Every probe
    carries `flip_rate` of flipped bits. `factor_of(readouts, f)` decodes factor f from read-outs.
    `readout_ridge` > 0 regularises W_sh by that fraction of H_a H_a^T's mean eigenvalue."""
    codes = scaffold.H[:, where]
    w_hs = CueMaps(patterns).map(codes, alpha)
    if readout_ridge > 0:
        gram = codes @ codes.T
        w_sh = patterns @ codes.T @ np.linalg.inv(gram + readout_ridge * np.trace(gram) / len(gram) * np.eye(len(gram)))
    else:
        w_sh = patterns @ np.linalg.pinv(codes)
    stored = np.zeros(scaffold.addresses, dtype=bool)
    stored[where] = True
    unit_codes = scaffold.H / np.linalg.norm(scaffold.H, axis=0, keepdims=True)
    cues = {name: flip(clean, flip_rate, rng) for name, clean in
            (("studied", patterns), ("gist", probes.gist), ("recombined", probes.recombined),
             ("unrelated", probes.unrelated))}
    out = {}
    for decoder in decoders:
        row, familiarity, recollection = {}, {}, {}
        for name, cue in cues.items():
            h0 = relu(w_hs @ cue)
            got = decode(scaffold, where, h0, decoder)
            readout = np.sign(w_sh @ scaffold.H[:, got])
            familiarity[name] = (unit_codes[:, got] * h0).sum(axis=0) / (np.linalg.norm(h0, axis=0) + 1e-12)
            recollection[name] = (cue * readout).mean(axis=0)
            if name == "studied":
                row["recall"] = float(np.mean(got == where))
            elif name in ("gist", "recombined"):
                truth = probes.gist_factors if name == "gist" else probes.recombined_factors
                every = np.all(np.stack([factor_of(readout, f) for f in range(truth.shape[1])], axis=1) == truth,
                               axis=1)
                row["construction" if name == "gist" else "false recall"] = float(np.mean(~stored[got] & every))
        for lure in ("recombined", "unrelated"):
            row[f"familiarity d' {lure}"] = dprime(familiarity["studied"], familiarity[lure])
            row[f"recollection d' {lure}"] = dprime(recollection["studied"], recollection[lure])
        out[decoder] = row
    return out
