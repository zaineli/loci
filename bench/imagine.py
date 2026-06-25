"""E5 - the price of imagination: recall, construction, false recall and recognition.

Content with never-experienced combinations: for every value b of the second factor, one partner a
of the first is held out, so (a, b) never occurs together in memory (`content.holdout`). Probes, all
with the condition's flipped bits (`loci.imagine`):
  gist         the factors of a held-out combination alone
  recombined   a new event of a held-out combination, with new detail (4 per combination)
  unrelated    an event unrelated to memory

Placements: random; oracle (the true factors as the address); encode+replay (the memory files itself
for recall, `loci.consolidate`, greedy replay); encode+anneal (the same, with annealed replay: noisy
moves cooling over 120 sweeps); oracle+replay (an aligned memory, then consolidated for recall).
Decoders: snap (the paper's), nearest (any tuple valid), stored (only stored states valid). kNN over
the stored patterns is the exemplar reference.

    uv run --extra bench python bench/imagine.py [--content factored,cards,lifelog] [--seeds 40-49]
    uv run --extra bench python bench/imagine.py --nh [--seeds 40-49]     # the read-out's P = Nh peak
"""

from __future__ import annotations

import os

for _threads in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from loci import consolidate, imagine, place
from loci.content import factored, holdout
from loci.memory import flip, mmse_alpha
from loci.scaffold import Scaffold

sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = Path(__file__).resolve().parents[1] / "results"
DIM, COUNT, LURES = 1_000, 800, 4
FLIPS = (0.1, 0.2)
PLACEMENTS = ("random", "oracle", "encode+replay", "encode+anneal", "oracle+replay")
NH = (400, 600, 700, 800, 900, 1_200)


def _content(kind: str, seed: int):
    """(content, probes, cards) with held-out combinations."""
    rng = np.random.default_rng(31_000 * seed + COUNT)
    if kind == "lifelog":
        import lifelog

        cards = lifelog.CARDS
        held = holdout(cards, rng)
        content = lifelog.lifelog(DIM, COUNT, rng, exclude=held)
        queries = imagine.never_experienced(held, cards)
        lure_factors = np.repeat(queries, LURES, axis=0)
        probes = imagine.Probes(lifelog.gist(content, queries), queries,
                                lifelog.recombinations(content, lure_factors, rng), lure_factors,
                                lifelog.unrelated(content, len(lure_factors), rng))
        return content, probes, cards
    cards = (7, 12, 4) if kind == "cards" else (9, 16, 5)
    held = holdout(cards, rng)
    content = factored(DIM, COUNT, rng, cards=cards, exclude=held)
    queries = imagine.never_experienced(held, cards)
    lure_factors = np.repeat(queries, LURES, axis=0)
    parts = sum(code[:, lure_factors[:, f]] for f, code in enumerate(content.codes))
    probes = imagine.Probes(np.sign(sum(code[:, queries[:, f]] for f, code in enumerate(content.codes))),
                            queries, np.sign(parts + rng.standard_normal(parts.shape)), lure_factors,
                            np.sign(rng.standard_normal((DIM, len(lure_factors)))))
    return content, probes, cards


def condition(kind: str, seed: int) -> list[dict]:
    started = time.time()
    content, probes, _ = _content(kind, seed)
    stored, factors = content.patterns, content.factors
    scaffold = Scaffold(seed=seed)
    alpha_place = mmse_alpha(COUNT, flip_rate=0.1)  # a placement is a property of the store, made once
    encoded, precision = consolidate.encode(stored, scaffold, alpha_place, capacity="growing")
    arms = {"random": place.scattered(COUNT, scaffold, np.random.default_rng(seed + 1)),
            "oracle": place.oracle(scaffold, factors),
            "encode+replay": consolidate.replay(precision, scaffold, encoded, np.random.default_rng(100 + seed))}
    heat = consolidate.typical_change(precision, scaffold, encoded, np.random.default_rng(seed))
    arms["encode+anneal"] = consolidate.replay(precision, scaffold, encoded, np.random.default_rng(200 + seed),
                                               sweeps=120, temperature=(heat, heat / 300))
    arms["oracle+replay"] = consolidate.replay(precision, scaffold, arms["oracle"], np.random.default_rng(100 + seed))
    rows = []
    for rate in FLIPS:
        alpha = mmse_alpha(COUNT, flip_rate=rate)
        for name in PLACEMENTS:
            where = arms[name]
            result = imagine.measure(scaffold, stored, where, alpha, probes, content.decode, rate,
                                     np.random.default_rng(1_000 * seed + round(100 * rate)))
            phases = scaffold.phases[where]
            nmi = [_nmi(phases[:, m] // (place.SLOTS if m == 2 else 1), factors[:, m]) for m in range(3)]
            for decoder, row in result.items():
                rows.append({"content": kind, "seed": seed, "flip": rate, "placement": name,
                             "decoder": decoder, "nmi": nmi, **row})
        rows.append({"content": kind, "seed": seed, "flip": rate, "placement": "knn", "decoder": "knn",
                     **_knn(stored, probes, rate, np.random.default_rng(1_000 * seed + round(100 * rate)))})
    print(f"{kind} seed {seed} ({time.time() - started:.0f}s)", flush=True)
    return rows


def _knn(stored: np.ndarray, probes: imagine.Probes, rate: float, rng: np.random.Generator) -> dict:
    """The exemplar reference: the nearest stored pattern, familiarity = its overlap with the cue."""
    best = {}
    for name, clean in (("studied", stored), ("gist", probes.gist), ("recombined", probes.recombined),
                        ("unrelated", probes.unrelated)):
        overlaps = stored.T @ flip(clean, rate, rng) / len(stored)
        best[name] = overlaps
    row = {"recall": float(np.mean(best["studied"].argmax(axis=0) == np.arange(stored.shape[1]))),
           "construction": 0.0, "false recall": 0.0}
    for lure in ("recombined", "unrelated"):
        row[f"familiarity d' {lure}"] = imagine.dprime(best["studied"].max(axis=0), best[lure].max(axis=0))
    return row


def nh_condition(seed: int) -> list[dict]:
    """Construction against the number of place cells at P = 800, oracle placement: the read-out
    W_sh = S H_a^+ interpolates exactly at P = Nh. The pseudo-inverse against a 1% ridge."""
    content, probes, _ = _content("factored", seed)
    rows = []
    for cells in NH:
        scaffold = Scaffold(seed=seed, place_cells=cells)
        where = place.oracle(scaffold, content.factors)
        for ridge in (0.0, 0.01):
            result = imagine.measure(scaffold, content.patterns, where, mmse_alpha(COUNT, flip_rate=0.1), probes,
                                     content.decode, 0.1, np.random.default_rng(1_000 * seed + 10),
                                     decoders=("snap", "nearest"), readout_ridge=ridge)
            for decoder, row in result.items():
                rows.append({"seed": seed, "Nh": cells, "readout_ridge": ridge, "decoder": decoder, **row})
    print(f"nh seed {seed}", flush=True)
    return rows


def _nmi(x: np.ndarray, y: np.ndarray) -> float:
    joint = np.zeros((x.max() + 1, y.max() + 1))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum()
    hx, hy = -(px[px > 0] * np.log(px[px > 0])).sum(), -(py[py > 0] * np.log(py[py > 0])).sum()
    return float(2 * mi / (hx + hy)) if hx + hy > 0 else 0.0


def _seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = map(int, text.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", default="factored,cards,lifelog")
    parser.add_argument("--seeds", default="40-49", help="the registered run uses 40-49; pilots used 0-2")
    parser.add_argument("--nh", action="store_true", help="the read-out peak sweep instead")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    seeds = _seeds(args.seeds)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if args.nh:
            rows = [row for batch in pool.map(nh_condition, seeds) for row in batch]
        else:
            work = [(kind, seed) for kind in args.content.split(",") for seed in seeds]
            rows = [row for batch in pool.map(condition, *zip(*work, strict=True)) for row in batch]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (args.out or ("imagine-nh.json" if args.nh else "imagine.json"))
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
