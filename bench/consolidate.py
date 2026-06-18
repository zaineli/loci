"""E4 - a memory that files itself: encoding where its own recall points, then replay.

Arms, all on the paper's scaffold with the same stored patterns and the same noisy cues:
  random          a uniformly random address each (the paper's sequential placement is the same)
  k-means         residual k-means on the patterns, per module (`place.kmeans`)
  oracle          the true factors as the address, where the content has three and they fit
  encode          one pass in arrival order: each item at the free address its own recall prefers
                  (`consolidate.encode`, capacity growing with the count so far: P is never known)
  encode+replay   then replay: each item moves to where its recall, own trace set aside, prefers
  random+replay   the ablation: replay from random addresses, no recall-guided encoding
  oracle+replay   replay starting from the oracle
Every arm is also predicted by the zero-fit theory (`theory.predict`), which never sees a cue.

Content kinds, chosen so the scaffold's modules (9, 16, 25 phases) match none of them except the first:
  factored        3 factors of 9, 16, 5 values (round 1's content)
  cards           3 factors of 7, 12, 4 values
  four            4 factors of 6, 10, 4, 8 values, for 3 modules
  hierarchy       5 classes x 4 subclasses each: no product structure at all
  lifelog         MiniLM-embedded sentences, sign-projected (bench/lifelog.py)

    uv run --extra bench python bench/consolidate.py [--content factored,...] [--seeds 40-49] [--jobs 8]
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

from loci import consolidate, place, theory
from loci.content import factored, hierarchical
from loci.memory import Memory, flip, mmse_alpha
from loci.scaffold import Scaffold

sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = Path(__file__).resolve().parents[1] / "results"
DIM = 1_000
KINDS = ("factored", "cards", "four", "hierarchy", "lifelog")
LOADS = (0.4, 0.8)
FLIPS = (0.1, 0.2)


def _content(kind: str, count: int, rng: np.random.Generator):
    if kind == "factored":
        return factored(DIM, count, rng)
    if kind == "cards":
        return factored(DIM, count, rng, cards=(7, 12, 4))
    if kind == "four":  # detail scaled so an item is still a quarter itself, as with three factors
        return factored(DIM, count, rng, detail=2 / np.sqrt(3), cards=(6, 10, 4, 8))
    if kind == "hierarchy":
        return hierarchical(DIM, count, rng)
    from lifelog import lifelog

    return lifelog(DIM, count, rng)


def _nmi(x: np.ndarray, y: np.ndarray) -> float:
    joint = np.zeros((x.max() + 1, y.max() + 1))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum()
    hx, hy = -(px[px > 0] * np.log(px[px > 0])).sum(), -(py[py > 0] * np.log(py[py > 0])).sum()
    return float(2 * mi / (hx + hy)) if hx + hy > 0 else 0.0


def condition(kind: str, seed: int, load: float) -> list[dict]:
    count = round(load * DIM)
    content = _content(kind, count, np.random.default_rng(10_000 * seed + count))
    stored, factors = content.patterns, content.factors
    scaffold = Scaffold(seed=seed)
    rows = []
    for rate in FLIPS:
        started = time.time()
        alpha = mmse_alpha(count, flip_rate=rate)
        cues = flip(stored, rate, np.random.default_rng(20_000 * seed + count + round(1_000 * rate)))
        encoded, precision = consolidate.encode(stored, scaffold, alpha, capacity="growing")
        arms = {"random": place.scattered(count, scaffold, np.random.default_rng(seed + 1)),
                "k-means": place.kmeans(stored, scaffold, np.random.default_rng(seed + 2))[0],
                "encode": encoded}
        oracle_fits = kind in ("factored", "cards", "lifelog")
        if oracle_fits:
            arms["oracle"] = place.oracle(scaffold, factors)
        replayed = {"encode+replay": "encode", "random+replay": "random"}
        if oracle_fits:
            replayed["oracle+replay"] = "oracle"
        for name, start in replayed.items():
            arms[name] = consolidate.replay(precision, scaffold, arms[start], np.random.default_rng(100 + seed))
        for name, where in arms.items():
            assert len(np.unique(where)) == count, name
            memory = Memory(scaffold, rule="ridge", alpha=alpha)
            memory.store(stored, where)
            got = memory.recall(cues)
            phases = scaffold.phases[where]
            predicted = theory.predict(scaffold, stored, where, alpha, flip_rate=rate,
                                       rng=np.random.default_rng(seed))
            rows.append({
                "content": kind, "seed": seed, "load": load, "P": count, "flip": rate, "placement": name,
                "recall": float(np.mean(got.address == where)),
                "modules": (got.phases == phases).mean(axis=0).tolist(),
                "theory": predicted.address, "theory_modules": predicted.modules.tolist(),
                "law": consolidate.law(precision, scaffold, where),
                "nmi": [[_nmi(phases[:, m], factors[:, f]) for f in range(factors.shape[1])]
                        for m in range(phases.shape[1])],
            })
        print(f"{kind} seed {seed} P {count} flip {rate} ({time.time() - started:.0f}s)", flush=True)
    return rows


def _seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = map(int, text.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", default=",".join(KINDS))
    parser.add_argument("--seeds", default="40-49", help="the registered run uses 40-49; pilots used 0-2")
    parser.add_argument("--loads", default=",".join(map(str, LOADS)))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--out", default="consolidate.json")
    args = parser.parse_args()
    work = [(kind, seed, float(load)) for kind in args.content.split(",") for seed in _seeds(args.seeds)
            for load in args.loads.split(",")]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [row for batch in pool.map(condition, *zip(*work, strict=True)) for row in batch]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(rows))
    print(f"{len(rows)} rows -> {RESULTS / args.out}")


if __name__ == "__main__":
    main()
