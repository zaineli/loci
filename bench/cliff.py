"""E1 - the cliff in the cue: where noisy-cue recall breaks, and whether the write rule is why.

The paper's "no memory cliff" is measured with clean cues. This sweeps what the paper did not:
cues with bits flipped (0-33%) or masked (25-50% unknown), past the point where the number of
stored items P reaches the cue dimension Ns, for three ways of writing the cue->address map:
  pinv   the paper's H_a S^+
  mmse   ridge at the strength `memory.mmse_alpha` matches to the cue's flip or mask rate
  hebb   H_a S^T / Ns
and, as a reference that is not a Vector-HaSH, a matched filter over the stored items
(argmax_mu s_mu . s~), which costs P x Ns synapses instead of Nh x Ns.

Placement is the paper's: item j at address j. Recorded per (Ns, P, cue, rule, seed): address
recovery (the scaffold settled on the right address) and content overlap (1 - 2 x bit error).

    uv run python bench/cliff.py [--seeds 10] [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from loci.memory import CueMaps, flip, mmse_alpha, overlap
from loci.scaffold import Scaffold, relu

OUT = Path(__file__).resolve().parents[1] / "results" / "cliff.json"
# Ns small enough that P can run to 3 Ns inside the 3,600 addresses, and the paper's own 3,600.
DIMS = (500, 1_000, 3_600)
RATIOS = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)
FLIPS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.33)
MASKS = (0.25, 0.5)
# Items recalled per condition: all of them up to this many, a fixed random subset beyond.
PROBE = 1_000


def mask(patterns: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """A partial cue: each entry unknown (0) with probability `rate`."""
    return patterns * (rng.random(patterns.shape) >= rate)


def _cues(patterns: np.ndarray, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, dict]]:
    """cue name -> (cues, the cue statistics the MMSE write is tuned for)."""
    out = {f"flip{f}": (flip(patterns, f, rng), {"flip_rate": f}) for f in FLIPS}
    out |= {f"mask{m}": (mask(patterns, m, rng), {"mask_rate": m}) for m in MASKS}
    return out


def _recall(scaffold: Scaffold, W_hs, W_sh, cues, truth, targets) -> tuple[float, float]:
    phases, h = scaffold.settle(relu(W_hs @ cues))
    got = scaffold.index(phases)
    return float((got == targets).mean()), float(overlap(np.sign(W_sh @ h), truth).mean())


def _matched(stored: np.ndarray, cues: np.ndarray, targets: np.ndarray) -> float:
    return float(((stored.T @ cues).argmax(axis=0) == targets).mean())


def condition(scaffold: Scaffold, patterns: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Every cue and rule for one stored set (item j at address j)."""
    count = patterns.shape[1]
    place = scaffold.H[:, :count]
    maps, W_sh = CueMaps(patterns), patterns @ np.linalg.pinv(place)
    probe = np.arange(count) if count <= PROBE else rng.choice(count, PROBE, replace=False)
    truth = patterns[:, probe]
    rows = []
    for name, (cues, stats) in _cues(truth, rng).items():
        rules = {
            "pinv": maps.map(place, 0.0),
            "mmse": maps.map(place, mmse_alpha(count, **stats)),
            "hebb": place @ patterns.T / patterns.shape[0],
        }
        for rule, W_hs in rules.items():
            recovered, content = _recall(scaffold, W_hs, W_sh, cues, truth, probe)
            rows.append({"cue": name, "rule": rule, "recovery": recovered, "overlap": content})
        rows.append({"cue": name, "rule": "matched", "recovery": _matched(patterns, cues, probe)})
    return rows


def run(seeds: int, dims: tuple[int, ...], ratios: tuple[float, ...]) -> list[dict]:
    rows: list[dict] = []
    for seed in range(seeds):
        scaffold = Scaffold(seed=seed)
        for dim in dims:
            rng = np.random.default_rng(1_000 * seed + dim)
            most = min(scaffold.addresses, int(max(ratios) * dim))
            patterns = np.sign(rng.standard_normal((dim, most)))
            for ratio in ratios:
                count = round(ratio * dim)
                if count > scaffold.addresses:
                    continue
                started = time.time()
                for row in condition(scaffold, patterns[:, :count], rng):
                    rows.append({"seed": seed, "Ns": dim, "P": count, **row})
                print(f"seed {seed} Ns {dim} P {count} ({time.time() - started:.1f}s)", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="2 seeds, Ns 500 and 1,000 only")
    args = parser.parse_args()
    dims = DIMS[:2] if args.quick else DIMS
    rows = run(2 if args.quick else args.seeds, dims, RATIOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_name("cliff-quick.json" if args.quick else OUT.name).write_text(json.dumps(rows))
    print(f"{len(rows)} rows -> {OUT.parent}")


if __name__ == "__main__":
    main()
