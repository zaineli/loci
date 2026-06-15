"""Exploratory, after E2: is the no-product control's win just more weights?

The grid scaffold (50 grid cells) and the no-product control (one module of 3,600 phases) at the
same place cells differ 4.4x in weights, almost all of it W_hg and W_gh. This matches them both
ways: the control shrunk to the grid's weights (91 place cells), and the grid grown to the
control's (1,750). Random placement, factored content, P/Ns = 0.8, clean, 10%- and 20%-flipped
cues, ridge write. Two reads of the same cue-map output: the scaffold's own snap, and the nearest
of all 3,600 place codes by cosine (a raw dot product favours high-norm codes, because a noisy
cue's place activity is dominated by the part every code shares). Not pre-registered.

    uv run python bench/budget.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from loci import place
from loci.content import factored
from loci.memory import CueMaps, flip, mmse_alpha
from loci.scaffold import Scaffold, relu

OUT = Path(__file__).resolve().parents[1] / "results" / "budget.json"
DIM, COUNT, SEEDS = 1_000, 800, 10
CODES = {"grid, 400 place cells": {}, "grid, 1,750 place cells": {"place_cells": 1_750},
         "flat, 400 place cells": {"periods": (60,)}, "flat, 91 place cells": {"periods": (60,), "place_cells": 91}}


def weights(scaffold: Scaffold) -> int:
    return 2 * scaffold.place_cells * (DIM + scaffold.grid_cells)


def main() -> None:
    rows = []
    for seed in range(SEEDS):
        content = factored(DIM, COUNT, np.random.default_rng(10_000 * seed + COUNT))
        maps = CueMaps(content.patterns)
        for name, config in CODES.items():
            scaffold = Scaffold(seed=seed, **config)
            where = place.scattered(COUNT, scaffold, np.random.default_rng(seed + 1))
            codes = scaffold.H / np.linalg.norm(scaffold.H, axis=0, keepdims=True)
            for rate in (0.0, 0.1, 0.2):
                cues = flip(content.patterns, rate, np.random.default_rng(20_000 * seed + int(1_000 * rate)))
                h0 = relu(maps.map(scaffold.H[:, where], mmse_alpha(COUNT, rate)) @ cues)
                phases, _ = scaffold.settle(h0)
                rows.append({"seed": seed, "code": name, "flip": rate, "weights": weights(scaffold),
                             "recovery": float(np.mean(scaffold.index(phases) == where)),
                             "nearest": float(np.mean((codes.T @ h0).argmax(axis=0) == where))})
        print(f"seed {seed}", flush=True)
    OUT.write_text(json.dumps(rows))
    print(f"{'':26} {'weights':>9}  {'clean':>5}  snap 10% / 20%   nearest (cosine) 10% / 20%")
    for name in CODES:
        picked = [r for r in rows if r["code"] == name]
        def mean(key: str, f: float, picked: list[dict] = picked) -> float:
            return float(np.mean([r[key] for r in picked if r["flip"] == f]))
        print(f"{name:26} {picked[0]['weights']:>9,}  {mean('recovery', 0.0):.3f}  "
              f"{mean('recovery', 0.1):.3f} / {mean('recovery', 0.2):.3f}    "
              f"{mean('nearest', 0.1):.3f} / {mean('nearest', 0.2):.3f}")


if __name__ == "__main__":
    main()
