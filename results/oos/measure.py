"""Step 4: measure every predicted condition with the repo's own machinery. Never runs the theory.

Refuses to run unless predictions.json still has the hash recorded in PREDICTIONS.txt. Recall is the
bench's own read (bench/placement.py `_reads`): cue map from loci.memory.CueMaps at the condition's
alpha, ReLU, the scaffold's settle, compared with the predicted placement's addresses. Each
condition is measured on DRAWS independent cue draws; the prediction is an expectation over cues,
so the mean over draws is the estimate it is compared with.
"""
import grid  # noqa: F401
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loci.memory import CueMaps
from loci.scaffold import relu

HERE = Path(__file__).resolve().parent
DRAWS = 3


def _check_hash():
    recorded = [ln.split()[0] for ln in (HERE / "PREDICTIONS.txt").read_text().splitlines()
                if ln.strip().endswith("predictions.json")]
    actual = hashlib.sha256((HERE / "predictions.json").read_bytes()).hexdigest()
    assert recorded and recorded[0] == actual, "predictions.json changed after it was recorded"


def uid(family, load, seed, placement) -> str:
    """Unique row key. The recorded `id` omits the load, so the three "load" settings share ids."""
    return f"{family}|load {load}|seed {seed}|{placement}"


def task(setting_index: int, seed: int):
    s = grid.SETTINGS[setting_index]
    started = time.time()
    patterns, factors, scaffold, alpha_read, alpha_place = grid.build(s, seed)
    stored = np.load(HERE / "predictions_where.npz")
    # Placements are rebuilt deterministically (the saved arrays of two of the three "load" settings
    # were overwritten by the id collision); every rebuilt array that has an uncorrupted saved copy
    # must equal it exactly.
    arms = grid.placements(s, seed, patterns, factors, scaffold, alpha_place)
    checked = 0
    for name, where in arms.items():
        key = grid.row_id(s, seed, name).replace("|", "__")
        if key in stored and len(stored[key]) == len(where) and s.family != "load" or (s.family == "load" and s.load == 0.7):
            assert np.array_equal(stored[key], where), f"placement not reproduced: {key}"
            checked += 1
    maps = CueMaps(patterns)
    draws = [grid.cues(s, patterns, seed, d) for d in range(DRAWS)]
    rows = []
    for name, where in arms.items():
        W_hs = maps.map(scaffold.H[:, where], alpha_read)
        address, modules = [], []
        for cue in draws:
            phases, _ = scaffold.settle(relu(W_hs @ cue))
            got = scaffold.index(phases)
            address.append(float(np.mean(got == where)))
            modules.append((scaffold.phases[got] == scaffold.phases[where]).mean(axis=0).tolist())
        rows.append({"uid": uid(s.family, s.load, seed, name), "address": float(np.mean(address)),
                     "address_draws": address, "modules": np.mean(modules, axis=0).tolist()})
    print(f"{s.family:16} load {s.load} seed {seed}  placements checked against saved {checked}/{len(arms)}"
          f"  ({time.time() - started:.0f}s)", flush=True)
    return rows


def main():
    _check_hash()
    heavy = [i for i, s in enumerate(grid.SETTINGS) if len(s.periods) > 3]
    light = [(i, seed) for i in range(len(grid.SETTINGS)) if i not in heavy for seed in grid.SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r in pool.map(task, *zip(*light)):
            rows += r
    for i in heavy:
        for seed in grid.SEEDS:
            rows += task(i, seed)
    (HERE / "measured.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} measurements written")


if __name__ == "__main__":
    main()
