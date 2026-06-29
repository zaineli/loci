"""Step 3: predict every out-of-sample condition with the frozen theory. Never runs a recall.

Writes predictions.json (one row per condition) and predictions_where.npz (each condition's
placement, so measure.py uses exactly the addresses that were predicted).
"""
import grid  # noqa: F401  (sets thread limits and paths first)
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frozen"))


def task(setting_index: int, seed: int):
    import t4_theory  # the frozen theory
    s = grid.SETTINGS[setting_index]
    started = time.time()
    patterns, factors, scaffold, alpha_read, alpha_place = grid.build(s, seed)
    arms = grid.placements(s, seed, patterns, factors, scaffold, alpha_place)
    rows, wheres = [], {}
    for name, where in arms.items():
        assert len(np.unique(where)) == len(where)
        out = t4_theory.theory(scaffold, patterns, where, alpha_read, grid.theory_rate(s),
                               np.random.default_rng(seed), draws=400, levels=("E", "T-relu"))
        rid = grid.row_id(s, seed, name)
        rows.append({"id": rid, "family": s.family, "setting": dataclasses.asdict(s), "seed": seed,
                     "placement": name, "P": int(patterns.shape[1]), "alpha_read": alpha_read,
                     "theory_rate": grid.theory_rate(s),
                     **{level: {"address": float(v[1]), "modules": [float(x) for x in v[0]]}
                        for level, v in out.items()}})
        wheres[rid] = where
    print(f"{s.family:16} seed {seed}  {len(rows)} rows  ({time.time() - started:.0f}s)", flush=True)
    return rows, wheres


def main():
    heavy = [i for i, s in enumerate(grid.SETTINGS) if len(s.periods) > 3]   # 176,400 addresses
    light = [(i, seed) for i in range(len(grid.SETTINGS)) if i not in heavy for seed in grid.SEEDS]
    rows, wheres = [], {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r, w in pool.map(task, *zip(*light)):
            rows += r; wheres.update(w)
    for i in heavy:                       # one at a time: each builds a ~0.6 GB place-code table
        for seed in grid.SEEDS:
            r, w = task(i, seed)
            rows += r; wheres.update(w)
    (HERE / "predictions.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(HERE / "predictions_where.npz", **{k.replace("|", "__"): v for k, v in wheres.items()})
    print(f"{len(rows)} predictions written")


if __name__ == "__main__":
    main()
