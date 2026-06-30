# Critic, round-2 gate, check G5: do the frozen theory and the committed library theory reproduce the hashed
# out-of-sample predictions? (If so, when the predictions were hashed is moot: they are a deterministic function
# of code committed before the measurement.) Two families, seed 30, run from results/oos/ as predict.py does.
import os, sys, json
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
OOS = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "results" / "oos")
sys.path.insert(0, OOS); sys.path.insert(0, OOS + "/frozen")
import numpy as np
import grid, t4_theory
from loci import theory
pred = {(r["family"], json.dumps(r["setting"], sort_keys=True), r["seed"], r["placement"]): r
        for r in json.load(open(OOS + "/predictions.json"))}
import dataclasses
for fam in ("Ns 2000", "cards 7/12/4"):
    idx = [i for i, s in enumerate(grid.SETTINGS) if s.family == fam][0]
    s = grid.SETTINGS[idx]; seed = 30
    patterns, factors, scaffold, alpha_read, alpha_place = grid.build(s, seed)
    arms = grid.placements(s, seed, patterns, factors, scaffold, alpha_place)
    for name, where in arms.items():
        out = t4_theory.theory(scaffold, patterns, where, alpha_read, grid.theory_rate(s), np.random.default_rng(seed),
                               draws=400, levels=("E", "T-relu"))
        lib = theory.predict(scaffold, patterns, where, alpha_read, flip_rate=grid.theory_rate(s),
                             rng=np.random.default_rng(seed)).address
        key = (s.family, json.dumps(dataclasses.asdict(s), sort_keys=True), seed, name)
        hashed = pred[key]["T-relu"]["address"]
        print(f"{fam:13} {name:8} hashed {hashed:.6f} | frozen re-run {out['T-relu'][1]:.6f} | library src/loci/theory.py {lib:.6f}")
