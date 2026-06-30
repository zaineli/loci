# Critic, round-2 gate, check G3: is T3's worst row (0.068) a theory miss or one-cue sampling noise?
# Rebuilds E4's worst rows exactly (bench/consolidate.py), then measures recall over 20 independent cue
# draws: the between-draw SD, and the theory's error against the mean over draws.
import os, sys, json
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "bench"))
import numpy as np
import consolidate as bench
from loci import consolidate, place, theory
from loci.memory import Memory, flip, mmse_alpha
from loci.scaffold import Scaffold

E4 = json.load(open(__import__("pathlib").Path(__file__).resolve().parents[2] / "results" / "consolidate.json"))
cases = [("four", 47, 0.4, 0.2, "encode"), ("cards", 48, 0.4, 0.1, "random"), ("cards", 48, 0.4, 0.2, "encode"),
         ("four", 42, 0.4, 0.2, "random+replay")]
for kind, seed, load, rate, arm in cases:
    count = round(load * bench.DIM)
    content = bench._content(kind, count, np.random.default_rng(10_000 * seed + count))
    S = content.patterns
    sc = Scaffold(seed=seed)
    alpha = mmse_alpha(count, flip_rate=rate)
    enc, K = consolidate.encode(S, sc, alpha, capacity="growing")
    arms = {"random": place.scattered(count, sc, np.random.default_rng(seed + 1)), "encode": enc}
    arms["random+replay"] = consolidate.replay(K, sc, arms["random"], np.random.default_rng(100 + seed))
    where = arms[arm]
    mem = Memory(sc, rule="ridge", alpha=alpha); mem.store(S, where)
    first = flip(S, rate, np.random.default_rng(20_000 * seed + count + round(1_000 * rate)))
    r0 = float(np.mean(mem.recall(first).address == where))
    draws = [float(np.mean(mem.recall(flip(S, rate, np.random.default_rng(777 + d))).address == where)) for d in range(20)]
    th = theory.predict(sc, S, where, alpha, flip_rate=rate, rng=np.random.default_rng(seed)).address
    ref = [r for r in E4 if (r["content"], r["seed"], r["load"], r["flip"], r["placement"]) == (kind, seed, load, rate, arm)][0]
    print(f"{kind} seed {seed} P {count} flip {rate} {arm:14} E4 recall {ref['recall']:.3f} (rebuilt {r0:.3f}) theory {th:.3f} "
          f"(E4 {ref['theory']:.3f}) | 20 fresh draws: mean {np.mean(draws):.3f} sd {np.std(draws, ddof=1):.3f} "
          f"-> theory error vs mean {th - np.mean(draws):+.3f}; E4 row error {ref['theory'] - ref['recall']:+.3f}", flush=True)
