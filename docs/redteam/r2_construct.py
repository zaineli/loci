# Critic, round-2 gate, check G4: is the construction score fair to a memory that organises factors differently?
# Decomposes bench/imagine.py's construction (gist lands on an EMPTY address AND the read-out decodes A, B and C)
# into: lands on an empty address; read-out decodes the held-out pair (A, B) only; decodes all three.
# If the self-filed memory's deficit shrinks a lot on (A, B)-only, part of the gap is factor C's granularity,
# which the oracle is handed (5 groups x 5 slots in module 2) and the self-filed memory must discover.
import os, sys
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "bench"))
import numpy as np
import imagine as bench
from loci import consolidate, imagine, place
from loci.memory import CueMaps, flip, mmse_alpha
from loci.scaffold import Scaffold, relu

kind = sys.argv[1] if len(sys.argv) > 1 else "factored"
seeds = [int(x) for x in sys.argv[2:]] or [40, 41, 42]
agg = {}
for seed in seeds:
    content, probes, _ = bench._content(kind, seed)
    S, F = content.patterns, content.factors
    sc = Scaffold(seed=seed)
    alpha = mmse_alpha(bench.COUNT, flip_rate=0.1)
    enc, K = consolidate.encode(S, sc, alpha, capacity="growing")
    arms = {"oracle": place.oracle(sc, F),
            "encode+replay": consolidate.replay(K, sc, enc, np.random.default_rng(100 + seed))}
    for name, where in arms.items():
        codes = sc.H[:, where]
        w_hs = CueMaps(S).map(codes, alpha)
        w_sh = S @ np.linalg.pinv(codes)
        stored = np.zeros(sc.addresses, bool); stored[where] = True
        rng = np.random.default_rng(1_000 * seed + 10)
        _ = flip(S, 0.1, rng)                                   # keep the bench's stream order: studied first
        gist = flip(probes.gist, 0.1, rng)
        h0 = relu(w_hs @ gist)
        for dec in ("nearest", "snap"):
            got = imagine.decode(sc, where, h0, dec)
            readout = np.sign(w_sh @ sc.H[:, got])
            hits = np.stack([content.decode(readout, f) == probes.gist_factors[:, f] for f in range(3)], 1)
            empty = ~stored[got]
            row = (empty.mean(), (empty & hits[:, 0] & hits[:, 1]).mean(), (empty & hits.all(1)).mean(),
                   hits[:, 0].mean(), hits[:, 1].mean(), hits[:, 2].mean())
            agg.setdefault((name, dec), []).append(row)
print(f"{kind}, seeds {seeds}, 10% flips: gist probes")
print(f"{'placement':14} {'decoder':8} {'empty':>6} {'A&B, empty':>11} {'A&B&C, empty (bench)':>22} {'A':>6} {'B':>6} {'C':>6}")
for (name, dec), v in agg.items():
    m = np.mean(v, 0)
    print(f"{name:14} {dec:8} {m[0]:6.3f} {m[1]:11.3f} {m[2]:22.3f} {m[3]:6.3f} {m[4]:6.3f} {m[5]:6.3f}")
