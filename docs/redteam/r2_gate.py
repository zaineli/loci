# Critic, round-2 gate, check G1: does the model's own recall-to-reject signal (cue vs read-out overlap)
# separate construction from false recall? README: "whatever decoding constructs it from a gist cue also
# recalls a recombined event as experienced" and "Its price is truth".
# Same content, probes, placements and cues as bench/imagine.py (factored, seeds 40-42, 10% flips).
# Gate: a recall counts as "experienced" only if overlap(cue, read-out) >= the 5th percentile of studied cues.
import os, sys
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "bench"))
import numpy as np
import imagine as bench
from loci import consolidate, imagine, place
from loci.memory import CueMaps, flip, mmse_alpha
from loci.scaffold import Scaffold, relu

kind = sys.argv[1] if len(sys.argv) > 1 else "factored"
rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
seeds = [int(x) for x in sys.argv[3:]] or [40, 41, 42]
for seed in seeds:
    content, probes, _ = bench._content(kind, seed)
    S, F = content.patterns, content.factors
    sc = Scaffold(seed=seed)
    alpha_place = mmse_alpha(bench.COUNT, flip_rate=0.1)
    alpha = mmse_alpha(bench.COUNT, flip_rate=rate)
    enc, K = consolidate.encode(S, sc, alpha_place, capacity="growing")
    arms = {"oracle": place.oracle(sc, F),
            "encode+replay": consolidate.replay(K, sc, enc, np.random.default_rng(100 + seed))}
    for name, where in arms.items():
        codes = sc.H[:, where]
        w_hs = CueMaps(S).map(codes, alpha)
        w_sh = S @ np.linalg.pinv(codes)
        stored = np.zeros(sc.addresses, bool); stored[where] = True
        rng = np.random.default_rng(1_000 * seed + round(100 * rate))   # the bench's stream
        cues = {n: flip(c, rate, rng) for n, c in (("studied", S), ("gist", probes.gist),
                                                  ("recombined", probes.recombined), ("unrelated", probes.unrelated))}
        out = {}
        for n, cue in cues.items():
            h0 = relu(w_hs @ cue)
            got = imagine.decode(sc, where, h0, "nearest")
            readout = np.sign(w_sh @ sc.H[:, got])
            match = (cue * readout).mean(0)
            if n in ("gist", "recombined"):
                truth = probes.gist_factors if n == "gist" else probes.recombined_factors
                every = np.all(np.stack([content.decode(readout, f) for f in range(3)], 1) == truth, 1)
                built = ~stored[got] & every
            else:
                built = None
            out[n] = (match, built)
        t = np.quantile(out["studied"][0], 0.05)
        mid = 0.5 * (out["studied"][0].mean() + out["recombined"][0].mean())
        g_m, g_b = out["gist"]; r_m, r_b = out["recombined"]
        print(f"{kind} {rate} seed {seed} {name:14} overlap(cue, read-out) mean: studied {out['studied'][0].mean():.3f} "
              f"gist {g_m.mean():.3f} recombined {r_m.mean():.3f} unrelated {out['unrelated'][0].mean():.3f} | "
              f"gate t={t:.3f}: gist constructed {g_b.mean():.3f}, constructed & passes {np.mean(g_b & (g_m >= t)):.3f} | "
              f"recombined 'false recall' {r_b.mean():.3f}, after the gate {np.mean(r_b & (r_m >= t)):.3f} | midpoint gate {mid:.3f}: studied pass {np.mean(out['studied'][0] >= mid):.3f}, false recall {np.mean(r_b & (r_m >= mid)):.3f}", flush=True)
