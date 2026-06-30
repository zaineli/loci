# Critic, round-2 gate, check G2: does "a memory that files itself" survive when the choice is made by the
# network's own pathway rather than by exact phase sums of K?
#   exact  (loci.consolidate):  score(address) = sum_m (Q_m^T c)[phi_m(address)]      c from K, via RLS
#   native (this script):       score(address) = sum_m (W_gh ReLU(H_a c))_m[phi_m(address)]
#          i.e. the cue map's own place activity (W_hs s = H_a c), the Hebbian grid map and the per-module
#          grid inputs the snap uses; replay uses the trace-free recall H_a(-alpha K e_i) the same way.
# Same content, scaffold, alpha, cues and capacity rules as bench/consolidate.py (E4 headline: factored,
# P/Ns 0.8, 10% flips, seeds 40-42), so the exact arms can be checked against results/consolidate.json.
import os, sys, time, json
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
import numpy as np
from loci import consolidate, place
from loci.content import factored
from loci.memory import Memory, flip, mmse_alpha
from loci.scaffold import Scaffold, relu

DIM, LOAD, RATE = 1_000, 0.8, 0.1
count = round(LOAD * DIM)
seeds = [int(x) for x in sys.argv[1:]] or [40, 41, 42]
E4 = json.load(open(__import__("pathlib").Path(__file__).resolve().parents[2] / "results" / "consolidate.json"))


MODE = os.environ.get("NATIVE_MODE", "raw")   # raw: summed grid input; z: per-module z-scored grid input


def scores_from(gi, sc):
    """(sizes...) array: sum over modules of each address's grid input (optionally z-scored per module)."""
    total = np.zeros(sc.sizes)
    for m, (o, s) in enumerate(zip(sc.offsets, sc.sizes)):
        shape = [1] * len(sc.sizes); shape[m] = s
        block = gi[o:o + s]
        if MODE == "z":
            block = (block - block.mean()) / (block.std() + 1e-12)
        total = total + block.reshape(shape)
    return total


def native_encode(S, sc, alpha):
    sizes, P = sc.sizes, S.shape[1]
    phases = np.zeros((P, len(sizes)), dtype=np.int64)
    taken = np.zeros(sizes, dtype=bool); taken[(0,) * len(sizes)] = True
    K = np.array([[1.0 / (S[:, 0] @ S[:, 0] + alpha)]])
    for n in range(1, P):
        item = S[:, n]; overlaps = S[:, :n].T @ item; c = K @ overlaps
        residual = item @ item + alpha - overlaps @ c
        codes = sc.H[:, sc.index(phases[:n])]
        gi = sc.W_gh @ relu(codes @ c)                      # the network's own recall of the new item
        score = scores_from(gi, sc)
        capped = np.zeros(sizes)
        for m, size in enumerate(sizes):
            load = np.bincount(phases[:n, m], minlength=size)
            shape = [1] * len(sizes); shape[m] = size
            capped = capped + (load >= np.ceil((n + 1) / size)).astype(float).reshape(shape)
        score = score - 1e6 * (capped > 0) * (np.abs(score).max() + 1)
        score[taken] = -np.inf
        best = np.unravel_index(int(score.argmax()), sizes)
        taken[best] = True; phases[n] = best
        K = np.block([[K + np.outer(c, c) / residual, -c[:, None] / residual],
                      [-c[None, :] / residual, np.array([[1.0 / residual]])]])
    return sc.index(phases), K


def native_replay(K, sc, where, alpha, rng, sweeps=30, slack=1.1):
    sizes, P = sc.sizes, len(where)
    phases = sc.phases[where].copy()
    caps = [int(np.ceil(slack * P / s)) for s in sizes]
    loads = [np.bincount(phases[:, m], minlength=s) for m, s in enumerate(sizes)]
    taken = np.zeros(sizes, dtype=bool); taken[tuple(phases.T)] = True
    codes = sc.H[:, sc.index(phases)].copy()
    bias = -alpha * K                                        # column i: item i's trace-free recall coefficients
    used = 0
    for sweep in range(sweeps):
        moved = 0
        for i in rng.permutation(P):
            gi = sc.W_gh @ relu(codes @ bias[:, i])
            score = scores_from(gi, sc)
            here = tuple(phases[i])
            for m, s in enumerate(sizes):
                shape = [1] * len(sizes); shape[m] = s
                full = ((loads[m] >= caps[m]) & (np.arange(s) != here[m])).astype(float).reshape(shape)
                score = np.where(full > 0, -np.inf, score)
            stay = score[here]
            score[taken] = -np.inf; score[here] = stay
            there = np.unravel_index(int(score.argmax()), sizes)
            if there != here and score[there] > stay + 1e-12:
                taken[here], taken[there] = False, True
                for m in range(len(sizes)):
                    loads[m][here[m]] -= 1; loads[m][there[m]] += 1
                phases[i] = there
                codes[:, i] = sc.H[:, sc.index(np.array([there]))[0]]
                moved += 1
        used = sweep + 1
        if moved == 0:
            break
    return sc.index(phases), used


for seed in seeds:
    t = time.time()
    content = factored(DIM, count, np.random.default_rng(10_000 * seed + count))
    S, F = content.patterns, content.factors
    sc = Scaffold(seed=seed)
    alpha = mmse_alpha(count, flip_rate=RATE)
    cues = flip(S, RATE, np.random.default_rng(20_000 * seed + count + round(1_000 * RATE)))

    def recall(where):
        mem = Memory(sc, rule="ridge", alpha=alpha); mem.store(S, where)
        return float(np.mean(mem.recall(cues).address == where))

    enc, K = consolidate.encode(S, sc, alpha, capacity="growing")
    ex = consolidate.replay(K, sc, enc, np.random.default_rng(100 + seed))
    nenc, _ = native_encode(S, sc, alpha)
    nrep, used = native_replay(K, sc, nenc, alpha, np.random.default_rng(100 + seed))
    ref = {r["placement"]: r["recall"] for r in E4 if r["content"] == "factored" and r["seed"] == seed
           and r["load"] == LOAD and r["flip"] == RATE}
    print(f"[{MODE}] seed {seed}: exact encode {recall(enc):.3f} (E4 {ref['encode']:.3f}) + replay {recall(ex):.3f} "
          f"(E4 {ref['encode+replay']:.3f}) | NATIVE encode {recall(nenc):.3f} + replay {recall(nrep):.3f} "
          f"({used} sweeps) | E4 k-means {ref['k-means']:.3f} random {ref['random']:.3f} oracle {ref['oracle']:.3f} "
          f"| law exact {consolidate.law(K, sc, ex):.3f} native {consolidate.law(K, sc, nrep):.3f} ({time.time() - t:.0f}s)",
          flush=True)
