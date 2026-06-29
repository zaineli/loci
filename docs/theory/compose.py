"""Does an aligned grid scaffold compose? Read-outs for held-out factor combinations.

For each (seed, P) we build factored content in which a set of (a, b) pairs never co-occur, place
it with every placement, and store it on the grid and flat scaffolds for each Nh. We then query
the held-out combinations in four ways:
  address  read out at the address voted for by the query's factor values. For each module, take
           the phase most enriched for items that carry a, b or c (summed over the three factors,
           so no module-to-factor assignment is assumed). For module 2, pick the 5-slot group
           first, then its highest-scoring free slot.
  cue      recall the factor-only cue sign(A+B+C) (clean, or with 10% flips) through the ridge
           path (alpha = MMSE for 10% flips) and the scaffold's snap. The read-out depends only
           on the address the cue lands on, so it cannot echo the cue.
  cue-pinv the same with alpha = 0 (the paper's pseudo-inverse).
  avg      the label-average route: h = mean H over items with a + with b + with c - 2 mean H.
           This is the additive model pushed through W_sh, and it works for any placement.
Baselines: kNN (nearest stored pattern to the cue) and the additive model,
sign(mean_a + mean_b + mean_c - 2 grand mean).

Usage:
  python compose.py main       [--seeds 0 1 2] [--P 400 800] [--Nh 400 800]
  python compose.py cards      (factor cardinalities 7, 12, 4; P 800, Nh 400)
"""

from __future__ import annotations

import os

for _t in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_t] = "2"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from loci import place  # noqa: E402
from loci.content import Factored  # noqa: E402
from loci.memory import flip, mmse_alpha  # noqa: E402
from loci.scaffold import Scaffold, relu  # noqa: E402

NS = 1000
HERE = Path(__file__).resolve().parent
FLIP = 0.1
PLACEMENTS = ("oracle", "learned", "learned-fact", "kmeans", "random", "sequential")
LAMS = (0.01, 0.1, 1.0)  # W_sh ridge, relative to the mean eigenvalue of H_a H_a^T (diagnostic only)


# ---- content ---------------------------------------------------------------------------------------


def holdout(cards, rng) -> list[tuple[int, int]]:
    """One held-out partner a per project b (a = perm[b mod |A|]): every a loses 1-2 partners of |B|."""
    perm = rng.permutation(cards[0])
    return [(int(perm[b % cards[0]]), b) for b in range(cards[1])]


def make_factored(P, rng, cards, held, detail=1.0) -> Factored:
    codes = tuple(rng.standard_normal((NS, k)) for k in cards)
    held_set = set(held)
    allowed = np.array([(a, b) for a in range(cards[0]) for b in range(cards[1]) if (a, b) not in held_set])
    ab = allowed[rng.integers(0, len(allowed), P)]
    factors = np.column_stack([ab, rng.integers(0, cards[2], P)])
    own = detail * rng.standard_normal((NS, P))
    signal = sum(codes[i][:, factors[:, i]] for i in range(3))
    return Factored(np.sign(signal + own), factors, codes, own)


def composite(codes, q) -> np.ndarray:
    return np.sign(sum(codes[f][:, q[:, f]] for f in range(3)))


def query_sets(factors, cards, held, rng, n_seen=80):
    heldq = np.array([(a, b, c) for a, b in held for c in range(cards[2])])
    combos = np.unique(factors, axis=0)
    seenq = combos[rng.choice(len(combos), min(n_seen, len(combos)), replace=False)]
    return {"held": heldq, "seen": seenq}


# ---- metrics ---------------------------------------------------------------------------------------


def nmi(x, y) -> float:
    j = np.zeros((x.max() + 1, y.max() + 1))
    np.add.at(j, (x, y), 1)
    j /= j.sum()
    px, py = j.sum(1), j.sum(0)
    nz = j > 0
    mi = (j[nz] * np.log(j[nz] / np.outer(px, py)[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()  # noqa: E731
    return float(2 * mi / (h(px) + h(py)))


def score(R, target, q, S, factors, decode, additive):
    """Mean metrics of read-outs R (Ns, n) for queries q (n, 3)."""
    dec = np.stack([decode(R, f) == q[:, f] for f in range(3)], axis=1)
    sims = S.T @ R / NS
    nn = sims.argmax(0)
    return {
        "ov_comp": float(np.mean((R * target).mean(0))),
        "all3": float(dec.all(1).mean()),
        "dec": dec.mean(0).round(3).tolist(),
        "ov_nn": float(sims.max(0).mean()),
        "nn_share": float((factors[nn] == q).sum(1).mean()),
        "ov_add": float(np.mean((R * additive).mean(0))),
    }


def additive_model(S, factors, q, cards):
    means = [np.stack([S[:, factors[:, f] == v].mean(1) if (factors[:, f] == v).any() else np.zeros(len(S))
                       for v in range(cards[f])], 1) for f in range(3)]
    grand = S.mean(1, keepdims=True)
    return np.sign(sum(means[f][:, q[:, f]] for f in range(3)) - 2 * grand)


# ---- addressing ------------------------------------------------------------------------------------


def vote(sc: Scaffold, where, factors, q):
    """Per module, the phase most enriched for the query's factor values; module 2 by 5-slot group,
    then its best free slot. Returns (phases (n, M), empty (n,))."""
    ph = sc.phases[where]
    P = len(where)
    taken = np.zeros(sc.sizes, dtype=bool)
    taken[tuple(ph.T)] = True
    out = np.zeros((len(q), len(sc.sizes)), dtype=np.int64)
    empty = np.zeros(len(q), dtype=bool)
    for n, row in enumerate(q):
        masks = [factors[:, f] == row[f] for f in range(3)]
        scores = []
        for m, l in enumerate(sc.sizes):
            base = np.bincount(ph[:, m], minlength=l) / P
            s = sum(np.bincount(ph[mk, m], minlength=l) / max(mk.sum(), 1) - base for mk in masks)
            scores.append(s)
        if len(sc.sizes) == 1:  # flat: the single most enriched phase
            out[n, 0] = scores[0].argmax()
            empty[n] = not taken[out[n, 0]]
            continue
        p0, p1 = scores[0].argmax(), scores[1].argmax()
        g = scores[2].reshape(-1, place.SLOTS).sum(1).argmax()
        slots = g * place.SLOTS + np.argsort(-scores[2][g * place.SLOTS:(g + 1) * place.SLOTS])
        free = [s for s in slots if not taken[p0, p1, s]]
        out[n] = (p0, p1, free[0] if free else slots[0])
        empty[n] = bool(free)
    return out, empty


def learned_fact(S, alpha, rng, mu=1.0, starts=20, groups=place.GROUPS):
    """`place.learned(error=True)` plus a label-free independence penalty: module m's objective is
    sum_k q_k^T (off + mu * sum_prev s_prev * Co_prev) q_k, where Co_prev[i, j] = 1 if items i and
    j share a group in an earlier module, and s_prev is that module's mean within-group benefit
    -off[i, j]. The penalty sum_k sum_g n_kg^2 is smallest when module m is independent of the
    earlier modules (a factorial code). At mu = 1 an already co-grouped pair earns nothing on average.
    Returns labels (P, 3)."""
    K = place.precision(S, alpha)
    off = K - np.diag(np.diag(K))
    P = S.shape[1]
    labels, pen = [], np.zeros_like(off)
    for k in groups:
        kern = off + pen
        runs = [place._swaps(kern, rng.permutation(np.arange(P) % k), rng) for _ in range(starts)]
        best = min(runs, key=lambda r: place._within(kern, r))
        labels.append(best)
        co = (best[:, None] == best[None, :]).astype(float)
        np.fill_diagonal(co, 0)
        pen = pen + mu * (-off[co > 0].mean()) * co
    return np.stack(labels, 1)


def placement_nmi(sc: Scaffold, where, factors):
    ph = sc.phases[where].copy()
    ph[:, 2] //= place.SLOTS
    return [[round(nmi(ph[:, m], factors[:, f]), 3) for f in range(3)] for m in range(3)]


# ---- one condition ---------------------------------------------------------------------------------


def evaluate(S, factors, cards, decode, qsets, targets, cues, seed, P, Nhs, placements, flat=True,
             item_target=None, log=print):
    """qsets/targets/cues: dicts keyed by query-set name. Returns a list of result rows."""
    alpha = mmse_alpha(P, flip_rate=FLIP)
    rng = np.random.default_rng(1000 + seed)
    grid0 = Scaffold(seed=seed)  # addresses only depend on sizes, not Nh
    t = time.time()
    where = {}
    labels_nmi = {}
    for name in placements:
        if name == "oracle":
            where[name] = place.oracle(grid0, factors)
        elif name == "learned":
            where[name] = place.learned(S, grid0, np.random.default_rng(seed + 4), alpha, error=True)[0]
        elif name == "learned-fact":
            where[name] = place.addresses(grid0, learned_fact(S, alpha, np.random.default_rng(seed + 4)))
        elif name == "kmeans":
            where[name] = place.kmeans(S, grid0, np.random.default_rng(seed + 2))[0]
        elif name == "random":
            where[name] = place.scattered(P, grid0, np.random.default_rng(seed + 1))
        elif name == "sequential":
            where[name] = place.sequential(P)
        assert len(np.unique(where[name])) == P
        labels_nmi[name] = placement_nmi(grid0, where[name], factors)
    log(f"  placements {time.time() - t:.0f}s; NMI(module, factor): " +
        "; ".join(f"{k} {np.round(np.diag(v), 2).tolist()}" for k, v in labels_nmi.items()))

    M_ridge = np.linalg.solve(S.T @ S + alpha * np.eye(P), S.T)
    M_pinv = np.linalg.pinv(S)
    noisy = {k: flip(cues[k], FLIP, np.random.default_rng(77 + seed)) for k in cues}
    rows = []
    base = {"seed": seed, "P": P}
    # baselines (no scaffold)
    for qs, q in qsets.items():
        add = additive_model(S, factors, q, cards)
        knn = S[:, (S.T @ cues[qs]).argmax(0)]
        order = np.argsort(-(S.T @ cues[qs]), axis=0)
        blends = [(f"knn{k}", np.sign(sum(S[:, order[j]] for j in range(k)) + 1e-9)) for k in (2, 5, 10, 20)]
        for method, R in (("knn", knn), *blends, ("additive", add), ("target", targets[qs])):
            rows.append({**base, "Nh": None, "scaffold": "none", "placement": "-", "query": qs,
                         "method": method, **score(R, targets[qs], q, S, factors, decode, add)})
    for Nh in Nhs:
        scaffolds = {"grid": Scaffold(seed=seed, place_cells=Nh)}
        if flat:
            scaffolds["flat"] = Scaffold(periods=(60,), seed=seed, place_cells=Nh)
        for sname, sc in scaffolds.items():
            for pname in placements:
                w = where[pname]
                Ha = sc.H[:, w]
                W_sh = S @ np.linalg.pinv(Ha)
                common = {**base, "Nh": Nh, "scaffold": sname, "placement": pname,
                          "nmi": labels_nmi[pname] if sname == "grid" else None}
                # stored reference: read-out at a stored item's own address
                pick = np.random.default_rng(5 + seed).choice(P, 80, replace=False)
                Rst = np.sign(W_sh @ Ha[:, pick])
                rows.append({**common, "query": "stored", "method": "address",
                             "ov_item": float(np.mean((Rst * S[:, pick]).mean(0))),
                             "ov_comp": float(np.mean((Rst * item_target(factors[pick])).mean(0)))
                             if item_target else None})
                grid_ph_all = grid0.phases  # grid phases of every address index
                for qs, q in qsets.items():
                    add = additive_model(S, factors, q, cards)
                    tgt = targets[qs]
                    # address query
                    ph, empty = vote(sc, w, factors, q)
                    R = np.sign(W_sh @ sc.place(ph))
                    rows.append({**common, "query": qs, "method": "address", "empty": float(empty.mean()),
                                 **score(R, tgt, q, S, factors, decode, add)})
                    if sname == "grid":
                        # the whole voted 5-slot group at (p0, p1): sum of the pre-sign read-outs
                        taken = np.zeros(sc.sizes, dtype=bool)
                        taken[tuple(sc.phases[w].T)] = True
                        pre = np.zeros((len(S), len(q)))
                        for n, (p0, p1, p2) in enumerate(ph):
                            g0 = (p2 // place.SLOTS) * place.SLOTS
                            slots = [(p0, p1, g0 + s_) for s_ in range(place.SLOTS)]
                            free = [x for x in slots if not taken[x]] or slots
                            pre[:, n] = (W_sh @ sc.place(np.array(free))).sum(1)
                        rows.append({**common, "query": qs, "method": "address5",
                                     **score(np.sign(pre), tgt, q, S, factors, decode, add)})
                        if pname in ("oracle", "learned-fact"):
                            gram = Ha @ Ha.T
                            scale = np.trace(gram) / len(gram)
                            for lam in LAMS:
                                W_l = S @ Ha.T @ np.linalg.inv(gram + lam * scale * np.eye(len(gram)))
                                R = np.sign(W_l @ sc.place(ph))
                                Rst_l = np.sign(W_l @ Ha[:, pick])
                                rows.append({**common, "query": qs, "method": f"address-lam{lam}",
                                             "ov_item": float(np.mean((Rst_l * S[:, pick]).mean(0))),
                                             **score(R, tgt, q, S, factors, decode, add)})
                    if sname == "flat":  # the flat code at the grid's voted index: a random code
                        gph, _ = vote(grid0, w, factors, q)
                        idx = grid0.index(gph)
                        R = np.sign(W_sh @ sc.place(idx[:, None]))
                        rows.append({**common, "query": qs, "method": "address-gridindex",
                                     **score(R, tgt, q, S, factors, decode, add)})
                        gvote = gph
                    else:
                        gvote = ph
                    # label-average route
                    hs = sum(np.stack([Ha[:, factors[:, f] == v].mean(1) for v in q[:, f]], 1) for f in range(3))
                    hs = hs - 2 * Ha.mean(1, keepdims=True)
                    R = np.sign(W_sh @ hs)
                    rows.append({**common, "query": qs, "method": "avg", **score(R, tgt, q, S, factors, decode, add)})
                    # cue queries
                    for method, Mc, cue in (("cue", M_ridge, cues[qs]), ("cue-flip", M_ridge, noisy[qs]),
                                            ("cue-pinv", M_pinv, cues[qs])):
                        h0 = relu(Ha @ (Mc @ cue))
                        lp = sc.snap(sc.W_gh @ h0)
                        addr = sc.index(lp)
                        landed_empty = ~np.isin(addr, w)
                        R = np.sign(W_sh @ sc.place(lp))
                        row = {**common, "query": qs, "method": method, "empty": float(landed_empty.mean()),
                               **score(R, tgt, q, S, factors, decode, add)}
                        # does it land on the voted (imagined) grid address?  on flat compare index
                        lg = grid_ph_all[addr]
                        match = (lg[:, 0] == gvote[:, 0]) & (lg[:, 1] == gvote[:, 1]) & \
                                (lg[:, 2] // place.SLOTS == gvote[:, 2] // place.SLOTS)
                        row["match"] = float(match.mean())
                        row["mod_match"] = [float(np.mean(lg[:, m] == gvote[:, m])) if m < 2 else
                                            float(np.mean(lg[:, 2] // place.SLOTS == gvote[:, 2] // place.SLOTS))
                                            for m in range(3)]
                        rows.append(row)
                        if method == "cue" and pname in ("oracle", "random"):
                            # no snap: the read-out of the raw place activity, which can echo the cue
                            R = np.sign(W_sh @ h0)
                            rows.append({**common, "query": qs, "method": "cue-nosnap",
                                         **score(R, tgt, q, S, factors, decode, add)})
    return rows


# ---- drivers ---------------------------------------------------------------------------------------


def run_factored(seed, P, Nhs, cards=(9, 16, 5), placements=PLACEMENTS, tag="main"):
    rng = np.random.default_rng(31_000 * seed + P + 7 * sum(cards))
    held = holdout(cards, rng)
    C = make_factored(P, rng, cards, held)
    qsets = query_sets(C.factors, cards, held, rng)
    targets = {k: composite(C.codes, q) for k, q in qsets.items()}
    cues = dict(targets)
    t = time.time()
    rows = evaluate(C.patterns, C.factors, cards, C.decode, qsets, targets, cues, seed, P, Nhs, placements,
                    item_target=lambda fq: composite(C.codes, fq), log=lambda s: print(f"[{tag} seed {seed} P {P}]{s}", flush=True))
    for r in rows:
        r["content"] = f"factored{cards}"
        r["tag"] = tag
    print(f"[{tag} seed {seed} P {P}] done {time.time() - t:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("main", "cards"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--P", type=int, nargs="+", default=[400, 800])
    ap.add_argument("--Nh", type=int, nargs="+", default=[400, 800])
    args = ap.parse_args()
    rows = []
    out = HERE / f"results-{args.mode}.jsonl"
    for seed in args.seeds:
        if args.mode == "main":
            for P in args.P:
                rows += run_factored(seed, P, args.Nh)
        else:
            for P, Nh in ((800, 400), (400, 800)):
                rows += run_factored(seed, P, [Nh], cards=(7, 12, 4), tag="cards")
        out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
