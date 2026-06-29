"""Section 5: redundant modules. (a) Pool(M), eq 5.2, from measured and predicted correlations;
(b) a 4th module (period 7, 49 phases) at Nh = 400: per-module snap, joint decode over stored
addresses (place-code matched filter, and summed grid input), against M = 3 and the flat control.

A light scaffold: the same projection law as loci.scaffold (faithful mask), templates W_gh summed in
chunks so the 176,400-address place-code table is never held in memory.
    python t5_modules.py
"""
from common import *
from structure import rho_h, light_projection
from math import prod


class Light:
    def __init__(self, periods, seed, Nh=400, theta=0.5):
        self.sizes = [p * p for p in periods]; self.M = len(periods)
        self.offsets = np.cumsum([0] + self.sizes[:-1]); self.theta = theta
        self.W = light_projection(periods, seed) if Nh == 400 else None
        self.Npos = prod(self.sizes)
        # W_gh = G H^T / Npos, summed over every phase tuple, one chunk per last-module phase
        Ng = sum(self.sizes); self.W_gh = np.zeros((Ng, Nh))
        grids = np.stack(np.meshgrid(*[np.arange(s) for s in self.sizes[:-1]], indexing="ij"), -1).reshape(-1, self.M - 1)
        for last in range(self.sizes[-1]):
            ph = np.column_stack([grids, np.full(len(grids), last)])
            h = self.place(ph)                                   # (Nh, n)
            g = self.grid(ph)
            self.W_gh += g @ h.T
        self.W_gh /= self.Npos

    def grid(self, ph):
        g = np.zeros((sum(self.sizes), len(ph)))
        for m, o in enumerate(self.offsets):
            g[o + ph[:, m], np.arange(len(ph))] = 1
        return g

    def place(self, ph):
        return relu(self.W[:, (self.offsets + ph).T].sum(1) if False else self.W @ self.grid(ph), self.theta)

    def snap(self, x):
        return np.stack([x[o:o + k].argmax(0) for o, k in zip(self.offsets, self.sizes)], 1)


def recall_modes(sc, S, ph, cues, alpha):
    """Per-module snap, joint stored decodes; returns address recovery for each read."""
    Ha = sc.place(ph)
    Wsh_map = Ha @ np.linalg.solve(S.T @ S + alpha * np.eye(S.shape[1]), S.T)
    h0 = relu(Wsh_map @ cues)
    x = sc.W_gh @ h0
    snap = sc.snap(x)
    out = {"snap": float(np.mean((snap == ph).all(1))), "snap modules": (snap == ph).mean(0).round(3).tolist()}
    out["stored (place code)"] = float(np.mean((Ha.T @ h0).argmax(0) == np.arange(S.shape[1])))
    Gs = sc.grid(ph)                                             # (Ng, P)
    out["stored (sum of grid inputs)"] = float(np.mean((Gs.T @ x).argmax(0) == np.arange(S.shape[1])))
    return out, Ha


def pool(Ha, ph, M):
    """Mean over items of sum_{j != i} corr(h_i, h_j)^2, split by shared-phase count q; and eq 5.2."""
    Hc = Ha - Ha.mean(0, keepdims=True); Hc /= np.linalg.norm(Hc, axis=0, keepdims=True)
    R = Hc.T @ Hc; np.fill_diagonal(R, 0)
    Q = (ph[:, None, :] == ph[None, :, :]).sum(-1); np.fill_diagonal(Q, -1)
    by_q = {q: float((R ** 2 * (Q == q)).sum() / len(R)) for q in range(M)}
    pred = sum(((Q == q).sum() / len(R)) * rho_h(q / M, M)[0] ** 2 for q in range(1, M))
    return by_q, pred


if __name__ == "__main__":
    P = 800
    for seed in (0, 1, 2):
        content = factored(DIM, P, np.random.default_rng(10_000 * seed + P)); S = content.patterns
        scs = {"M=3 (3,4,5)": Light((3, 4, 5), seed), "M=4 (3,4,5,7)": Light((3, 4, 5, 7), seed)}
        flat = Scaffold(periods=(60,), seed=seed)
        for rate in (0.1, 0.2):
            alpha = mmse_alpha(P, flip_rate=rate)
            cues = flip(S, rate, np.random.default_rng(20_000 * seed + P + int(1_000 * rate)))
            for name, sc in scs.items():
                rng = np.random.default_rng(seed + 1)
                idx = rng.choice(sc.Npos, P, replace=False)
                ph = np.stack([idx % k for k in sc.sizes], 1)        # CRT: address x -> x mod l_m
                res, Ha = recall_modes(sc, S, ph, cues, alpha)
                extra = ""
                if rate == 0.1:
                    by_q, pred = pool(Ha, ph, sc.M)
                    extra = f"  Pool by q {{{', '.join(f'{q}: {v:.2f}' for q, v in by_q.items())}}}  total(q>=1) {sum(v for q, v in by_q.items() if q > 0):.2f}  eq5.2 {pred:.2f}"
                print(f"seed {seed} flip {rate} {name:14} " + "  ".join(f"{k} {v}" for k, v in res.items()) + extra, flush=True)
            where = np.random.default_rng(seed + 1).permutation(flat.addresses)[:P]
            Hf = flat.H[:, where]
            h0 = relu(Hf @ np.linalg.solve(S.T @ S + alpha * np.eye(P), S.T) @ cues)
            print(f"seed {seed} flip {rate} {'flat (3600)':14} snap {np.mean(flat.snap(flat.W_gh @ h0)[:, 0] == where):.3f}  "
                  f"stored {np.mean((Hf.T @ h0).argmax(0) == np.arange(P)):.3f}", flush=True)
