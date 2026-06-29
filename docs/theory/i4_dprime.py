"""Check of Prop I.7: recognition d' in the critic's C8 setting (20% of (A, B) pairs held out, P = 800,
Nh = 400, 10% flips on every probe, 300 probes per class).

Signals:  F1 = cos(h0, h_out)            (the critic's: h0 vs the place code it is decoded to)
          F2 = mean(s~ * sign(W_sh h_out))  (cue vs read-out)
Decoders: snap (paper), joint (cosine over stored), all (cosine over all 3,600 tuples).
Per class we report the mixture moments of (I.8) for studied items (correct vs failed recall), the
identity check, the cap sqrt(2 rho / (1 - rho)), and the Gaussian-cue (T-mc) prediction of every d'.
"""
from common import *

Ns, P, f = 1000, 800, 0.1


def dprime(x, y):
    return (x.mean() - y.mean()) / np.sqrt(0.5 * (x.var() + y.var()))


def signals(h0, cues, grid, Ha, Wsh, stored_idx, Hn_all, where):
    Hn = Ha / np.linalg.norm(Ha, axis=0, keepdims=True)
    outs = {"snap": grid.index(grid.snap(grid.W_gh @ h0)),
            "joint": where[(Hn.T @ h0).argmax(0)],
            "all": (Hn_all.T @ h0).argmax(0)}
    res = {}
    for d, idx in outs.items():
        h = grid.H[:, idx]
        f1 = (h0 * h).sum(0) / (np.linalg.norm(h0, axis=0) * np.linalg.norm(h, axis=0) + 1e-12)
        f2 = (cues * np.sign(Wsh @ h)).mean(0)
        res[d] = (idx, f1, f2)
    return res


def run(seed, placement):
    rng = np.random.default_rng(500 + seed)
    held = rng.random((9, 16)) < 0.2
    C = factored(Ns, 3000, np.random.default_rng(10_000 * seed + P))
    keep = np.flatnonzero(~held[C.factors[:, 0], C.factors[:, 1]])[:P]
    S, Fa = C.patterns[:, keep], C.factors[keep]
    grid = Scaffold(seed=seed); alpha = mmse_alpha(P, flip_rate=f)
    n = 300
    pairs = np.argwhere(held); pick = pairs[rng.integers(0, len(pairs), n)]
    lures_f = np.column_stack([pick, rng.integers(0, 5, n)])
    rec = np.sign(sum(C.codes[i][:, lures_f[:, i]] for i in range(3)) + rng.standard_normal((Ns, n)))
    clean = {"studied": S[:, :n], "recombination": rec, "unrelated": np.sign(rng.standard_normal((Ns, n)))}
    probes = {k: flip(v, f, np.random.default_rng(seed + 9)) for k, v in clean.items()}
    where = place.scattered(P, grid, np.random.default_rng(seed + 1)) if placement == "random" else place.oracle(grid, Fa)
    Ha = grid.H[:, where]; K = np.linalg.inv(S.T @ S + alpha * np.eye(P)); Wc = K @ S.T
    Wsh = S @ np.linalg.pinv(Ha); Hn_all = grid.H / np.linalg.norm(grid.H, axis=0, keepdims=True)
    meas = {k: signals(relu(Ha @ (Wc @ x)), x, grid, Ha, Wsh, np.arange(n), Hn_all, where) for k, x in probes.items()}
    # T-mc: Gaussian cue model, one draw per probe (as one flip realisation), cue for F2 = a * clean (its mean)
    a, s2 = 0.8, 0.36
    covu = s2 * Ha @ (K - alpha * K @ K) @ Ha.T
    w, V = np.linalg.eigh(covu); Lu = V * np.sqrt(np.clip(w, 0, None))
    tm = {}
    for k, x in clean.items():
        Z = np.random.default_rng(seed + 31 + len(k)).standard_normal((n, Ha.shape[0])) @ Lu.T
        h0 = relu(a * (Ha @ (Wc @ x)) + Z.T)
        tm[k] = signals(h0, probes[k], grid, Ha, Wsh, np.arange(n), Hn_all, where)
    return meas, tm, where[:n]


def summarise(meas, truth):
    out = {}
    for d in ("snap", "joint", "all"):
        for s in (1, 2):
            st = meas["studied"][d][s]; rc = meas["recombination"][d][s]; un = meas["unrelated"][d][s]
            ok = meas["studied"][d][0] == truth
            rho = ok.mean()
            mix = {"rho": rho, "mu_ok": st[ok].mean() if ok.any() else np.nan, "mu_err": st[~ok].mean() if (~ok).any() else np.nan,
                   "sd_ok": st[ok].std() if ok.any() else np.nan, "sd_err": st[~ok].std() if (~ok).sum() > 1 else 0.0,
                   "mu_rec": rc.mean(), "mu_un": un.mean()}
            out[d, s] = (dprime(st, rc), dprime(st, un), mix)
    return out


if __name__ == "__main__":
    for placement in ("random", "aligned"):
        res = [run(s, placement) for s in (0, 1, 2)]
        print(f"\n===== {placement}")
        for d in ("snap", "joint", "all"):
            for s, sname in ((1, "F1 cos(h0,h_out)"), (2, "F2 cue-readout")):
                M = [summarise(r[0], r[2])[d, s] for r in res]; T = [summarise(r[1], r[2])[d, s] for r in res]
                rho = np.mean([m[2]["rho"] for m in M]); cap = np.sqrt(2 * rho / (1 - rho)) if rho < 1 else np.inf
                mix = {k: np.nanmean([m[2][k] for m in M]) for k in M[0][2]}
                print(f"{d:5} {sname:17}: d'(studied vs recomb) meas {' '.join(f'{m[0]:+.2f}' for m in M)} | T-mc {' '.join(f'{t[0]:+.2f}' for t in T)}"
                      f"   d'(studied vs unrelated) meas {' '.join(f'{m[1]:+.2f}' for m in M)} | T-mc {' '.join(f'{t[1]:+.2f}' for t in T)}"
                      f"   [rho {rho:.3f}, cap {cap:.2f}; mu_ok {mix['mu_ok']:.3f} mu_err {mix['mu_err']:.3f} "
                      f"mu_recomb {mix['mu_rec']:.3f} mu_unrel {mix['mu_un']:.3f}; sd_ok {mix['sd_ok']:.3f}]", flush=True)
