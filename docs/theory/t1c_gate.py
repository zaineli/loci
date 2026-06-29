"""Section 1, the gate: benefit_i = 2 (C_max - mean C) / r_i for each stored item re-arriving (LOO, eq 2.3:
r_i = 1/K_ii, C = -Q^T K e_i / K_ii without i). How much of its spread across items is 1/r and how much
is familiarity? Factored content, and a mixture: half factored items, half pure noise (novel)."""
from common import *
P, f = 800, 0.1
for kind in ("factored", "mixed"):
    for seed in (0, 1, 2):
        C = factored(DIM, P, np.random.default_rng(10_000 * seed + P)); S = C.patterns.copy()
        novel = np.zeros(P, bool)
        if kind == "mixed":
            novel[::2] = True
            S[:, novel] = np.sign(np.random.default_rng(seed).standard_normal((DIM, novel.sum())))
        alpha = mmse_alpha(P, flip_rate=f); K = np.linalg.inv(S.T @ S + alpha * np.eye(P))
        lab = C.factors[:, 0]                             # module 0 aligned with A (oracle)
        Kd = np.diag(K); off = K - np.diag(Kd)
        Cl = -(off @ np.eye(9)[lab]) / Kd[:, None]        # LOO phase sums of recall coefficients
        spread = Cl.max(1) - Cl.mean(1); r = 1 / Kd
        ben = 2 * spread / r
        lb, lr, ls = np.log(ben), np.log(r), np.log(spread)
        share_r = np.var(lr) / np.var(lb)
        msg = f"{kind:8} seed {seed}: r in [{r.min():.0f}, {r.max():.0f}] (alpha {alpha:.0f}, alpha+Ns {alpha + DIM:.0f});  var(log benefit) = {np.var(lb):.3f}, var(log r) = {np.var(lr):.3f}, var(log spread) = {np.var(ls):.3f}, corr(log r, log spread) = {np.corrcoef(lr, ls)[0,1]:+.2f}"
        if kind == "mixed":
            msg += f"\n            novel items: mean r {r[novel].mean():.0f}, mean benefit {ben[novel].mean():.2e};  familiar: mean r {r[~novel].mean():.0f}, mean benefit {ben[~novel].mean():.2e}"
        print(msg)
