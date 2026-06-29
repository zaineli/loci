"""Explain the Hebbian snap's disagreement with the address-greedy choice (section 1, eq 1.8).

The template overlap T_m[k, a] = t_mk^T h_a is a function on the full product of phases, so it has an
exact ANOVA: T_m[k, a] = mu_mk + sum_m' F^{mm'}[k, phi_m'(a)] + interactions. The no-ReLU Hebbian
input is then x_m = sum_m' F^{mm'} C^(m') + mu_m 1^T c + (interactions) c, with C^(m') = G_m' c the
recall coefficients summed per phase of module m'. Prediction, from the scaffold alone plus the C's:
  within    : argmax F^{mm} C^(m)                     (only module m's own templates)
  +cross    : argmax sum_m' F^{mm'} C^(m')            (cross-module leak included)
compared with the exact no-ReLU Hebbian input and with the address-greedy argmax C^(m).
"""
from common import *

Ns, P, f = 1000, 800, 0.1
GROUPS = place.GROUPS
CAP = [int(np.ceil(P / k)) for k in GROUPS]


def anova(sc):
    """F[m][m'] (k_m x k_m') main effects of module m's template overlaps along module m' phases."""
    F, R2 = {}, {}
    for m, (o, k) in enumerate(zip(sc.offsets, sc.sizes)):
        T = sc.W_gh[o:o + k] @ sc.H                                   # (k, Npos)
        grand = T.mean(1, keepdims=True)
        fit = np.repeat(grand, sc.addresses, 1)
        for mp, kp in enumerate(sc.sizes):
            onehot = np.eye(kp)[sc.phases[:, mp]]                     # (Npos, kp)
            F[m, mp] = (T @ onehot) / onehot.sum(0) - grand           # (k, kp) marginal means - grand
            fit = fit + F[m, mp] @ onehot.T
        resid = T - fit
        R2[m] = 1 - (resid - resid.mean(1, keepdims=True)).var() / (T - grand).var()
    return F, R2


def trajectory(seed):
    Cc = factored(Ns, P, np.random.default_rng(10_000 * seed + P)); S = Cc.patterns
    sc = Scaffold(seed=seed); alpha = mmse_alpha(P, flip_rate=f)
    labels = np.zeros((P, 3), dtype=np.int64); taken = np.zeros(sc.sizes, dtype=bool)
    where = np.zeros(P, dtype=np.int64)
    ph = next(c for c in place._candidates(sc, (0, 0, 0)) if not taken[c]); taken[ph] = True
    where[0] = sc.index(np.array([ph]))[0]
    K = np.array([[1.0 / (S[:, 0] @ S[:, 0] + alpha)]])
    recs = []
    for n in range(1, P):
        Sn = S[:, :n]; s = S[:, n]; b = Sn.T @ s; c = K @ b; r = s @ s + alpha - b @ c
        masks = [np.bincount(labels[:n, m], minlength=k) >= CAP[m] for m, k in enumerate(GROUPS)]
        Ga = sc.G[:, where[:n]]
        u = sc.H[:, where[:n]] @ c
        recs.append((Ga @ c, sc.W_gh @ u, sc.W_gh @ relu(u), [mm.copy() for mm in masks], c.sum(), np.linalg.norm(c)))
        gi = sc.W_gh @ relu(u); pick = []
        for m, k in enumerate(GROUPS):
            inp = gi[sc.offsets[m]: sc.offsets[m] + sc.sizes[m]].copy()
            if m == 2: inp = inp.reshape(k, place.SLOTS).sum(1)
            inp[masks[m]] = -np.inf; pick.append(int(inp.argmax()))
        labels[n] = pick
        ph = next(cc for cc in place._candidates(sc, tuple(pick)) if not taken[cc]); taken[ph] = True
        where[n] = sc.index(np.array([ph]))[0]
        K = np.block([[K + np.outer(c, c) / r, -c[:, None] / r], [-c[None, :] / r, np.array([[1.0 / r]])]])
    return sc, recs


def pick(vec, m, sc, mask):
    inp = vec.copy()
    if m == 2: inp = inp.reshape(GROUPS[2], place.SLOTS).sum(1)
    inp[mask] = -np.inf
    return int(inp.argmax())


if __name__ == "__main__":
    import sys
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    for seed in seeds:
        sc, recs = trajectory(seed)
        F, R2 = anova(sc)
        if seed == seeds[0]:
            for m in range(3):
                beta = np.mean(np.diag(F[m, m])) * sc.sizes[m] / (sc.sizes[m] - 1)
                own = F[m, m] - beta * (np.eye(sc.sizes[m]) - 1 / sc.sizes[m])
                cross = [np.sqrt((F[m, mp] ** 2).mean()) / beta for mp in range(3) if mp != m]
                print(f"module {m}: main effects explain {R2[m]:.3f} of template variance; beta {beta:.3f}; "
                      f"own-module deviation rms/beta {np.sqrt((own**2).mean())/beta:.3f}; "
                      f"cross-module rms/beta {np.round(cross, 3)}")
        tallies = {k: np.zeros(3) for k in ("hebb_noReLU", "within", "within+cross", "within+cross+mean", "snap(ReLU)")}
        for gc, hebb, snapin, masks, csum, cnorm in recs:
            C = [gc[sc.offsets[m]: sc.offsets[m] + sc.sizes[m]] for m in range(3)]
            for m in range(3):
                truth = pick(C[m], m, sc, masks[m])
                within = F[m, m] @ C[m]
                cross = sum(F[m, mp] @ C[mp] for mp in range(3))
                tallies["hebb_noReLU"][m] += pick(hebb[sc.offsets[m]: sc.offsets[m] + sc.sizes[m]], m, sc, masks[m]) == truth
                tallies["snap(ReLU)"][m] += pick(snapin[sc.offsets[m]: sc.offsets[m] + sc.sizes[m]], m, sc, masks[m]) == truth
                tallies["within"][m] += pick(within, m, sc, masks[m]) == truth
                tallies["within+cross"][m] += pick(cross, m, sc, masks[m]) == truth
        n = len(recs)
        print(f"seed {seed}: agreement with the address-greedy choice, per module")
        for k, v in tallies.items():
            if k == "within+cross+mean": continue
            print(f"   {k:14} " + " ".join(f"{x / n:.3f}" for x in v))
