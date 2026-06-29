"""Scaffold-only quantities the theory needs (no content, no cues).

(a) place-code correlation vs shared phases q: Hermite prediction (eq 5.1) vs measured, M = 3 and 4;
(b) the grid-input templates t_mk: how far W_gh restricted to place codes is from beta Q^T + const;
(c) whether the one-hot grid code is linear in the place code: ||G - (G H^+) H|| / ||G||.
"""
from common import *
from math import factorial
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermeval


def rho_h(rho, M, p=0.6699, theta=0.5, nmax=40):
    """Centred correlation of ReLU(z - theta) for pre-activations with correlation rho (eq 5.1)."""
    s = np.sqrt(M * p); t = theta / s
    a1 = norm.sf(t) ** 2
    coef = [norm.pdf(t) ** 2 * hermeval(t, [0] * (n - 2) + [1]) ** 2 / factorial(n) for n in range(2, nmax)]
    num = a1 * rho + sum(c * rho ** n for c, n in zip(coef, range(2, nmax)))
    var = (1 + t * t) * norm.sf(t) - t * norm.pdf(t) - (norm.pdf(t) - t * norm.sf(t)) ** 2
    return num / var, a1 / var


def measured_corr(periods, seed=0, pairs=4000):
    """Pearson correlation between place codes of address pairs sharing exactly q phases."""
    rng = np.random.default_rng(1)
    sizes = [p * p for p in periods]; M = len(sizes)
    sc = Scaffold(periods=periods, seed=seed) if np.prod(sizes) <= 5000 else None
    W = sc.W_hg if sc is not None else light_projection(periods, seed)
    offs = np.cumsum([0] + sizes[:-1])
    out = {}
    for q in range(M + 1):
        cs = []
        for _ in range(pairs // (M + 1)):
            a = np.array([rng.integers(s) for s in sizes]); b = a.copy()
            diff = rng.choice(M, M - q, replace=False)
            for m in diff:
                b[m] = (a[m] + 1 + rng.integers(sizes[m] - 1)) % sizes[m]
            ha = relu(W[:, offs + a].sum(1), 0.5); hb = relu(W[:, offs + b].sum(1), 0.5)
            cs.append(np.corrcoef(ha, hb)[0, 1])
        out[q] = float(np.mean(cs))
    return out


def light_projection(periods, seed):
    """W_hg exactly as Scaffold._projection draws it, without building the address tables."""
    rng = np.random.default_rng(seed)
    Ng = sum(p * p for p in periods); Nh = 400
    w = rng.standard_normal((Nh, Ng))
    cut = int(0.4 * Nh * Ng)
    rows = rng.integers(0, Nh, size=cut); cols = rng.integers(0, Ng, size=cut)
    mask = np.ones_like(w); mask[rows, cols] = 0.0
    return w * mask


if __name__ == "__main__":
    for periods in ((3, 4, 5), (3, 4, 5, 7)):
        M = len(periods)
        pred = [rho_h(q / M, M)[0] for q in range(M + 1)]
        meas = measured_corr(periods)
        print(f"M={M} q: " + "  ".join(f"{q}: pred {pred[q]:.3f} meas {meas[q]:.3f}" for q in range(M + 1)),
              f"  (linear share a1^2/var = {rho_h(0.5, M)[1]:.3f})")
    # (b), (c) on the default scaffold
    sc = Scaffold(seed=0)
    Hall, G = sc.H, sc.G
    D = G @ np.linalg.pinv(Hall)
    print("||G - G H^+ H|| / ||G - mean|| =", np.linalg.norm(G - D @ Hall) / np.linalg.norm(G - G.mean(1, keepdims=True)))
    for m, (o, k) in enumerate(zip(sc.offsets, sc.sizes)):
        T = sc.W_gh[o:o + k] @ Hall                  # (k, Npos): template overlap with every address
        on = T[sc.phases[:, m], np.arange(sc.addresses)]
        Tc = T - T.mean(0, keepdims=True)           # remove the address-wide constant
        beta = np.mean(Tc[sc.phases[:, m], np.arange(sc.addresses)]) * k / (k - 1)
        ideal = beta * (np.eye(k)[sc.phases[:, m]].T - 1 / k)
        resid = Tc - ideal
        # how much of the residual is explained by a phase-only diagonal (template norms) vs the rest
        diag_by_k = np.array([Tc[kk, sc.phases[:, m] == kk].mean() for kk in range(k)])
        print(f"module {m}: on-phase overlap mean {on.mean():.4f}, beta {beta:.4f}; "
              f"residual/beta rms {np.sqrt((resid**2).mean())/beta:.3f}; "
              f"on-phase diag CV across k {diag_by_k.std()/diag_by_k.mean():.3f}")
