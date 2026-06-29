"""Where level E loses: mean margin (true phase minus best competitor) under E and T-lin, split into the
item's own template column (a T e_i) and the bias through the templates (-a alpha T K e_i)."""
from common import *
content, grid, alpha, cues = bench_condition(0, 0.8, 0.1)
for placement in ("random", "oracle"):
    where = placements(content, grid, 0, which=(placement,))[placement]
    S = content.patterns; P = S.shape[1]; a = 0.8; s2 = 0.36
    K = np.linalg.inv(S.T @ S + alpha * np.eye(P)); covc = s2 * (K - alpha * K @ K)
    ph = grid.phases[where]
    for m, (o, k) in enumerate(zip(grid.offsets, grid.sizes)):
        Q = np.eye(k)[ph[:, m]].T                      # (k, P)
        T = grid.W_gh[o:o + k] @ grid.H[:, where]
        beta = np.mean([T[ph[i, m], i] - np.delete(T[:, i], ph[i, m]).mean() for i in range(P)])
        out = {}
        for name, A in (("E", Q * beta), ("T", T)):
            own = a * A                                 # column i: a A e_i
            bias = -a * alpha * A @ K                   # column i: -a alpha A K e_i
            noise = np.sqrt(np.diag(A @ covc @ A.T)).mean()
            def margin(M):
                tr = M[ph[:, m], np.arange(P)]
                M2 = M.copy(); M2[ph[:, m], np.arange(P)] = -np.inf
                return tr - M2.max(0)
            out[name] = (np.median(margin(own + bias)) / beta, np.mean(margin(own + bias) < 0),
                         np.median(margin(own)) / beta, noise / beta)
        print(f"{placement:7} module {m}: " + " | ".join(f"{n}: median margin {v[0]:.3f} (own only {v[2]:.3f}), "
              f"frac mean-wrong {v[1]:.3f}, noise sd {v[3]:.3f}" for n, v in out.items()))
