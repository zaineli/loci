"""Why the error law alone (level E) overestimates random placement: per-module phase-input noise, in
units of the on-phase gap beta_m, split into own-module (error law), cross-module leak, and interactions.
x_m = T_m c~, T_m = W_gh,m H_a = F^{mm} Q_m^T + sum_{m'!=m} F^{mm'} Q_m'^T + I_m (ANOVA, t1b)."""
from common import *
from t1b_margin import anova
for placement in ("random", "oracle"):
    for seed in (0, 1, 2):
        content, grid, alpha, cues = bench_condition(seed, 0.8, 0.1)
        where = placements(content, grid, seed, which=(placement,))[placement]
        S = content.patterns; P = S.shape[1]; s2 = 0.36
        K = np.linalg.inv(S.T @ S + alpha * np.eye(P)); covc = s2 * (K - alpha * K @ K)
        F, _ = anova(grid); ph = grid.phases[where]
        Qs = [np.eye(k)[ph[:, m]] for m, k in enumerate(grid.sizes)]
        parts = []
        for m, (o, k) in enumerate(zip(grid.offsets, grid.sizes)):
            beta = np.mean(np.diag(F[m, m])) * k / (k - 1)
            T = grid.W_gh[o:o + k] @ grid.H[:, where]
            own = F[m, m] @ Qs[m].T
            cross = sum(F[m, mp] @ Qs[mp].T for mp in range(3) if mp != m)
            inter = T - T.mean(0, keepdims=True) - (own - own.mean(0, keepdims=True)) - (cross - cross.mean(0, keepdims=True))
            v = lambda A: np.mean(np.diag((A - A.mean(0, keepdims=True)) @ covc @ (A - A.mean(0, keepdims=True)).T)) / beta ** 2
            ideal = beta * (Qs[m].T - 1 / k)
            parts.append((v(ideal), v(own), v(cross), v(inter), v(T)))
        print(f"{placement:7} seed {seed}: per module [ideal Q (=E), own templates, cross-module leak, interactions, total T] noise var / beta^2:")
        for m, p in enumerate(parts):
            print(f"      module {m}: " + "  ".join(f"{x:.4f}" for x in p))
