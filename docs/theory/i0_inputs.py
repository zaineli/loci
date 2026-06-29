"""Theory inputs for 'the price of imagination' (no recall is simulated here).

(1) Analytic phase sums of a factor-only cue's recall (Prop I.1): theta = (kf/k1) (N + eps I)^-1 e,
    N = M^T M the factor co-occurrence counts, eps = (Ns(1 - 3 k1^2) + alpha) / (Ns k1^2);
    per-module contrast of the target phase over the best competitor, module 2 at slot level.
(2) Slot occupancy omega = (1 - e^-mu)/mu of the most-used slot, mu = items per (a, b, c) cell.
(3) Template distortion scale of the Hebbian snap per unit phase sum (ANOVA of W_gh H), Nh = 400, 800.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *
import compose
from t1b_margin import anova

K1 = np.sqrt(2 / np.pi) / 2            # stored item: sign(X + z), var 4
KF = np.sqrt(2 / np.pi) / np.sqrt(3)   # factor cue: sign(X), var 3
CARDS = (9, 16, 5)


def analytic_theta(F, P, alpha, Ns=1000):
    M = np.hstack([np.eye(k)[F[:, f]] for f, k in enumerate(CARDS)])       # (P, 30)
    N = M.T @ M
    eps = (Ns * (1 - 3 * K1 ** 2) + alpha) / (Ns * K1 ** 2)
    return M, N, eps


def predicted_phase_sums(M, N, eps, q, where_phases):
    """Phase sums C^(m) = Q_m^T M theta for the factor cue q = (a, b, c); module 2 at 25 phases."""
    e = np.zeros(30); e[q[0]] = 1; e[9 + q[1]] = 1; e[25 + q[2]] = 1
    theta = (KF / K1) * np.linalg.solve(N + eps * np.eye(30), e)
    c = M @ theta                                                             # per-item coefficient
    return [np.bincount(where_phases[:, m], weights=c, minlength=k) for m, k in enumerate((9, 16, 25))], c


if __name__ == "__main__":
    for P in (400, 800):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(31_000 * seed + P + 7 * sum(CARDS))
            held = compose.holdout(CARDS, rng)
            C = compose.make_factored(P, rng, CARDS, held)
            grid = Scaffold(seed=seed)
            where = place.oracle(grid, C.factors); ph = grid.phases[where]
            alpha = mmse_alpha(P, flip_rate=0.1)
            M, N, eps = analytic_theta(C.factors, P, alpha)
            q = np.array([(a, b, c) for a, b in held for c in range(5)])
            margins = []
            for row in q:
                sums, _ = predicted_phase_sums(M, N, eps, row, ph)
                m0 = sums[0][row[0]] - np.delete(sums[0], row[0]).max()
                m1 = sums[1][row[1]] - np.delete(sums[1], row[1]).max()
                grp = np.arange(25) // 5 == row[2]
                m2 = sums[2][grp].max() - sums[2][~grp].max()
                margins.append((m0, m1, m2, sums[2][grp].max()))
            margins = np.array(margins)
            cells = len({tuple(x) for x in C.factors})
            mu = P / (9 * 16 * 5 - len(held) * 5)
            occ = np.bincount(ph[:, 2] % 5, minlength=5) / P
            print(f"P {P} seed {seed}: eps {eps:.2f}; predicted contrasts (mean, min over 80 held queries): "
                  f"m0 {margins[:,0].mean():.3f}/{margins[:,0].min():.3f}  m1 {margins[:,1].mean():.3f}/{margins[:,1].min():.3f}  "
                  f"m2(slot) {margins[:,2].mean():.3f}/{margins[:,2].min():.3f} (best group-c slot sum {margins[:,3].mean():.3f});  "
                  f"slot occupancy {np.round(occ, 3)}  omega=(1-e^-mu)/mu {(1-np.exp(-mu))/mu:.3f} (mu {mu:.2f})")
    for Nh in (400, 800):
        rms = []
        for seed in (0, 1, 2):
            sc = Scaffold(seed=seed, place_cells=Nh)
            F, R2 = anova(sc)
            row = []
            for m in range(3):
                k = sc.sizes[m]
                beta = np.mean(np.diag(F[m, m])) * k / (k - 1)
                own = F[m, m] - beta * (np.eye(k) - 1 / k)
                row.append([np.sqrt((own ** 2).mean()) / beta] + [np.sqrt((F[m, mp] ** 2).mean()) / beta for mp in range(3) if mp != m]
                           + [1 - R2[m]])
            rms.append(row)
        rms = np.array(rms).mean(0)
        print(f"Nh {Nh}: template deviation per unit phase sum, module m: [own, cross1, cross2, interaction variance share]")
        for m in range(3):
            print(f"   module {m}: " + " ".join(f"{x:.3f}" for x in rms[m]))
