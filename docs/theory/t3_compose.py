"""Section 3: composition at an empty address. Aligned (oracle) placement; read-out sign(W_sh h_new)
at addresses (a, b, slot in group c) whose (a, b, c) cell holds no stored item; overlap with the
composite sign(A[a] + B[b] + C[c]). Predictions (3.2)-(3.3) from the isotropic leverage and from w.

    python t3_compose.py
"""
from common import *
from scipy.stats import norm

KAPPA = np.sqrt(2 / np.pi) / 2           # 0.399: linear part of E[s | X], X ~ N(0, 3)
NOISE = 1 - 3 * KAPPA ** 2               # 0.523: everything but the linear signal


def omega(v, draws=400_000, rng=np.random.default_rng(0)):
    """(3.3): E sign(m + eta) sign(X), m = sum_f erf(x_f / sqrt 6), eta ~ N(0, v)."""
    x = rng.standard_normal((draws, 3))
    m = sum(__import__("scipy.special", fromlist=["erf"]).erf(x[:, j] / np.sqrt(6)) for j in range(3))
    return float(np.mean(np.sign(m + np.sqrt(v) * rng.standard_normal(draws)) * np.sign(x.sum(1))))


def w_prediction(w, factors, cell):
    """Gaussian prediction of the overlap from the smoother weights w alone (no content values)."""
    a, b, c = cell
    om = [np.bincount(factors[:, j], weights=w, minlength=k) for j, k in enumerate((9, 16, 5))]
    cov = KAPPA * (om[0][a] + om[1][b] + om[2][c])
    var = KAPPA ** 2 * sum((o ** 2).sum() for o in om) + NOISE * (w ** 2).sum()
    rho = cov / np.sqrt(3 * var)
    return 2 / np.pi * np.arcsin(np.clip(rho, -1, 1))


def run(seed, P, n_test=150, slot_policy='random'):
    content = factored(DIM, P, np.random.default_rng(10_000 * seed + P))
    S, F = content.patterns, content.factors
    A, B, Cc = content.codes
    grid = Scaffold(seed=seed); flat = Scaffold(periods=(60,), seed=seed)
    where = place.oracle(grid, F)
    rand_where = place.scattered(P, grid, np.random.default_rng(seed + 1))
    stored_cells = set(map(tuple, F))
    taken = np.zeros(grid.addresses, dtype=bool); taken[where] = True
    ph = grid.phases[where]
    used = [set(ph[:, m]) for m in range(3)]
    rng = np.random.default_rng(500 + seed)
    tests = []
    for a in range(9):
        for b in range(16):
            for c in range(5):
                if (a, b, c) in stored_cells:
                    continue
                for slot in (rng.permutation(5) if slot_policy == 'random' else range(5)):
                    addr = grid.index(np.array([[a, b, c * 5 + slot]]))[0]
                    if not taken[addr] and c * 5 + slot in used[2] and a in used[0] and b in used[1]:
                        tests.append(((a, b, c), addr)); break
    rng.shuffle(tests); tests = tests[:n_test]
    Wsh = {"grid aligned": S @ np.linalg.pinv(grid.H[:, where]),
           "grid random": S @ np.linalg.pinv(grid.H[:, rand_where]),
           "flat": S @ np.linalg.pinv(flat.H[:, where])}
    Hpinv = np.linalg.pinv(grid.H[:, where])
    means = [np.stack([S[:, F[:, j] == v].mean(1) if (F[:, j] == v).any() else np.zeros(DIM)
                       for v in range(k)], 1) for j, k in enumerate((9, 16, 5))]
    sbar = S.mean(1)
    out = {k: [] for k in ("grid aligned", "grid random", "flat", "kNN 2-factor item", "kNN mean of 2-factor items",
                           "additive model", "mean content", "pred w-based", "L_new")}
    for (a, b, c), addr in tests:
        target = np.sign(A[:, a] + B[:, b] + Cc[:, c])
        for name, W in Wsh.items():
            sc_ = flat if name == "flat" else grid
            out[name].append(np.mean(np.sign(W @ sc_.H[:, addr]) * target))
        share = (F == np.array([a, b, c])).sum(1)
        two = np.where(share == 2)[0]
        out["kNN 2-factor item"].append(np.mean(np.sign(S[:, rng.choice(two)]) * target) if len(two) else np.nan)
        out["kNN mean of 2-factor items"].append(np.mean(np.sign(S[:, two].mean(1)) * target) if len(two) else np.nan)
        add = means[0][:, a] + means[1][:, b] + means[2][:, c] - 2 * sbar
        out["additive model"].append(np.mean(np.sign(add) * target))
        out["mean content"].append(np.mean(np.sign(sbar) * target))
        w = Hpinv @ grid.H[:, addr]
        out["pred w-based"].append(w_prediction(w, F, (a, b, c)))
        out["L_new"].append(float(w @ w))
    return {k: float(np.nanmean(v)) for k, v in out.items()}, len(tests)


if __name__ == "__main__":
    Nh = 400
    print("isotropic predictions Omega(0.523 * Nh/(P-Nh-1)) and the additive model's Omega(0.038):")
    for P in (600, 800, 900):
        print(f"   P {P}: L_iso {Nh / (P - Nh - 1):.2f}  Omega {omega(NOISE * Nh / (P - Nh - 1)):.3f}")
    print(f"   additive: Omega {omega(1/89 + 1/50 + 1/160):.3f};  kNN 2-factor (2/pi)asin(2/sqrt12) = "
          f"{2/np.pi*np.arcsin(2/np.sqrt(12)):.3f};  mean floor (2/pi)asin(0.353) = {2/np.pi*np.arcsin(np.sqrt(0.373/3)):.3f}")
    for policy in ("random", "first-free (most-used slot)"):
        print(f"--- empty address = (a, b, {policy} slot of group c)")
        for P in (200, 400, 600, 800, 900):
            rows = [run(seed, P, slot_policy=policy.split()[0]) for seed in (0, 1, 2)]
            keys = rows[0][0].keys()
            avg = {k: np.mean([r[0][k] for r in rows]) for k in keys}
            print(f"P {P} ({rows[0][1]} cells/seed): " + "  ".join(f"{k} {v:.3f}" for k, v in avg.items())
                  + f"  | Omega(0.523 L_new) {omega(NOISE * avg['L_new'], draws=100_000):.3f}", flush=True)
