"""Check of Props I.1-I.2: does the factor-only cue for a held-out (a, b) land on (a, b, C-group)?

Content and holdout as gold2/compose (16 held pairs x 5 c = 80 queries), oracle placement.
Per (P, Nh, seed): analytic phase-sum contrasts (I.1-I.3) vs exact; landing by
  greedy    per-module argmax of the exact phase sums C^(m) (no scaffold)
  anova     the additive template model sum_m' F^{mm'} C^(m') (scaffold ANOVA, no ReLU)
  hebb      W_gh H_a c (no ReLU)
  snap      the model's own snap of ReLU(H_a c)  [compose measured 0.963 at 800/400]
and with 10% flips: T-mc (Gaussian c~, exact ReLU, 200 draws per query) vs 20 real flip draws.
pinv (alpha = 0), clean, at the end.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *
import compose
from t1b_margin import anova
from i0_inputs import analytic_theta, predicted_phase_sums, CARDS

F_CACHE = {}


def landing(ph, q):
    """ph (n, 3) phases landed; q (n, 3) queries -> per-module match (module 2 by group)."""
    return np.stack([ph[:, 0] == q[:, 0], ph[:, 1] == q[:, 1], ph[:, 2] // 5 == q[:, 2]], 1)


def pick(x, sc):
    return np.stack([x[o:o + k].argmax(0) for o, k in zip(sc.offsets, sc.sizes)], 1)


def run(seed, P, Nh, draws=200, flips=20):
    rng = np.random.default_rng(31_000 * seed + P + 7 * sum(CARDS))
    held = compose.holdout(CARDS, rng)
    C = compose.make_factored(P, rng, CARDS, held)
    S = C.patterns
    grid0 = Scaffold(seed=seed); sc = Scaffold(seed=seed, place_cells=Nh)
    where = place.oracle(grid0, C.factors); ph = grid0.phases[where]
    q = np.array([(a, b, c) for a, b in held for c in range(5)])
    cues = compose.composite(C.codes, q)
    alpha = mmse_alpha(P, flip_rate=0.1)
    K = np.linalg.inv(S.T @ S + alpha * np.eye(P))
    c = K @ (S.T @ cues)                                      # (P, n)
    G = sc.G[:, where]                                         # exact phase sums: G c
    sums = G @ c
    out = {}
    out["greedy"] = landing(pick(sums, sc), q)
    if (seed, Nh) not in F_CACHE:
        F_CACHE[seed, Nh] = anova(sc)[0]
    F = F_CACHE[seed, Nh]
    xa = np.vstack([sum(F[m, mp] @ sums[sc.offsets[mp]:sc.offsets[mp] + sc.sizes[mp]] for mp in range(3)) for m in range(3)])
    out["anova"] = landing(pick(xa, sc), q)
    Ha = sc.H[:, where]
    out["hebb"] = landing(pick(sc.W_gh @ (Ha @ c), sc), q)
    out["snap"] = landing(sc.snap(sc.W_gh @ relu(Ha @ c)), q)
    # analytic vs exact contrasts
    M, N, eps = analytic_theta(C.factors, P, alpha)
    dev = []
    for n_, row in enumerate(q):
        pred, _ = predicted_phase_sums(M, N, eps, row, ph)
        ex = [sums[sc.offsets[m]:sc.offsets[m] + sc.sizes[m], n_] for m in range(3)]
        grp = np.arange(25) // 5 == row[2]
        cp = [pred[0][row[0]] - np.delete(pred[0], row[0]).max(), pred[1][row[1]] - np.delete(pred[1], row[1]).max(),
              pred[2][grp].max() - pred[2][~grp].max()]
        ce = [ex[0][row[0]] - np.delete(ex[0], row[0]).max(), ex[1][row[1]] - np.delete(ex[1], row[1]).max(),
              ex[2][grp].max() - ex[2][~grp].max()]
        dev.append((cp, ce))
    dev = np.array(dev)                                        # (n, 2, 3)
    # 10% flips: T-mc (Gaussian) and real flips
    a, s2 = 0.8, 0.36
    covu = s2 * Ha @ (K - alpha * K @ K) @ Ha.T
    w, V = np.linalg.eigh(covu); Lu = V * np.sqrt(np.clip(w, 0, None))
    mu = a * (Ha @ c)                                          # (Nh, n)
    Z = np.random.default_rng(seed + 11).standard_normal((draws, Nh)) @ Lu.T
    tm = np.zeros(3); tj = 0.0
    for n_ in range(len(q)):
        x = relu(mu[:, n_][None] + Z) @ sc.W_gh.T               # (draws, 50)
        hit = landing(pick(x.T, sc), np.repeat(q[n_][None], draws, 0))
        tm += hit.mean(0); tj += hit.all(1).mean()
    out["T-mc flips (module)"] = tm / len(q); out["T-mc flips"] = tj / len(q)
    real = []
    for d in range(flips):
        noisy = flip(cues, 0.1, np.random.default_rng(1000 * seed + d))
        cc = K @ (S.T @ noisy)
        real.append(landing(sc.snap(sc.W_gh @ relu(Ha @ cc)), q))
    out["real flips"] = np.array(real)                       # (flips, n, 3)
    # pinv, clean
    cp_ = np.linalg.pinv(S) @ cues
    out["pinv"] = landing(sc.snap(sc.W_gh @ relu(Ha @ cp_)), q)
    Ftil = sc.W_gh[sc.offsets[2]:] @ Ha                             # module-2 template overlaps (25, P)
    lin = sc.W_gh[sc.offsets[2]:] @ (Ha @ c)                           # Hebbian input, module 2
    main = xa[2]
    inter = lin - lin.mean(0) - (main - main.mean(0)) * (np.std(lin - lin.mean(0)) / np.std(main - main.mean(0)))
    out["m2 margin"] = dev[:, 1, 2]
    return out, dev


if __name__ == "__main__":
    for P, Nh in ((800, 400), (800, 800), (400, 400), (400, 800)):
        res = [run(s, P, Nh) for s in (0, 1, 2)]
        print(f"=== P {P}, Nh {Nh}")
        dev = np.concatenate([r[1] for r in res])
        print("   contrasts, analytic (I.1-I.3) vs exact phase sums, mean over queries: "
              + "  ".join(f"m{m}: {dev[:, 0, m].mean():.3f} vs {dev[:, 1, m].mean():.3f} (corr {np.corrcoef(dev[:, 0, m], dev[:, 1, m])[0, 1]:+.2f})" for m in range(3)))
        for key in ("greedy", "anova", "hebb", "snap", "real flips", "pinv"):
            v = np.mean([r[0][key].mean(0) if r[0][key].ndim == 2 else r[0][key] for r in res], 0)
            j = np.mean([r[0][key].all(1).mean() if r[0][key].ndim == 2 else np.nan for r in res])
            if key == "real flips":
                v = np.mean([r[0][key].mean((0, 1)) for r in res], 0); j = np.mean([r[0][key].all(2).mean() for r in res])
            print(f"   {key:12} landing {j:.3f}   per module {np.round(v, 3)}")
        marg = np.concatenate([r[0]["m2 margin"] for r in res]); fail = np.concatenate([~r[0]["snap"][:, 2] for r in res])
        if fail.any():
            pct = [np.mean(marg < m) for m in marg[fail]]
            print(f"   snap failures: {fail.sum()} of {len(fail)}; their exact module-2 margin percentile: mean {np.mean(pct):.2f} (0 = smallest); "
                  f"failures' margin {marg[fail].mean():.3f} vs all {marg.mean():.3f}")
        tj = np.mean([r[0]["T-mc flips"] for r in res]); tm = np.mean([r[0]["T-mc flips (module)"] for r in res], 0)
        print(f"   {'T-mc flips':12} landing {tj:.3f}   per module {np.round(tm, 3)}   (prediction for 'real flips')", flush=True)
