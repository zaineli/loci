"""(2.4) at a single-move fixed point: every improving swap has K_ij > 0 or a capacity-blocked single move."""
from t2_replay import *
for seed in (0, 1, 2):
    Cc = factored(Ns, P, np.random.default_rng(10_000 * seed + P)); S = Cc.patterns
    alpha = mmse_alpha(P, flip_rate=f); K = place.precision(S, alpha)
    lab0, _ = online(S, alpha, "greedy")
    for slack, name in ((1.1, "cap 1.1P/k"), (100.0, "no cap")):
        rng = np.random.default_rng(seed)
        labs = lab0.copy()
        for m, k in enumerate(GROUPS):
            cap = int(np.ceil(slack * P / k)); cp = Coupling(K, labs[:, m].copy(), k)
            for _ in range(300):
                moved = 0
                for i in rng.permutation(P):
                    d = cp.delta(i); d[(cp.count >= cap) & (np.arange(k) != cp.lab[i])] = np.inf
                    b = int(d.argmin())
                    if d[b] < -1e-15: cp.move(i, b); moved += 1
                if not moved: break
            lab = cp.lab; Q = np.eye(k)[lab]; Cp = K @ Q; dg = np.diag(K)
            dmove = 2 * (Cp - (Cp[np.arange(P), lab] - dg)[:, None])          # dmove[i, b]
            dij = dmove[:, lab]                                                # i -> group of j
            dji = dij.T
            swap = dij + dji - 4 * K
            diff = lab[:, None] != lab[None, :]
            imp = diff & (swap < -1e-12)
            blocked = (cp.count[lab][None, :] >= cap) | (cp.count[lab][:, None] >= cap)
            explained = (K > 0) | blocked | (dij < 0) | (dji < 0)
            print(f"seed {seed} module {m} {name:10}: improving swaps {imp.sum() // 2:6d}; of them with K_ij>0 "
                  f"{(imp & (K > 0)).sum() // 2:6d}, capacity-blocked {(imp & blocked & ~(K > 0)).sum() // 2:5d}, unexplained "
                  f"{(imp & ~explained).sum() // 2}")
