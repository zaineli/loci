"""Address-level online encoding and replay (eq 2.7): the error law on REALIZED phases, all three
modules at full phase resolution (9, 16, 25), every tuple holding at most one item.

    python t2c_address.py [seed ...]
"""
from t2_replay import *

SIZES = (9, 16, 25)


def realized_J(K, phases):
    return sum(J_of(K, phases[:, [m]]) for m in range(3))


def online_addr(S, alpha, sc, rule="greedy"):
    """Each arrival goes to the free address maximising sum_m C^(m) at its phases (caps ceil(P/k_m))."""
    cap = [int(np.ceil(P / k)) for k in SIZES]
    taken = np.zeros(SIZES, dtype=bool)
    ph = np.zeros((P, 3), dtype=np.int64)
    taken[0, 0, 0] = True
    K = np.array([[1.0 / (S[:, 0] @ S[:, 0] + alpha)]])
    for n in range(1, P):
        Sn = S[:, :n]; s = S[:, n]; b = Sn.T @ s; c = K @ b; r = s @ s + alpha - b @ c
        score = np.zeros(SIZES); pen = np.zeros(SIZES)
        for m, k in enumerate(SIZES):
            count = np.bincount(ph[:n, m], minlength=k)
            C = np.bincount(ph[:n, m], weights=c, minlength=k)
            if rule == "centred":
                C = C - count * c.sum() / n
            shape = [1, 1, 1]; shape[m] = k
            score = score + C.astype(float).reshape(shape)
            pen = pen + (count >= cap[m]).astype(float).reshape(shape)
        score[pen > 0] -= 1e6                       # caps are soft only when no capped-free tuple is left
        score[taken] = -np.inf
        a = np.unravel_index(int(score.argmax()), SIZES)
        taken[a] = True; ph[n] = a
        K = np.block([[K + np.outer(c, c) / r, -c[:, None] / r], [-c[None, :] / r, np.array([[1.0 / r]])]])
    return ph


def replay_addr(K, ph, rng, sweeps=50, T0=0.0, T1=0.0, zero_sweeps=0, slack=1.1):
    """Coordinate descent (T = 0) or Gibbs (T > 0) on the realized law over free addresses."""
    ph = ph.copy()
    cap = [int(np.ceil(slack * P / k)) for k in SIZES]
    Cp = [K @ np.eye(k)[ph[:, m]] for m, k in enumerate(SIZES)]
    count = [np.bincount(ph[:, m], minlength=k) for m, k in enumerate(SIZES)]
    taken = np.zeros(SIZES, dtype=bool); taken[tuple(ph.T)] = True
    diag = np.diag(K)
    temps = (list(np.geomspace(T0, T1, sweeps)) if T0 > 0 else [0.0] * sweeps) + [0.0] * zero_sweeps
    for T in temps:
        moved = 0
        for i in rng.permutation(P):
            cur = tuple(ph[i])
            score = np.zeros(SIZES)
            for m, k in enumerate(SIZES):
                d = 2 * (Cp[m][i] - (Cp[m][i, cur[m]] - diag[i]))
                d[cur[m]] = 0.0
                d[(count[m] >= cap[m]) & (np.arange(k) != cur[m])] = np.inf
                shape = [1, 1, 1]; shape[m] = k
                score = score + d.reshape(shape)
            score[taken] = np.inf; score[cur] = 0.0
            if T > 0:
                flat = score.ravel(); w = np.exp(-(flat - flat[np.isfinite(flat)].min()) / T)
                w[~np.isfinite(flat)] = 0
                new = np.unravel_index(int(rng.choice(flat.size, p=w / w.sum())), SIZES)
            else:
                new = np.unravel_index(int(score.argmin()), SIZES)
                if not score[new] < -1e-15: new = cur
            if new != cur:
                taken[cur] = False; taken[new] = True
                for m in range(3):
                    if new[m] != cur[m]:
                        Cp[m][:, cur[m]] -= K[:, i]; Cp[m][:, new[m]] += K[:, i]
                        count[m][cur[m]] -= 1; count[m][new[m]] += 1
                ph[i] = new; moved += 1
        if T == 0 and moved == 0:
            break
    return ph


if __name__ == "__main__":
    import sys
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    for seed in seeds:
        t0 = time.time()
        Cc = factored(Ns, P, np.random.default_rng(10_000 * seed + P)); S = Cc.patterns
        sc = Scaffold(seed=seed); alpha = mmse_alpha(P, flip_rate=f)
        cues = flip(S, f, np.random.default_rng(7 + seed)); K = place.precision(S, alpha)
        rng = np.random.default_rng(100 + seed)

        def report(name, ph=None, lab=None):
            where = sc.index(ph) if ph is not None else place.addresses(sc, lab)
            ph = sc.phases[where]
            assert len(np.unique(where)) == P
            mem = Memory(sc, rule="ridge", alpha=alpha); mem.store(S, where)
            got = mem.recall(cues)
            acc = (got.phases == ph).mean(0)
            g2 = ph[:, 2] // 5
            nm = [nmi(ph[:, 0], Cc.factors[:, 0]), nmi(ph[:, 1], Cc.factors[:, 1]), nmi(ph[:, 2], Cc.factors[:, 2])]
            print(f"{seed} {name:34} recall {np.mean(got.address == where):.3f} mod-acc {np.round(acc, 3)} "
                  f"J_addr {realized_J(K, ph):.3f}  NMI m0:A {nm[0]:.2f} m1:B {nm[1]:.2f} m2(25):C {nm[2]:.2f} "
                  f"NMI(m0,m1) {nmi(ph[:, 0], ph[:, 1]):.2f}", flush=True)

        lab0, _ = online(S, alpha, "greedy"); T = typical_dJ(K, lab0)
        report("label online greedy", lab=lab0)
        report("label + replay 8", lab=replay(K, lab0, rng, sweeps=8))
        report("label + annealed Gibbs", lab=replay(K, lab0, rng, sweeps=100, T0=T, T1=T / 300, zero_sweeps=20))
        pa = online_addr(S, alpha, sc)
        report("ADDRESS online greedy", ph=pa)
        report("ADDRESS + replay (T=0)", ph=replay_addr(K, pa, rng, sweeps=100))
        report("ADDRESS + annealed Gibbs", ph=replay_addr(K, pa, rng, sweeps=100, T0=T, T1=T / 300, zero_sweeps=20))
        pc = online_addr(S, alpha, sc, rule="centred")
        report("ADDRESS online centred", ph=pc)
        report("ADDRESS centred + replay (T=0)", ph=replay_addr(K, pc, rng, sweeps=100))
        batch = place.learned(S, sc, np.random.default_rng(seed + 4), alpha, error=True)[1]
        report("batch swaps (labels)", lab=batch)
        orc = sc.phases[place.oracle(sc, Cc.factors)]
        report("oracle", ph=orc)
        report("oracle + ADDRESS replay (T=0)", ph=replay_addr(K, orc, rng, sweeps=100))
        print(f"{seed} ({time.time() - t0:.0f}s)", flush=True)
