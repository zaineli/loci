"""Why lower label-J can mean worse recall: per-module J on labels vs on realized addresses, overflow
from open addressing, redundancy between modules, and per-module recall accuracy."""
from t2_replay import *

def diag(name, lab, sc, S, K, cues, alpha, factors):
    where = place.addresses(sc, lab)
    real = sc.phases[where]
    mem = Memory(sc, rule="ridge", alpha=alpha); mem.store(S, where)
    got = mem.recall(cues)
    acc = (got.phases == real).mean(0)
    Jl = [J_of(K, lab[:, [m]]) for m in range(3)]
    Ja = [J_of(K, real[:, [m]]) for m in range(3)]
    over = [(real[:, 0] != lab[:, 0]).mean(), (real[:, 1] != lab[:, 1]).mean(), (real[:, 2] // 5 != lab[:, 2]).mean()]
    print(f"{name:28} recall {np.mean(got.address == where):.3f} mod-acc {np.round(acc,3)}  J_label {np.round(Jl,3)} "
          f"J_addr {np.round(Ja,3)} (sum {sum(Ja):.3f})  overflow {np.round(over,3)}  NMI(m0,m1) {nmi(lab[:,0], lab[:,1]):.2f}", flush=True)
    return sum(Ja), float(np.mean(got.address == where))

if __name__ == "__main__":
    import sys
    from scipy.stats import spearmanr
    seeds = [int(x) for x in sys.argv[1:]] or [0]
    for seed in seeds:
        Cc = factored(Ns, P, np.random.default_rng(10_000 * seed + P)); S = Cc.patterns
        sc = Scaffold(seed=seed); alpha = mmse_alpha(P, flip_rate=f)
        cues = flip(S, f, np.random.default_rng(7 + seed)); K = place.precision(S, alpha)
        rng = np.random.default_rng(seed)
        lab0, _ = online(S, alpha, "greedy"); T = typical_dJ(K, lab0)
        arms = {"online": lab0, "replay8": replay(K, lab0, rng, sweeps=8),
                "replay-conv": replay(K, lab0, rng, sweeps=200), "worst-first": best_first(K, lab0),
                "swaps": swaps(K, lab0, rng),
                "anneal": replay(K, lab0, rng, sweeps=100, T0=T, T1=T / 300, zero_sweeps=20),
                "batch": place.learned(S, sc, np.random.default_rng(seed + 4), alpha, error=True)[1],
                "oracle": Cc.factors}
        pts = [diag(k, v, sc, S, K, cues, alpha, Cc.factors) for k, v in arms.items()]
        print("Spearman(J_addr, recall) over arms:", round(spearmanr(*zip(*pts)).statistic, 3))
