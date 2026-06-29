"""Section 2 checks: bias = gradient (2.1-2.3), the fill-up/clone diagnosis, and the escapes (P2b).

Content, scaffold, alpha, cues and caps as gold2/lead/replay.py. Every descent is on the same J
(sum over phases of q^T K q, per module), with single-item capacity ceil(1.1 P / k) as the lead's.

    python t2_replay.py [seed ...]
"""
from common import *
import time

Ns, P, f = 1000, 800, 0.1
GROUPS = place.GROUPS


# ---------------------------------------------------------------- exact identities ------------------
def check_identities(S, K, alpha, labels, rng):
    i = int(rng.integers(S.shape[1]))
    c = K @ (S.T @ S[:, i])
    e = np.zeros(S.shape[1]); e[i] = 1
    err21 = np.abs(c - (e - alpha * K[:, i])).max()
    lab = labels[:, 0].copy(); k = GROUPS[0]
    J0 = J_of(K, lab[:, None]); a = lab[i]; worst = 0.0
    bias = -alpha * K[:, i].copy(); bias[i] = 0
    B = np.bincount(lab, weights=bias, minlength=k)
    # leave-one-out coefficients
    others = np.delete(np.arange(S.shape[1]), i)
    So = S[:, others]; cl = np.linalg.solve(So.T @ So + alpha * np.eye(len(others)), So.T @ S[:, i])
    loo = np.zeros(S.shape[1]); loo[others] = cl
    Bloo = alpha * K[i, i] * np.bincount(lab, weights=loo, minlength=k)
    for b in range(k):
        if b == a: continue
        lab2 = lab.copy(); lab2[i] = b
        dJ = J_of(K, lab2[:, None]) - J0
        worst = max(worst, abs(dJ - 2 / alpha * (B[a] - B[b])) / abs(dJ))
    return err21, worst, np.abs(B - Bloo).max() / np.abs(B).max()


# ---------------------------------------------------------------- online encoders ---------------------
def online(S, alpha, rule="greedy", rng=None, T_enc=1.0):
    """Items in order; K grown by the block inverse. rule: greedy (lead), centred, seeded, gibbs."""
    labels = np.zeros((P, 3), dtype=np.int64)
    cap = [int(np.ceil(P / k)) for k in GROUPS]
    K = np.array([[1.0 / (S[:, 0] @ S[:, 0] + alpha)]])
    fill_choice = []
    r_seed = alpha + 0.85 * Ns
    for n in range(1, P):
        Sn = S[:, :n]; s = S[:, n]; b = Sn.T @ s; c = K @ b; r = s @ s + alpha - b @ c
        for m, k in enumerate(GROUPS):
            count = np.bincount(labels[:n, m], minlength=k); full = count >= cap[m]
            C = np.bincount(labels[:n, m], weights=c, minlength=k)
            if rule == "greedy":
                score = C.copy()
            elif rule == "centred":                   # price group size by the mean coefficient
                score = C - count * c.sum() / n
            elif rule == "seeded":                    # novelty gate: a poorly explained item seeds a group
                score = C.copy()
                if r > r_seed or (count == 0).any() and n < k:
                    score = -count.astype(float)
            elif rule == "gibbs":                     # independent noise per module, z-scored familiarity
                z = (C - C[~full].max()) / (C[~full].std() + 1e-12)
                score = z + rng.gumbel(size=k) * T_enc
            score = score.astype(float); score[full] = -np.inf
            labels[n, m] = int(score.argmax())
            if m == 0:
                big = np.where(full, -1, count).argmax()
                fill_choice.append(labels[n, 0] == big)
        K = np.block([[K + np.outer(c, c) / r, -c[:, None] / r], [-c[None, :] / r, np.array([[1.0 / r]])]])
    return labels, float(np.mean(fill_choice))


# ---------------------------------------------------------------- replays on J ----------------------------
class Coupling:
    def __init__(self, K, lab, k):
        self.K, self.lab, self.k = K, lab, k
        self.Cp = K @ np.eye(k)[lab]                  # Cp[i, g] = sum_{j in g} K_ij
        self.count = np.bincount(lab, minlength=k)

    def delta(self, i):
        """Delta J for moving i to each group (2.2); 0 for staying."""
        a = self.lab[i]
        d = 2 * (self.Cp[i] - (self.Cp[i, a] - self.K[i, i]))
        d[a] = 0.0
        return d

    def move(self, i, b):
        a = self.lab[i]
        if a == b: return
        self.Cp[:, a] -= self.K[:, i]; self.Cp[:, b] += self.K[:, i]
        self.count[a] -= 1; self.count[b] += 1; self.lab[i] = b


def replay(K, labels, rng, sweeps=8, order="random", T0=0.0, T1=0.0, zero_sweeps=0):
    """Single-item replay per module. order: random | worst-first. T0 > 0: Gibbs, T geometric T0 -> T1."""
    labels = labels.copy()
    for m, k in enumerate(GROUPS):
        cap = int(np.ceil(1.1 * P / k))
        cp = Coupling(K, labels[:, m], k)
        temps = (list(np.geomspace(T0, T1, sweeps)) if T0 > 0 else [0.0] * sweeps) + [0.0] * zero_sweeps
        for T in temps:
            if order == "worst-first":
                gains = np.array([-(cp.delta(i)).min() for i in range(P)])
                seq = np.argsort(-gains)
            else:
                seq = rng.permutation(P)
            moved = 0
            for i in seq:
                d = cp.delta(i)
                blocked = (cp.count >= cap) & (np.arange(k) != cp.lab[i])
                d[blocked] = np.inf
                if T > 0:
                    w = np.exp(-(d - d.min()) / T); w[blocked] = 0
                    b = int(rng.choice(k, p=w / w.sum()))
                else:
                    b = int(d.argmin())
                    if not d[b] < -1e-15: b = cp.lab[i]
                if b != cp.lab[i]:
                    cp.move(i, b); moved += 1
            if T == 0 and moved == 0 and order != "worst-first":
                break
        labels[:, m] = cp.lab
    return labels


def best_first(K, labels, max_moves=20 * P):
    """Strict most-mis-placed-first: always make the single largest improving move anywhere."""
    labels = labels.copy()
    for m, k in enumerate(GROUPS):
        cap = int(np.ceil(1.1 * P / k))
        cp = Coupling(K, labels[:, m], k)
        diag = np.diag(K)
        for _ in range(max_moves):
            D = 2 * (cp.Cp - (cp.Cp[np.arange(P), cp.lab] - diag)[:, None])
            D[np.arange(P), cp.lab] = 0.0
            D[:, cp.count >= cap] = np.where(np.eye(k)[cp.lab][:, cp.count >= cap] > 0, 0.0, np.inf)
            idx = np.unravel_index(D.argmin(), D.shape)
            if not D[idx] < -1e-15: break
            cp.move(idx[0], idx[1])
        labels[:, m] = cp.lab
    return labels


def swaps(K, labels, rng):
    off = K - np.diag(np.diag(K))
    return np.stack([place._swaps(off, labels[:, m], rng) for m in range(3)], axis=1)


def typical_dJ(K, labels):
    cp = Coupling(K, labels[:, 0].copy(), GROUPS[0])
    d = np.array([cp.delta(i) for i in range(P)])
    return float(np.std(d[d != 0]))


if __name__ == "__main__":
    import sys
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    for seed in seeds:
        t0 = time.time()
        Cc = factored(Ns, P, np.random.default_rng(10_000 * seed + P)); S = Cc.patterns
        sc = Scaffold(seed=seed); alpha = mmse_alpha(P, flip_rate=f)
        cues = flip(S, f, np.random.default_rng(7 + seed))
        K = place.precision(S, alpha)
        rng = np.random.default_rng(seed)

        def report(name, lab):
            where = place.addresses(sc, lab)
            mem = Memory(sc, rule="ridge", alpha=alpha); mem.store(S, where)
            rec = float(np.mean(mem.recall(cues).address == where))
            nm = [nmi(lab[:, m], Cc.factors[:, g]) for m in range(3) for g in range(3)]
            print(f"{seed} {name:34} recall {rec:.3f}  J {J_of(K, lab):8.4f}  NMI m0:A {nm[0]:.2f} m0:C {nm[2]:.2f} "
                  f"m1:B {nm[4]:.2f} m2:C {nm[8]:.2f}  (m0 A/B/C {nm[0]:.2f}/{nm[1]:.2f}/{nm[2]:.2f})", flush=True)
            return rec

        lab0, fill = online(S, alpha, "greedy")
        e21, e22, e23 = check_identities(S, K, alpha, lab0, rng)
        print(f"{seed} identities: (2.1) max err {e21:.1e}; (2.2) rel err {e22:.1e}; (2.3) LOO rel err {e23:.1e}")
        print(f"{seed} online greedy: module-0 choice = largest non-full group in {fill:.1%} of arrivals")
        T = typical_dJ(K, lab0)
        report("online greedy", lab0)
        report("  + replay 8 (random order)", replay(K, lab0, rng, sweeps=8))
        report("  + replay to convergence", replay(K, lab0, rng, sweeps=200))
        report("  + worst-first to convergence", best_first(K, lab0))
        report("  + pair swaps (co-replay)", swaps(K, lab0, rng))
        report("  + replay, then swaps", swaps(K, replay(K, lab0, rng, sweeps=200), rng))
        report("  + annealed Gibbs 100 sweeps", replay(K, lab0, rng, sweeps=100, T0=T, T1=T / 300, zero_sweeps=20))
        lab_s, _ = online(S, alpha, "seeded")
        report("online novelty-seeded", lab_s)
        report("  + replay to convergence", replay(K, lab_s, rng, sweeps=200))
        lab_c, fill_c = online(S, alpha, "centred")
        print(f"{seed} centred online: largest-group choice {fill_c:.1%}")
        report("online centred (size-priced)", lab_c)
        report("  + replay to convergence", replay(K, lab_c, rng, sweeps=200))
        report("  + annealed Gibbs 100 sweeps", replay(K, lab_c, rng, sweeps=100, T0=T, T1=T / 300, zero_sweeps=20))
        lab_g, _ = online(S, alpha, "gibbs", rng=rng)
        report("online Gibbs (per-module noise)", lab_g)
        report("  + annealed Gibbs 100 sweeps", replay(K, lab_g, rng, sweeps=100, T0=T, T1=T / 300, zero_sweeps=20))
        batch = place.learned(S, sc, np.random.default_rng(seed + 4), alpha, error=True)[1]
        report("batch swaps (20 starts)", batch)
        report("oracle", Cc.factors)
        print(f"{seed} ({time.time() - t0:.0f}s)", flush=True)
