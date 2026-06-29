"""Checks for section 1: the Schur increment (1.3), the novelty gate (1.6), snap vs greedy (1.7-1.9),
and the clone prediction P1d. Online encoding as in gold2/lead/online.py (same content, scaffold,
alpha, caps), with K grown by the block inverse (1.1) instead of a fresh solve each step.

    python t1_online.py [seed ...]
"""
from common import *
from scipy.stats import norm

Ns, P, f = 1000, 800, 0.1
GROUPS = place.GROUPS
CAP = [int(np.ceil(P / k)) for k in GROUPS]


def check_increment(S, K, labels, alpha, s, rng):
    """(1.3) against brute force, for every phase of every module."""
    n = S.shape[1]
    c = K @ (S.T @ s)
    r = s @ s + alpha - s @ S @ c
    K2 = np.linalg.inv(np.column_stack([S, s]).T @ np.column_stack([S, s]) + alpha * np.eye(n + 1))
    worst = 0.0
    for m, k in enumerate(GROUPS):
        Q = np.eye(k)[labels[:n, m]]
        J0 = np.einsum("ik,ij,jk->", Q, K, Q)
        C = Q.T @ c
        for kk in range(k):
            Q2 = np.vstack([Q, np.eye(k)[kk]])
            J1 = np.einsum("ik,ij,jk->", Q2, K2, Q2)
            pred = ((C - np.eye(k)[kk]) ** 2).sum() / r
            worst = max(worst, abs((J1 - J0) - pred) / abs(J1 - J0))
    return worst, r


def check_gate(S, K, alpha, s, Q, rng):
    """(1.6): hold the explained part fixed, add novelty orthogonal to range(S); regret x r is fixed."""
    U, _, _ = np.linalg.svd(S, full_matrices=False)
    u = U @ (U.T @ s)
    v = rng.standard_normal(len(s)); v -= U @ (U.T @ v); v /= np.linalg.norm(v)
    out = []
    for scale in (0.0, 5.0, 15.0, 30.0):
        x = u + scale * v
        c = K @ (S.T @ x); r = x @ x + alpha - x @ S @ c
        C = Q.T @ c
        regret = 2 * (C.max() - C.min()) / r
        out.append((scale ** 2, r, regret, regret * r))
    return out


def centred(x, sc):
    out = x.copy()
    for o, k in zip(sc.offsets, sc.sizes):
        out[o:o + k] -= out[o:o + k].mean()
    return out


def run(seed, mode="snap", verify=0):
    C_ = factored(Ns, P, np.random.default_rng(10_000 * seed + P)); S = C_.patterns
    sc = Scaffold(seed=seed); alpha = mmse_alpha(P, flip_rate=f)
    rng = np.random.default_rng(99 + seed)
    D_glob = sc.G @ np.linalg.pinv(sc.H)            # global linear decoder, place -> grid
    # per-phase gain correction for the Hebbian templates (on-phase mean overlap, address constant removed)
    gain = []
    for m, (o, k) in enumerate(zip(sc.offsets, sc.sizes)):
        T = sc.W_gh[o:o + k] @ sc.H; T = T - T.mean(0, keepdims=True)
        gain.append(np.array([T[kk, sc.phases[:, m] == kk].mean() for kk in range(k)]))
    gain = np.concatenate(gain)
    labels = np.zeros((P, 3), dtype=np.int64)
    taken = np.zeros(sc.sizes, dtype=bool)
    where = np.zeros(P, dtype=np.int64)
    ph0 = next(c for c in place._candidates(sc, (0, 0, 0)) if not taken[c]); taken[ph0] = True
    where[0] = sc.index(np.array([ph0]))[0]
    K = np.array([[1.0 / (S[:, 0] @ S[:, 0] + alpha)]])
    variants = ("snap", "noReLU_hebb", "noReLU_hebb_gain", "noReLU_Dglob", "ReLU_Dstored", "noReLU_Dstored")
    agree = {v: [] for v in variants}
    agree_addr = {v: [] for v in variants}
    mismatch = []
    rec = []            # per step: n, r, margin per module, ||c||, relative distortion
    worst_inc = 0.0; gates = None
    for n in range(1, P):
        Sn = S[:, :n]; s = S[:, n]
        b = Sn.T @ s; c = K @ b; r = s @ s + alpha - b @ c
        if verify and n in (50, 200, 500):
            w, _ = check_increment(Sn, K, labels, alpha, s, rng); worst_inc = max(worst_inc, w)
            if n == 500:
                gates = check_gate(Sn, K, alpha, s, np.eye(9)[labels[:n, 0]], rng)
        masks, exact = [], []
        for m, k in enumerate(GROUPS):
            full = np.bincount(labels[:n, m], minlength=k) >= CAP[m]
            Cm = np.bincount(labels[:n, m], weights=c, minlength=k)
            Cm[full] = -np.inf
            masks.append(full); exact.append(int(Cm.argmax()))
        actual = sc.phases[where[:n]].copy(); actual[:, 2] //= place.SLOTS
        addr_exact = []
        for m, k in enumerate(GROUPS):
            Cm = np.bincount(actual[:, m], weights=c, minlength=k); Cm[masks[m]] = -np.inf
            addr_exact.append(int(Cm.argmax()))
        mismatch.append((actual != labels[:n]).mean(0))
        Ha = sc.H[:, where[:n]]
        u = Ha @ c
        Ga = sc.G[:, where[:n]]
        if n <= sc.place_cells:
            proj_c = c                                  # H_a has full column rank: G_a H_a^+ H_a = G_a
        else:
            proj_c = Ha.T @ np.linalg.solve(Ha @ Ha.T, u)
        inputs = {
            "snap": sc.W_gh @ relu(u),
            "noReLU_hebb": sc.W_gh @ u,
            "noReLU_hebb_gain": centred(sc.W_gh @ u, sc) / gain,
            "noReLU_Dglob": D_glob @ u,
            "ReLU_Dstored": Ga @ np.linalg.lstsq(Ha, relu(u), rcond=None)[0],
            "noReLU_Dstored": Ga @ proj_c,
        }
        choice = {}
        for v, gi in inputs.items():
            ch = []
            for m, k in enumerate(GROUPS):
                inp = gi[sc.offsets[m]: sc.offsets[m] + sc.sizes[m]].copy()
                if m == 2:
                    inp = inp.reshape(k, place.SLOTS).sum(1)
                inp[masks[m]] = -np.inf
                ch.append(int(inp.argmax()))
            choice[v] = ch
            agree[v].append(np.array(ch) == np.array(exact))
            agree_addr[v].append(np.array(ch) == np.array(addr_exact))
        # margin model inputs: C margins and the no-ReLU Hebbian distortion
        row = [n, r, float(np.linalg.norm(c))]
        for m, k in enumerate(GROUPS):
            Cm = np.bincount(labels[:n, m], weights=c, minlength=k).astype(float)
            ok = ~masks[m]
            srt = np.sort(Cm[ok])[::-1]
            row.append(srt[0] - srt[1] if ok.sum() > 1 else np.inf)
        rec.append(row)
        pick = choice["snap"] if mode == "snap" else exact
        labels[n] = pick
        ph = next(cc for cc in place._candidates(sc, tuple(int(x) for x in pick)) if not taken[cc])
        taken[ph] = True; where[n] = sc.index(np.array([ph]))[0]
        # grow K by the block inverse (1.1)
        K = np.block([[K + np.outer(c, c) / r, -c[:, None] / r], [-c[None, :] / r, np.array([[1.0 / r]])]])
    # recall + NMIs of the final placement
    cues = flip(S, f, np.random.default_rng(7 + seed))
    mem = Memory(sc, rule="ridge", alpha=alpha); mem.store(S, where)
    recall = float(np.mean(mem.recall(cues).address == where))
    nm = [[nmi(labels[:, m], C_.factors[:, g]) for g in range(3)] for m in range(3)]
    return dict(agree={v: np.mean(a, 0) for v, a in agree.items()},
                agree_addr={v: np.mean(a, 0) for v, a in agree_addr.items()},
                mismatch=np.array(mismatch), agree_steps={v: np.array(a) for v, a in agree_addr.items()}, rec=np.array(rec), labels=labels,
                recall=recall, nmi=nm, worst_inc=worst_inc, gates=gates, factors=C_.factors,
                Kerr=float(np.abs(K - np.linalg.inv(S.T @ S + alpha * np.eye(P))).max() / np.abs(K).max()))


def clone_time(labels):
    """First item index at which module 0 and module 1 labellings stop being identical up to relabelling."""
    for n in range(2, len(labels) + 1, 5):
        if nmi(labels[:n, 0], labels[:n, 1]) < 0.999:
            return n
    return len(labels)


if __name__ == "__main__":
    import sys
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    for seed in seeds:
        for mode in ("snap", "exact"):
            out = run(seed, mode, verify=(mode == "exact"))
            print(f"seed {seed} trajectory={mode}: recall {out['recall']:.3f}")
            for v, a in out["agree"].items():
                b = out["agree_addr"][v]
                print(f"   agreement {v:18} with label-greedy " + " ".join(f"{x:.3f}" for x in a)
                      + "   with address-greedy " + " ".join(f"{x:.3f}" for x in b))
            mm = out["mismatch"]
            print("   label != actual phase (mean over steps), per module:", mm.mean(0).round(3),
                  " at n=100:", mm[99].round(3), " at n=799:", mm[-1].round(3))
            st = out["agree_steps"]["noReLU_Dstored"]; sn = out["agree_steps"]["snap"]
            print("   address-greedy agreement by phase of the run (n<=400 | n>400): Dstored",
                  st[:399].mean(0).round(3), st[399:].mean(0).round(3), " snap", sn[:399].mean(0).round(3), sn[399:].mean(0).round(3))
            print("   NMI(module m, factor A/B/C):", [" ".join(f"{x:.2f}" for x in row) for row in out["nmi"]],
                  " NMI(m0,m1) final %.2f" % nmi(out["labels"][:, 0], out["labels"][:, 1]),
                  " clone until n=%d" % clone_time(out["labels"]))
            if mode == "exact":
                print(f"   (1.3) worst relative error vs brute force: {out['worst_inc']:.2e};  "
                      f"block-inverse K drift after {P} steps: {out['Kerr']:.2e}")
                print("   (1.6) |v|^2, r, regret, regret*r:",
                      "; ".join(f"{a:.0f} {b:.1f} {c:.2e} {d:.4f}" for a, b, c, d in out["gates"]))
            rec = out["rec"]
            np.save(f"t1_rec_seed{seed}_{mode}.npy", rec)
            print(f"   r: first 20 items mean {rec[:20,1].mean():.0f}, last 100 mean {rec[-100:,1].mean():.0f}; "
                  f"median C-margins per module {np.median(rec[:, 3:], 0).round(3)}", flush=True)
