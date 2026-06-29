"""Check of Props I.4-I.6: the decoder frontier, the lure lemma, and construction == false recall.

Setting as gold2/compose (16 held-out (a, b) pairs), P = 800, Nh = 400, alpha = MMSE(10%).
Probes: studied items (10% flips), factor cues of held-out (a, b, c) (clean and 10% flips),
recombination lures sign(A_a + B_b + C_c + z') on held-out (a, b) (10% flips), unrelated lures.
Decoders: the paper's snap; D_lam = argmax over all 3,600 tuples of cos(h0, h_t) + lam 1[stored]
(lam = 0: nearest code over all tuples; lam >= 2: joint cosine decode over stored addresses).
'measured' = real flips / real lures; 'T-mc' = Gaussian cue model of Part C4 (no flips simulated);
'lemma' = I.4's prediction of a lure: the factor cue at gain gamma' with isotropic noise sigma_e^2.
"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *
import compose
from i0_inputs import CARDS

LAMS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 2.0)
F = 0.1; A_ = 1 - 2 * F; S2 = 4 * F * (1 - F)
GAMMA = A_ * np.sqrt(3) / 2                     # 0.693
SIG_E = 1 - 2 * GAMMA * A_ * (2 / 3) + GAMMA ** 2   # 0.741


def decode_all(h0, Hn_all, stored_mask, lams):
    """(n,) best tuple index per lam, from cosine scores over all tuples."""
    sc = Hn_all.T @ h0                           # (3600, n), common 1/||h0|| dropped
    sc = sc / (np.linalg.norm(h0, axis=0, keepdims=True) + 1e-12)
    return {lam: (sc + lam * stored_mask[:, None]).argmax(0) for lam in lams}


def metrics(outs, grid, kind, truth):
    """outs: {decoder: tuple index (n,)} -> {decoder: rate}; truth depends on kind."""
    res = {}
    for name, idx in outs.items():
        ph = grid.phases[idx]
        if kind == "studied":
            res[name] = float(np.mean(idx == truth))
        elif kind == "factor":
            res[name] = float(np.mean((ph[:, 0] == truth[:, 0]) & (ph[:, 1] == truth[:, 1]) & (ph[:, 2] // 5 == truth[:, 2])))
        elif kind == "lure":
            res[name] = float(np.mean((ph[:, 0] == truth[:, 0]) & (ph[:, 1] == truth[:, 1])))          # conjunction a-b asserted
            res[name + "|full"] = float(np.mean((ph[:, 0] == truth[:, 0]) & (ph[:, 1] == truth[:, 1]) & (ph[:, 2] // 5 == truth[:, 2])))
    return res


def run(seed, placement, draws=40):
    rng = np.random.default_rng(31_000 * seed + 800 + 7 * sum(CARDS))
    held = compose.holdout(CARDS, rng)
    C = compose.make_factored(800, rng, CARDS, held)
    S, Fa = C.patterns, C.factors; P = S.shape[1]
    grid = Scaffold(seed=seed)
    where = place.oracle(grid, Fa) if placement == "aligned" else place.scattered(P, grid, np.random.default_rng(seed + 1))
    Ha = grid.H[:, where]
    Hn_all = grid.H / np.linalg.norm(grid.H, axis=0, keepdims=True)
    stored = np.zeros(grid.addresses); stored[where] = 1.0
    alpha = mmse_alpha(P, flip_rate=F)
    K = np.linalg.inv(S.T @ S + alpha * np.eye(P)); Wc = K @ S.T
    r2 = np.random.default_rng(900 + seed)
    q = np.array([(a, b, c) for a, b in held for c in range(5)])
    fac = compose.composite(C.codes, q)
    lure_q = np.repeat(q, 2, axis=0)
    lures = np.sign(sum(C.codes[f][:, lure_q[:, f]] for f in range(3)) + r2.standard_normal((1000, len(lure_q))))
    studied_idx = r2.choice(P, 300, replace=False)
    probes = {"studied": (S[:, studied_idx], where[studied_idx]), "factor": (fac, q), "factor10": (fac, q),
              "lure": (lures, lure_q), "unrelated": (np.sign(r2.standard_normal((1000, 160))), None)}
    out = {"measured": {}, "T-mc": {}}
    covu = S2 * Ha @ (K - alpha * K @ K) @ Ha.T
    w, V = np.linalg.eigh(covu); Lu = V * np.sqrt(np.clip(w, 0, None))
    Z = np.random.default_rng(seed + 5).standard_normal((draws, Ha.shape[0])) @ Lu.T
    for name, (X, truth) in probes.items():
        noisy = X if name == "factor" else flip(X, F, np.random.default_rng(77 + seed + hash(name) % 100))
        h0 = relu(Ha @ (Wc @ noisy))
        outs = {"snap": grid.index(grid.snap(grid.W_gh @ h0))}
        outs.update({f"lam{l}": v for l, v in decode_all(h0, Hn_all, stored, LAMS).items()})
        kind = {"factor10": "factor"}.get(name, name)
        if truth is not None:
            out["measured"][name] = metrics(outs, grid, kind, truth)
        # T-mc: Gaussian model for flipped probes (clean factor cue is deterministic)
        if name != "factor" and truth is not None:
            mu = A_ * (Ha @ (Wc @ X))
            acc = {}
            for d in range(draws):
                h0d = relu(mu + Z[d][:, None])
                od = {"snap": grid.index(grid.snap(grid.W_gh @ h0d))}
                od.update({f"lam{l}": v for l, v in decode_all(h0d, Hn_all, stored, LAMS).items()})
                for k_, v in metrics(od, grid, kind, truth).items():
                    acc[k_] = acc.get(k_, 0) + v / draws
            out["T-mc"][name] = acc
    # lemma: (i) regression of u(lure) on u(factor cue of the same (a, b, c)); (ii) predicted false recall
    u_f = Ha @ (Wc @ np.repeat(fac, 2, axis=1))
    u_l = Ha @ (Wc @ flip(lures, F, np.random.default_rng(77 + seed + hash("lure") % 100)))
    gam = float((u_l * u_f).sum() / (u_f * u_f).sum())
    resid = u_l - gam * u_f
    pred_cov_trace = SIG_E * np.trace(Ha @ (K - alpha * K @ K) @ Ha.T)
    out["lemma"] = {"gamma_measured": gam, "gamma_pred": GAMMA,
                    "resid_var_ratio": float((resid ** 2).sum(0).mean() / pred_cov_trace)}
    Ze = np.sqrt(SIG_E / S2) * Z                          # noise of variance sigma_e^2 instead of sigma^2
    mu = GAMMA * (Ha @ (Wc @ np.repeat(fac, 2, axis=1)))
    acc = {}
    for d in range(draws):
        h0d = relu(mu + Ze[d][:, None])
        od = {"snap": grid.index(grid.snap(grid.W_gh @ h0d))}
        od.update({f"lam{l}": v for l, v in decode_all(h0d, Hn_all, stored, LAMS).items()})
        for k_, v in metrics(od, grid, "lure", lure_q).items():
            acc[k_] = acc.get(k_, 0) + v / draws
    out["lemma"]["false recall"] = acc
    return out


if __name__ == "__main__":
    allres = {}
    for placement in ("aligned", "random"):
        rs = [run(s, placement) for s in (0, 1, 2)]
        allres[placement] = rs
        lem = {k: np.mean([r["lemma"][k] for r in rs]) for k in ("gamma_measured", "gamma_pred", "resid_var_ratio")}
        print(f"\n===== {placement}: lemma gamma' measured {lem['gamma_measured']:.3f} (pred {lem['gamma_pred']:.3f}); "
              f"residual variance / predicted sigma_e^2 trace {lem['resid_var_ratio']:.2f}")
        decs = ["snap"] + [f"lam{l}" for l in LAMS]
        print(f"{'decoder':9} | {'recall10 meas/T':15} | {'construct clean':15} | {'construct10 meas/T':18} | "
              f"{'false a-b meas/T/lemma':24} | {'false full meas/lemma':20}")
        for d in decs:
            g = lambda src, name, key: np.mean([r[src][name][key] for r in rs])
            gl = lambda key: np.mean([r["lemma"]["false recall"][key] for r in rs])
            print(f"{d:9} | {g('measured','studied',d):.3f}/{g('T-mc','studied',d):.3f}     | {g('measured','factor',d):.3f}           | "
                  f"{g('measured','factor10',d):.3f}/{g('T-mc','factor10',d):.3f}        | "
                  f"{g('measured','lure',d):.3f}/{g('T-mc','lure',d):.3f}/{gl(d):.3f}      | "
                  f"{g('measured','lure',d + '|full'):.3f}/{gl(d + '|full'):.3f}", flush=True)
    json.dump(allres, open("i3_results.json", "w"))
