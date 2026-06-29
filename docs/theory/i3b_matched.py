"""Model-free test of Prop I.5: a factor cue at the lure's SNR constructs exactly as often as the lure
is falsely recalled, for every decoder in the family. Lure (10% flips): gain gamma' = 0.693, noise 0.741,
SNR 0.648; a factor cue with f' flips has SNR (1-2f')^2 / 4f'(1-f') = 0.648 at f' = 0.1866."""
from i3_frontier import *
FM = 0.1866
LAMS2 = (0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2)
rows = {}
for seed in (0, 1, 2):
    rng = np.random.default_rng(31_000 * seed + 800 + 7 * sum(CARDS))
    held = compose.holdout(CARDS, rng)
    C = compose.make_factored(800, rng, CARDS, held)
    S, Fa = C.patterns, C.factors; P = S.shape[1]
    grid = Scaffold(seed=seed); where = place.oracle(grid, Fa); Ha = grid.H[:, where]
    Hn_all = grid.H / np.linalg.norm(grid.H, axis=0, keepdims=True)
    stored = np.zeros(grid.addresses); stored[where] = 1.0
    alpha = mmse_alpha(P, flip_rate=F); Wc = np.linalg.solve(S.T @ S + alpha * np.eye(P), S.T)
    q = np.array([(a, b, c) for a, b in held for c in range(5)])
    r2 = np.random.default_rng(4000 + seed)
    for rep in range(5):
        lq = q
        lures = flip(np.sign(sum(C.codes[f_][:, lq[:, f_]] for f_ in range(3)) + r2.standard_normal((1000, len(lq)))), F, r2)
        facm = flip(compose.composite(C.codes, q), FM, r2)
        for name, X in (("lure 10% (false a-b)", lures), ("factor cue 18.7% (construct a-b)", facm)):
            h0 = relu(Ha @ (Wc @ X))
            outs = {"snap": grid.index(grid.snap(grid.W_gh @ h0))}
            outs.update({f"lam{l}": v for l, v in decode_all(h0, Hn_all, stored, LAMS2).items()})
            for d, idx in outs.items():
                ph = grid.phases[idx]
                rows.setdefault((name, d), []).append(np.mean((ph[:, 0] == q[:, 0]) & (ph[:, 1] == q[:, 1])))
                rows.setdefault((name, d, "full"), []).append(np.mean((ph[:, 0] == q[:, 0]) & (ph[:, 1] == q[:, 1]) & (ph[:, 2] // 5 == q[:, 2])))
print(f"{'decoder':8} | lure false (a,b) / factor-cue construct (a,b) | same, with C-group")
for d in ["snap"] + [f"lam{l}" for l in LAMS2]:
    a1 = np.mean(rows[("lure 10% (false a-b)", d)]); b1 = np.mean(rows[("factor cue 18.7% (construct a-b)", d)])
    a2 = np.mean(rows[("lure 10% (false a-b)", d, "full")]); b2 = np.mean(rows[("factor cue 18.7% (construct a-b)", d, "full")])
    print(f"{d:8} | {a1:.3f} / {b1:.3f}  (diff {a1-b1:+.3f})           | {a2:.3f} / {b2:.3f}  (diff {a2-b2:+.3f})")
