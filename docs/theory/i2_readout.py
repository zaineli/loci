"""Check of Prop I.3: the read-out at an empty address, against P/Nh, pinv and ridge W_sh.

Content/holdout as gold2/compose, oracle placement. Read-outs (overlap with the composite sign(A+B+C)):
  landed   at the address the clean factor cue snaps to (the construction itself)
  slot0    at the empty address (a, b, first slot of group c)
  stored   each stored item's own read-out vs the item (the paper's metric)
Theory: Omega(0.52 L) with L measured = ||H_a^+ h_new||^2, and with L_iso(rho).
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *
import compose
from t3_compose import omega
from i0_inputs import CARDS

NOISE = 0.52


def L_iso(rho):
    return rho / (1 - rho) if rho < 1 else (1 / (rho - 1) if rho > 1 else np.inf)


def run(seed, P, Nh, lams=(0.0, 0.01, 0.1)):
    rng = np.random.default_rng(31_000 * seed + P + 7 * sum(CARDS))
    held = compose.holdout(CARDS, rng)
    C = compose.make_factored(P, rng, CARDS, held)
    S, Fa = C.patterns, C.factors
    grid0 = Scaffold(seed=seed); sc = Scaffold(seed=seed, place_cells=Nh)
    where = place.oracle(grid0, Fa)
    Ha = sc.H[:, where]
    q = np.array([(a, b, c) for a, b in held for c in range(5)])
    target = compose.composite(C.codes, q)
    alpha = mmse_alpha(P, flip_rate=0.1)
    cvec = np.linalg.solve(S.T @ S + alpha * np.eye(P), S.T @ target)
    landed = sc.snap(sc.W_gh @ relu(Ha @ cvec))
    taken = np.zeros(grid0.sizes, bool); taken[tuple(grid0.phases[where].T)] = True
    slot0 = np.array([(a, b, c * 5 + next((s for s in range(5) if not taken[a, b, c * 5 + s]), 0)) for a, b, c in q])
    gram = Ha @ Ha.T; scale = np.trace(gram) / Nh
    out = {}
    for lam in lams:
        W = S @ np.linalg.pinv(Ha) if lam == 0 else S @ Ha.T @ np.linalg.inv(gram + lam * scale * np.eye(Nh))
        pick = np.arange(0, P, max(1, P // 100))
        res = {"stored": float(np.mean(np.sign(W @ Ha[:, pick]) * S[:, pick]))}
        for name, ph in (("landed", landed), ("slot0", slot0)):
            hn = sc.place(ph)
            res[name] = float(np.mean(np.sign(W @ hn) * target))
            if lam == 0:
                w = np.linalg.pinv(Ha) @ hn
                res[f"L_{name}"] = float(np.mean((w ** 2).sum(0)))
        out[lam] = res
    return out


if __name__ == "__main__":
    grid_pts = [(400, 100), (400, 200), (400, 300), (400, 360), (400, 400), (400, 440), (400, 500), (400, 600), (400, 800), (400, 900),
                (800, 200), (800, 400), (800, 600), (800, 800), (800, 900)]
    print("Nh    P  P/Nh | pinv: landed  slot0  stored | L_landed L_slot0 L_iso | theory Omega(0.52 L_meas) landed/slot0, Omega(0.52 L_iso) "
          "| ridge 0.01: landed slot0 stored | ridge 0.1: landed slot0 stored")
    for Nh, P in grid_pts:
        rs = [run(s, P, Nh) for s in (0, 1, 2)]
        m = {lam: {k: np.mean([r[lam][k] for r in rs]) for k in rs[0][lam]} for lam in rs[0]}
        rho = P / Nh
        Li = L_iso(rho)
        th_l = omega(NOISE * m[0.0]["L_landed"], draws=100_000); th_s = omega(NOISE * m[0.0]["L_slot0"], draws=100_000)
        th_i = omega(NOISE * Li, draws=100_000) if np.isfinite(Li) else 0.0
        print(f"{Nh:4} {P:4} {rho:5.2f} | {m[0.0]['landed']:.3f} {m[0.0]['slot0']:.3f} {m[0.0]['stored']:.3f} | "
              f"{m[0.0]['L_landed']:8.2f} {m[0.0]['L_slot0']:7.2f} {Li:6.2f} | {th_l:.3f}/{th_s:.3f}, {th_i:.3f} | "
              f"{m[0.01]['landed']:.3f} {m[0.01]['slot0']:.3f} {m[0.01]['stored']:.3f} | {m[0.1]['landed']:.3f} {m[0.1]['slot0']:.3f} {m[0.1]['stored']:.3f}",
              flush=True)
