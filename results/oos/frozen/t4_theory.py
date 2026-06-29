"""Section 4: predict per-module accuracy and address recovery from K, the placement and the scaffold,
with no recall simulation, and compare with measured ridge recall (bench conditions, same seeds).

Levels (eq 4.1-4.3):
  E        error law only: x = [Q_0 Q_1 Q_2]^T c~, Gaussian with (4.1) moments, per-module argmax
  T-lin    through the Hebbian templates, no ReLU: x = W_gh H_a c~
  T-relu   through templates and ReLU, moment-matched (exact per-cell mean/var, first-order covariance)
  T-mc     Gaussian c~ (CLT only), exact ReLU, templates: checks the Gaussian-cue assumption
    python t4_theory.py [--loads 0.2,0.8] [--rates 0.1] [--seeds 0,1,2]
"""
from common import *
from scipy.stats import norm
import argparse, json, time

MEASURED = json.load(open("/Users/zain/work/systems/loci/results/placement-factored.json"))


def measured_row(seed, load, rate, placement):
    for r in MEASURED:
        if (r["seed"], r["load"], r["flip"], r["scaffold"], r["placement"]) == (seed, load, rate, "grid", placement):
            return r


def psd_sqrt(cov):
    w, V = np.linalg.eigh(cov)
    return V * np.sqrt(np.clip(w, 0, None))


def mc_argmax(mean, chol, true_ph, sc, draws, rng):
    """mean (n, 50), noise N(0, chol chol^T) common law; returns per-module and joint accuracy."""
    n = mean.shape[0]
    acc = np.zeros(len(sc.sizes)); joint = 0.0
    Z = rng.standard_normal((draws, chol.shape[0])) @ chol.T          # (draws, 50)
    for start in range(0, n, 100):
        x = mean[start:start + 100, None, :] + Z[None]                  # (b, draws, 50)
        ok = np.ones(x.shape[:2], bool)
        for m, (o, k) in enumerate(zip(sc.offsets, sc.sizes)):
            hit = x[..., o:o + k].argmax(-1) == true_ph[start:start + 100, m][:, None]
            acc[m] += hit.mean(1).sum(); ok &= hit
        joint += ok.mean(1).sum()
    return acc / n, joint / n


def theory(sc, S, where, alpha, rate, rng, draws=400, levels=("E", "T-lin", "T-relu", "T-mc")):
    P = S.shape[1]; a = 1 - 2 * rate; s2 = 4 * rate * (1 - rate)
    K = np.linalg.inv(S.T @ S + alpha * np.eye(P))
    mean_c = a * (np.eye(P) - alpha * K)                                # column i: E c~ for item i
    cov_c = s2 * (K - alpha * K @ K)
    true_ph = sc.phases[where]
    G = sc.G[:, where]                                                  # (50, P): [Q_0 Q_1 Q_2]^T
    out = {}
    if "E" in levels:
        mu = (G @ mean_c).T
        cov = G @ cov_c @ G.T
        chol = psd_sqrt(cov)
        out["E"] = mc_argmax(mu, chol, true_ph, sc, draws, rng)
    Ha = sc.H[:, where]
    mu_u = (Ha @ mean_c).T                                              # (P, Nh)
    cov_u = Ha @ cov_c @ Ha.T
    if "T-lin" in levels:
        cov = sc.W_gh @ cov_u @ sc.W_gh.T
        chol = psd_sqrt(cov)
        out["T-lin"] = mc_argmax(mu_u @ sc.W_gh.T, chol, true_ph, sc, draws, rng)
    if "T-relu" in levels:
        sd = np.sqrt(np.diag(cov_u))
        z = mu_u / sd
        Eh = mu_u * norm.cdf(z) + sd * norm.pdf(z)
        Eh2 = (mu_u ** 2 + sd ** 2) * norm.cdf(z) + mu_u * sd * norm.pdf(z)
        Vh = Eh2 - Eh ** 2
        D = norm.cdf(z)                                                 # (P, Nh)
        acc = np.zeros(len(sc.sizes)); joint = 0.0
        Zs = rng.standard_normal((draws, sc.grid_cells))
        for i in range(P):
            M = sc.W_gh * D[i]                                          # (50, Nh)
            cov = M @ cov_u @ M.T + (sc.W_gh * (Vh[i] - D[i] ** 2 * sd ** 2)) @ sc.W_gh.T
            chol = psd_sqrt(cov)
            x = (sc.W_gh @ Eh[i])[None] + Zs @ chol.T
            ok = np.ones(draws, bool)
            for m, (o, k) in enumerate(zip(sc.offsets, sc.sizes)):
                hit = x[:, o:o + k].argmax(1) == true_ph[i, m]
                acc[m] += hit.mean(); ok &= hit
            joint += ok.mean()
        out["T-relu"] = (acc / P, joint / P)
    if "T-mc" in levels:
        chol_u = psd_sqrt(cov_u)
        Xi = rng.standard_normal((draws // 2, chol_u.shape[0])) @ chol_u.T
        acc = np.zeros(len(sc.sizes)); joint = 0.0
        for start in range(0, P, 50):
            h = relu(mu_u[start:start + 50, None, :] + Xi[None])        # (b, draws, Nh)
            x = h @ sc.W_gh.T
            ok = np.ones(x.shape[:2], bool)
            for m, (o, k) in enumerate(zip(sc.offsets, sc.sizes)):
                hit = x[..., o:o + k].argmax(-1) == true_ph[start:start + 50, m][:, None]
                acc[m] += hit.mean(1).sum(); ok &= hit
            joint += ok.mean(1).sum()
        out["T-mc"] = (acc / P, joint / P)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", default="0.2,0.4,0.6,0.8,0.9")
    ap.add_argument("--rates", default="0.1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--placements", default="sequential,random,kmeans,oracle")
    ap.add_argument("--out", default="t4_results.json")
    args = ap.parse_args()
    rows = []
    for rate in map(float, args.rates.split(",")):
        for load in map(float, args.loads.split(",")):
            for seed in map(int, args.seeds.split(",")):
                t0 = time.time()
                content, grid, alpha, cues = bench_condition(seed, load, rate)
                arms = placements(content, grid, seed, which=args.placements.split(","), rate=rate)
                for name, where in arms.items():
                    rec, mods = ridge_recall(grid, content.patterns, where, cues, alpha)
                    ref = measured_row(seed, load, rate, name)
                    th = theory(grid, content.patterns, where, alpha, rate, np.random.default_rng(seed))
                    row = dict(seed=seed, load=load, rate=rate, placement=name, measured=rec,
                               measured_modules=mods.tolist(), json_ridge=ref["ridge"], json_modules=ref["ridge_modules"],
                               law_error=ref["law_error"],
                               **{k: dict(modules=v[0].tolist(), joint=float(v[1]), product=float(np.prod(v[0])))
                                  for k, v in th.items()})
                    rows.append(row)
                    print(f"rate {rate} load {load} seed {seed} {name:10} measured {rec:.3f} {np.round(mods, 3)} (json {ref['ridge']:.3f})  "
                          + "  ".join(f"{k} {v[1]:.3f} {np.round(v[0], 3)}" for k, v in th.items()), flush=True)
                print(f"   ({time.time() - t0:.0f}s)", flush=True)
                json.dump(rows, open(args.out, "w"))
