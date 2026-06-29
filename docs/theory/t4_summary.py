"""Summarise t4_results.json: predicted vs measured address recovery and module accuracy."""
import json, numpy as np
from scipy.stats import spearmanr, kendalltau
rows = json.load(open("t4_results.json"))
LEVELS = ("E", "T-lin", "T-relu", "T-mc")
PL = ("sequential", "random", "kmeans", "learned", "error", "oracle")


def cell(rate, load, placement, key):
    rs = [r for r in rows if r["rate"] == rate and r["load"] == load and r["placement"] == placement]
    if key == "measured":
        return np.mean([r["measured"] for r in rs]), np.mean([r["measured_modules"] for r in rs], 0)
    return np.mean([r[key]["joint"] for r in rs]), np.mean([r[key]["modules"] for r in rs], 0)


for rate in (0.1, 0.2):
    print(f"\n=== flip {rate}: address recovery, mean over seeds 0-2 (measured | E | T-lin | T-relu | T-mc)")
    for load in (0.2, 0.4, 0.6, 0.8, 0.9):
        line = f"load {load}: "
        for p in PL:
            if not any(r["placement"] == p and r["load"] == load and r["rate"] == rate for r in rows):
                continue
            m = cell(rate, load, p, "measured")[0]
            line += f"{p[:4]} {m:.3f}|" + "|".join(f"{cell(rate, load, p, k)[0]:.3f}" for k in LEVELS) + "  "
        print(line)
print("\n=== headline (load 0.8, flip 0.1): module accuracies, measured vs T-relu")
for p in PL:
    m, mm = cell(0.1, 0.8, p, "measured"); t, tm = cell(0.1, 0.8, p, "T-relu"); e, em = cell(0.1, 0.8, p, "E")
    print(f"{p:10} measured {m:.3f} {np.round(mm, 3)}   T-relu {t:.3f} {np.round(tm, 3)}   E {e:.3f} {np.round(em, 3)}")
print("\n=== error of each level over all (seed, load, flip, placement) rows")
for k in LEVELS:
    d = np.array([r[k]["joint"] - r["measured"] for r in rows])
    dm = np.array([np.array(r[k]["modules"]) - np.array(r["measured_modules"]) for r in rows])
    print(f"{k:7} address: mean {d.mean():+.3f}  MAE {np.abs(d).mean():.3f}  max |err| {np.abs(d).max():.3f}  "
          f"frac within 0.05 {np.mean(np.abs(d) <= 0.05):.2f};  modules MAE {np.abs(dm).mean():.3f}")
    worst = rows[int(np.abs(d).argmax())]
    print(f"        worst: {worst['placement']} load {worst['load']} flip {worst['rate']} seed {worst['seed']}: "
          f"measured {worst['measured']:.3f} predicted {worst[k]['joint']:.3f}")
print("\n=== ordering of placements within each (seed, load, flip): Kendall tau, measured vs predicted")
for k in LEVELS:
    taus = []
    for rate in (0.1, 0.2):
        for load in (0.2, 0.4, 0.6, 0.8, 0.9):
            for seed in (0, 1, 2):
                rs = [r for r in rows if r["rate"] == rate and r["load"] == load and r["seed"] == seed]
                if len(rs) > 2:
                    taus.append(kendalltau([r["measured"] for r in rs], [r[k]["joint"] for r in rs]).statistic)
    print(f"{k:7} mean tau {np.nanmean(taus):.3f}  min {np.nanmin(taus):.3f}")
print("\n=== joint vs product of module accuracies (T-relu), and measured joint vs product of measured modules")
d1 = np.array([r["T-relu"]["joint"] - r["T-relu"]["product"] for r in rows])
d2 = np.array([r["measured"] - np.prod(r["measured_modules"]) for r in rows])
print(f"T-relu joint - product: mean {d1.mean():+.4f} max {np.abs(d1).max():.3f};  measured: mean {d2.mean():+.4f} max {np.abs(d2).max():.3f}")
