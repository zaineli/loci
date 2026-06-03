"""E1 in the paper's own terms, and three questions the cliff leaves open.

`cliff.py` measures address recovery against P / Ns. This holds it against the paper's own
supplementary figures, in the paper's own metric, then asks three things of it:
  s7     SI Fig S7-right: mutual information per input bit against P, clean and 10%-flipped cues,
         at the paper's Ns = Npos = 3,600 - the configuration that puts P = Ns at the right edge of
         the axis. MI is reported two ways because they differ by ~2x under noise: `mi_pooled` is
         the notebook's (1 - H2 of the overlap averaged over all patterns; what the paper plots) and
         `mi_per_pattern` takes 1 - H2 of each pattern's overlap, then averages.
  s8     SI Fig S8: p(right address) against flip rate at P in {250, ..., 3,000}, for the
         pseudo-inverse and for the noise-matched ridge that removes the loss.
  snr    Flipped and masked cues on one axis, SNR = a^2 / Var(xi). If the cliff is the linear cue
         map's noise gain, recovery depends on the cue only through its SNR.
  blind  A ridge that does not know the noise, alpha = c P, against the noise-matched alpha, and
         what it costs a clean cue.
The published curves are read from the vector paths in the SI PDF (the plotted polylines, not
read by eye), so the comparison is with the plotted values themselves.

What it is not: a new model. It runs `loci.scaffold` and `loci.memory` as they are, with the
paper's placement (item j at address j), Nh = 400 and periods (3, 4, 5).

    uv run python bench/paper_figs.py [--seeds 10] [--only s7,s8,snr,blind]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from cliff import mask
from scipy.linalg import cho_factor, cho_solve

from loci.memory import CueMaps, flip, mmse_alpha
from loci.scaffold import Scaffold, relu

OUT = Path(__file__).resolve().parents[1] / "results" / "paper_figs.json"
NS = 3_600  # the paper's Ns = Npos

# ---- the published curves ----------------------------------------------------------------------
# SI Fig S7-right, Vector-HaSH (black), 10% flips: the polyline has a vertex every 50 patterns from
# P = 1; these are the vertices on the notebook's grid P = 1, 201, ..., 3,401.
S7_P = tuple(range(1, NS + 1, 200)) + (NS,)
S7_NOISY = dict(zip(range(1, 3_402, 200), (
    1.0, 0.9942, 0.9680, 0.5874, 0.3568, 0.2523, 0.1907, 0.1511, 0.1223, 0.0989, 0.0786, 0.0606,
    0.0443, 0.0301, 0.0175, 0.0087, 0.0029, 0.0005), strict=True))
# SI Fig S7-left (clean cues): an adaptive polyline, so it is interpolated at the grid above.
S7_CLEAN_VERTICES = (
    (11, 1.0), (421, 0.9993), (431, 0.9956), (441, 0.9851), (461, 0.9473), (481, 0.8979),
    (511, 0.8063), (561, 0.6785), (581, 0.6383), (611, 0.5812), (631, 0.5490), (661, 0.5057),
    (681, 0.4795), (701, 0.4568), (731, 0.4260), (751, 0.4077), (791, 0.3743), (831, 0.3463),
    (871, 0.3225), (901, 0.3063), (941, 0.2871), (971, 0.2742), (1011, 0.2589), (1071, 0.2386),
    (1141, 0.2191), (1221, 0.2001), (1321, 0.1803), (1371, 0.1718), (1571, 0.1450), (1651, 0.1363),
    (1861, 0.1177), (2021, 0.1068), (2361, 0.0891), (2591, 0.0802), (3201, 0.0633), (3471, 0.0578),
    (3591, 0.0557))
# SI Fig S8, p(correct) at flip rates 0, 0.05, ..., 0.5, one curve per number stored.
S8_GRID = tuple(round(0.05 * k, 2) for k in range(11))
S8_PUBLISHED = {
    250: (1.0, 1.0, 0.9996, 0.9972, 0.9868, 0.9716, 0.9324, 0.7756, 0.4284, 0.0540, 0.0004),
    500: (1.0, 0.9996, 0.9942, 0.9836, 0.9528, 0.8880, 0.7348, 0.4666, 0.1550, 0.0164, 0.0004),
    1_000: (1.0, 0.9957, 0.9709, 0.9150, 0.8001, 0.6077, 0.3581, 0.1576, 0.0386, 0.0042, 0.0005),
    2_000: (1.0, 0.9545, 0.7971, 0.5627, 0.3336, 0.1705, 0.0693, 0.0236, 0.0061, 0.0017, 0.0002),
    3_000: (1.0, 0.6064, 0.2514, 0.1003, 0.0399, 0.0170, 0.0086, 0.0035, 0.0017, 0.0008, 0.0004),
}
S8_FLIPS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
# Cues per (P, flip) per seed: several noise draws per pattern at small P, as the notebook's 100
# `basin_runs` are, so every point rests on >= 3,000 recalls a seed.
TRIALS = 3_000

# ---- the two open questions --------------------------------------------------------------------
SMALL = 1_000  # Ns for snr and blind: small enough that P can pass Ns inside 3,600 addresses
SNR_RATIOS = (0.4, 0.8, 1.0, 1.25)
# Each family spans SNR ~0.04-50; the grids are dense where recovery turns over, so that the
# interpolation between them is not what the gap measures.
SNR_FLIPS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
             0.22, 0.25, 0.28, 0.31, 0.35, 0.4)
SNR_MASKS = (0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
             0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
BLIND_RATIOS = (0.4, 0.8, 1.0, 1.5)
BLIND_FLIPS = (0.0, 0.05, 0.1, 0.2)
BLIND_MASKS = (0.25, 0.5)
BLIND_C = (0.1, 0.3, 1.0)


def mi(overlap: np.ndarray) -> np.ndarray:
    """MI per bit of a binary symmetric channel with overlap m: 1 - H2((1 + m) / 2)."""
    p = np.clip((1 + np.asarray(overlap, dtype=float)) / 2, 1e-15, 1 - 1e-15)
    return 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)


def snr_flip(rate: float) -> float:
    return (1 - 2 * rate) ** 2 / (4 * rate * (1 - rate))


def snr_mask(rate: float) -> float:
    return (1 - rate) / rate


def recover(scaffold: Scaffold, W_hs: np.ndarray, cues: np.ndarray) -> np.ndarray:
    """The address each cue settles on."""
    phases, _ = scaffold.settle(relu(W_hs @ cues))
    return scaffold.index(phases)


# ---- s7 ----------------------------------------------------------------------------------------


def s7(seed: int) -> list[dict]:
    """MI per input bit against P, every stored pattern cued, as the notebook does."""
    scaffold = Scaffold(seed=seed)
    rng = np.random.default_rng([7, seed])
    patterns = np.sign(rng.standard_normal((NS, scaffold.addresses)))
    cues = np.concatenate([patterns, flip(patterns, 0.1, rng)], axis=1)
    # Nested stored sets, so every Gram block is a corner of one product. The pseudo-inverse is
    # (S^T S)^-1 S^T for P <= Ns: one Cholesky per P instead of an SVD (~50x faster at 3,600).
    gram, cross = patterns.T @ patterns, patterns.T @ cues
    rows = []
    for count in S7_P:
        started = time.time()
        place, stored = scaffold.H[:, :count], patterns[:, :count]
        W_sh = stored @ np.linalg.pinv(place)
        pick = np.r_[np.arange(count), NS + np.arange(count)]
        weights = cho_solve(cho_factor(gram[:count, :count]), cross[:count, pick])
        phases, h = scaffold.settle(relu(place @ weights))
        got = scaffold.index(phases)
        l1 = np.abs(np.sign(W_sh @ h) - np.tile(stored, 2)).mean(axis=0) / 2  # per pattern
        for k, cue in enumerate(("clean", "flip0.1")):
            part = slice(k * count, (k + 1) * count)
            m = 1 - 2 * l1[part]
            rows.append({"seed": seed, "P": count, "cue": cue,
                         "recovery": float((got[part] == np.arange(count)).mean()),
                         "overlap": float(m.mean()),
                         "mi_pooled": float(mi(m.mean())),
                         "mi_per_pattern": float(mi(m).mean())})
        print(f"s7 seed {seed} P {count} ({time.time() - started:.1f}s)", flush=True)
    return rows


# ---- s8 ----------------------------------------------------------------------------------------


def mmse_map(maps: CueMaps, place: np.ndarray, patterns: np.ndarray, rate: float) -> np.ndarray:
    """The noise-matched ridge. At a 50% flip the cue says nothing and alpha -> inf, where the
    ridge is proportional to H_a S^T - Hebbian; the scale is irrelevant to the snap."""
    if rate >= 0.5:
        return place @ patterns.T
    return maps.map(place, mmse_alpha(patterns.shape[1], flip_rate=rate))


def s8(seed: int) -> list[dict]:
    scaffold = Scaffold(seed=seed)
    rng = np.random.default_rng([8, seed])
    patterns = np.sign(rng.standard_normal((NS, max(S8_PUBLISHED))))
    rows = []
    for count in S8_PUBLISHED:
        started = time.time()
        stored, place = patterns[:, :count], scaffold.H[:, :count]
        maps = CueMaps(stored)
        draws = -(-TRIALS // count)
        truth, targets = np.tile(stored, draws), np.tile(np.arange(count), draws)
        pinv = maps.map(place, 0.0)
        for rate in S8_FLIPS:
            cues = flip(truth, rate, rng)
            for rule, W_hs in (("pinv", pinv), ("mmse", mmse_map(maps, place, stored, rate))):
                got = recover(scaffold, W_hs, cues)
                rows.append({"seed": seed, "P": count, "flip": rate, "rule": rule,
                             "recovery": float((got == targets).mean())})
        print(f"s8 seed {seed} P {count} ({time.time() - started:.1f}s)", flush=True)
    return rows


# ---- snr and blind -----------------------------------------------------------------------------


def _cue_sets(truth: np.ndarray, flips, masks, rng) -> list[tuple[str, float, np.ndarray, dict]]:
    """(family, rate, cues, the statistics `mmse_alpha` is tuned for) per cue level."""
    out = [("flip", f, flip(truth, f, rng), {"flip_rate": f}) for f in flips]
    return out + [("mask", r, mask(truth, r, rng), {"mask_rate": r}) for r in masks]


def snr(seed: int) -> list[dict]:
    scaffold = Scaffold(seed=seed)
    rng = np.random.default_rng([9, seed])
    patterns = np.sign(rng.standard_normal((SMALL, int(max(SNR_RATIOS) * SMALL))))
    rows = []
    for ratio in SNR_RATIOS:
        started = time.time()
        count = round(ratio * SMALL)
        stored, place = patterns[:, :count], scaffold.H[:, :count]
        maps = CueMaps(stored)
        draws = -(-SMALL // count)
        truth, targets = np.tile(stored, draws), np.tile(np.arange(count), draws)
        pinv = maps.map(place, 0.0)
        for family, rate, cues, stats in _cue_sets(truth, SNR_FLIPS, SNR_MASKS, rng):
            level = snr_flip(rate) if family == "flip" else snr_mask(rate)
            mmse = maps.map(place, mmse_alpha(count, **stats))
            for rule, W_hs in (("pinv", pinv), ("mmse", mmse)):
                got = recover(scaffold, W_hs, cues)
                rows.append({"seed": seed, "ratio": ratio, "P": count, "family": family,
                             "rate": rate, "snr": level, "rule": rule,
                             "recovery": float((got == targets).mean())})
        print(f"snr seed {seed} P/Ns {ratio} ({time.time() - started:.1f}s)", flush=True)
    return rows


def blind(seed: int) -> list[dict]:
    scaffold = Scaffold(seed=seed)
    rng = np.random.default_rng([10, seed])
    patterns = np.sign(rng.standard_normal((SMALL, int(max(BLIND_RATIOS) * SMALL))))
    rows = []
    for ratio in BLIND_RATIOS:
        started = time.time()
        count = round(ratio * SMALL)
        stored, place = patterns[:, :count], scaffold.H[:, :count]
        maps = CueMaps(stored)
        draws = -(-SMALL // count)
        truth, targets = np.tile(stored, draws), np.tile(np.arange(count), draws)
        fixed = {"pinv": maps.map(place, 0.0)}
        fixed |= {f"c{c}": maps.map(place, c * count) for c in BLIND_C}
        for family, rate, cues, stats in _cue_sets(truth, BLIND_FLIPS, BLIND_MASKS, rng):
            rules = fixed | {"matched": maps.map(place, mmse_alpha(count, **stats))}
            for rule, W_hs in rules.items():
                got = recover(scaffold, W_hs, cues)
                rows.append({"seed": seed, "ratio": ratio, "P": count, "cue": f"{family}{rate}",
                             "rule": rule, "recovery": float((got == targets).mean())})
        print(f"blind seed {seed} P/Ns {ratio} ({time.time() - started:.1f}s)", flush=True)
    return rows


# ---- the criteria (docs/PREREG.md, E1 items 1-3) -----------------------------------------------


def _mean(rows: list[dict], key: str = "recovery", **match) -> float:
    vals = [r[key] for r in rows if all(r[k] == v for k, v in match.items())]
    return float(np.mean(vals))


def _sd(rows: list[dict], key: str = "recovery", **match) -> float:
    return float(np.std([r[key] for r in rows if all(r[k] == v for k, v in match.items())]))


def summarise_s7(rows: list[dict]) -> dict:
    clean_ref = np.interp(S7_P, *zip(*S7_CLEAN_VERTICES, strict=True))
    table, worst = [], {"clean": (0.0, None), "flip0.1": (0.0, None)}
    for count, clean_pub in zip(S7_P, clean_ref, strict=True):
        for cue in ("clean", "flip0.1"):
            entry = {"P": count, "cue": cue}
            for key in ("recovery", "mi_pooled", "mi_per_pattern"):
                entry[key] = _mean(rows, key, P=count, cue=cue)
                entry[key + "_sd"] = _sd(rows, key, P=count, cue=cue)
            published = S7_NOISY.get(count) if cue == "flip0.1" else float(clean_pub)
            if count > 3_591:  # past the last plotted vertex
                published = None
            entry["published"] = published
            if published is not None:
                entry["deviation"] = entry["mi_pooled"] - published
                if abs(entry["deviation"]) > abs(worst[cue][0]):
                    worst[cue] = (entry["deviation"], count)
            table.append(entry)
    return {"table": table,
            "max_deviation": {cue: {"value": v, "P": p} for cue, (v, p) in worst.items()},
            "pass": all(abs(v) <= 0.03 for v, _ in worst.values())}


def summarise_s8(rows: list[dict]) -> dict:
    table, worst = [], (0.0, None)
    for count, curve in S8_PUBLISHED.items():
        for rate in S8_FLIPS:
            entry = {"P": count, "flip": rate,
                     "published": curve[S8_GRID.index(rate)]}
            for rule in ("pinv", "mmse"):
                entry[rule] = _mean(rows, P=count, flip=rate, rule=rule)
                entry[rule + "_sd"] = _sd(rows, P=count, flip=rate, rule=rule)
            entry["deviation"] = entry["pinv"] - entry["published"]
            if abs(entry["deviation"]) > abs(worst[0]):
                worst = (entry["deviation"], (count, rate))
            table.append(entry)
    return {"table": table, "max_deviation": {"value": worst[0], "at_P_flip": worst[1]},
            "pass": abs(worst[0]) <= 0.03}


def summarise_snr(rows: list[dict]) -> dict:
    out = {}
    for ratio in SNR_RATIOS:
        for rule in ("pinv", "mmse"):
            curves = {}
            for family, rates in (("flip", SNR_FLIPS), ("mask", SNR_MASKS)):
                pts = sorted(((snr_flip(q) if family == "flip" else snr_mask(q)),
                              _mean(rows, ratio=ratio, rule=rule, family=family, rate=q))
                             for q in rates)
                curves[family] = np.array(pts)
            lo = max(c[:, 0].min() for c in curves.values())
            hi = min(c[:, 0].max() for c in curves.values())
            grid = np.geomspace(lo, hi, 400)
            f, m = (np.interp(np.log(grid), np.log(c[:, 0]), c[:, 1]) for c in curves.values())
            gap = np.abs(f - m)
            out[f"{ratio}/{rule}"] = {"max_gap": float(gap.max()),
                                     "at_snr": float(grid[gap.argmax()]),
                                     "flip_there": float(f[gap.argmax()]),
                                     "mask_there": float(m[gap.argmax()]),
                                     "snr_range": [float(lo), float(hi)],
                                     "curves": {k: v.round(4).tolist() for k, v in curves.items()}}
    passed = {rule: all(out[f"{r}/{rule}"]["max_gap"] <= 0.05 for r in SNR_RATIOS)
              for rule in ("pinv", "mmse")}
    return {"by_ratio": out, "pass": passed}


def summarise_blind(rows: list[dict]) -> dict:
    cues = [f"flip{f}" for f in BLIND_FLIPS] + [f"mask{m}" for m in BLIND_MASKS]
    table = {f"{ratio}/{cue}": {rule: _mean(rows, ratio=ratio, cue=cue, rule=rule)
                                for rule in ("pinv", "matched", *(f"c{c}" for c in BLIND_C))}
             for ratio in BLIND_RATIOS for cue in cues}
    verdicts = {}
    for c in BLIND_C:
        rule = f"c{c}"
        short = {k: v[rule] - v["matched"] for k, v in table.items()}
        flips_only = {k: s for k, s in short.items() if "/flip" in k}
        clean_cost = {r: table[f"{r}/flip0.0"]["pinv"] - table[f"{r}/flip0.0"][rule]
                      for r in BLIND_RATIOS}
        worst_flip = min(flips_only, key=flips_only.get)
        worst_any = min(short, key=short.get)
        verdicts[rule] = {
            "worst_shortfall_flips": {"value": flips_only[worst_flip], "at": worst_flip},
            "worst_shortfall_all_cues": {"value": short[worst_any], "at": worst_any},
            "clean_cost": clean_cost,
            "pass": flips_only[worst_flip] >= -0.05
            and all(v <= 0.02 for r, v in clean_cost.items() if r <= 1.0)}
    return {"table": table, "by_c": verdicts, "pass": any(v["pass"] for v in verdicts.values())}


def figure(result: dict) -> None:
    """Four panels in the style of `figures.py`: measured solid, published dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures import ACCENT, INK, MUTED

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    ax = axes[0]
    table = result["s7"]["summary"]["table"]
    for cue, color, label in (("clean", MUTED, "clean"), ("flip0.1", INK, "10% flipped")):
        pts = [e for e in table if e["cue"] == cue]
        ax.plot([e["P"] for e in pts], [e["mi_pooled"] for e in pts], "-", color=color, lw=1.6,
                label=f"{label}, measured")
        pub = [e for e in pts if e["published"] is not None]
        ax.plot([e["P"] for e in pub], [e["published"] for e in pub], "--", color=color, lw=1.0,
                label=f"{label}, published")
    noisy = [e for e in table if e["cue"] == "flip0.1"]
    ax.plot([e["P"] for e in noisy], [e["mi_per_pattern"] for e in noisy], ":", color=ACCENT,
            lw=1.4, label="10% flipped, MI per pattern")
    ax.set(title="SI Fig S7: MI per input bit", xlabel="patterns stored  (Ns = 3,600)")
    ax.legend(frameon=False, fontsize=7)
    s8_table = result["s8"]["summary"]["table"]
    for ax, rule, color, title in ((axes[1], "pinv", ACCENT, "SI Fig S8: pseudo-inverse"),
                                   (axes[2], "mmse", INK, "S8 with the noise-matched ridge")):
        for k, count in enumerate(S8_PUBLISHED):
            pts = [e for e in s8_table if e["P"] == count]
            shade = 0.3 + 0.7 * k / (len(S8_PUBLISHED) - 1)
            ax.plot([e["flip"] for e in pts], [e[rule] for e in pts], "-", color=color,
                    alpha=shade, lw=1.6, label=f"P = {count:,}")
            if rule == "pinv":
                ax.plot(S8_GRID, S8_PUBLISHED[count], "--", color=MUTED, lw=0.9,
                        label="published" if k == 0 else None)
        ax.set(title=title, xlabel="fraction of cue bits flipped")
        ax.legend(frameon=False, fontsize=7)
    axes[1].set_ylabel("right address recovered")
    ax = axes[3]
    curves = result["snr"]["summary"]["by_ratio"]
    for rule, color in (("pinv", ACCENT), ("mmse", INK)):
        for family, dash in (("flip", "-"), ("mask", "--")):
            pts = np.array(curves[f"0.8/{rule}"]["curves"][family])
            ax.plot(pts[:, 0], pts[:, 1], dash, color=color, lw=1.4,
                    label=f"{rule}, {'flipped' if family == 'flip' else 'masked'}")
    ax.set(xscale="log", title="Flips and masks, one SNR axis (P/Ns 0.8)",
           xlabel="cue SNR  a^2 / Var(xi)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=160)


SECTIONS = {"s7": (s7, summarise_s7), "s8": (s8, summarise_s8),
            "snr": (snr, summarise_snr), "blind": (blind, summarise_blind)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--only", default=",".join(SECTIONS))
    args = parser.parse_args()
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.only.split(","):
        run, summarise = SECTIONS[name]
        started = time.time()
        rows = [row for seed in range(args.seeds) for row in run(seed)]
        result[name] = {"seeds": args.seeds, "summary": summarise(rows), "rows": rows,
                        "seconds": round(time.time() - started, 1)}
        print(f"{name}: pass = {result[name]['summary']['pass']} "
              f"({result[name]['seconds']}s)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result))
    if all(name in result for name in ("s7", "s8", "snr")):
        figure(result)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
