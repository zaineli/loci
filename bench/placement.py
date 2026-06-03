"""E2 - where a memory is put decides what survives.

The same stored patterns and the same noisy cues, bound to different addresses. Placements:
  sequential  the paper's: item j at address j
  random      a uniformly random address each
  kmeans      residual k-means on the patterns, per module (raw similarity, not learned)
  learned     per-module groups minimising the noise law, by swap descent (`loci.place.learned`)
  oracle      the true factors as the address
  error       learned on the error law (added by PREREG amendment 4; judged on fresh seeds)
on the grid scaffold and on the no-product control: one module of 3,600 phases, so the same number
of place cells, random sparse place codes and a nearest-code snap, but no shared phases. Reads: the
paper's pseudo-inverse, the noise-matched ridge, ridge + decode to the best *stored* address, and
(exploratory) ridge + the nearest of all 3,600 place codes, with no module snap.
kNN over the stored patterns (P x Ns bits, not a Vector-HaSH) is the exemplar reference.

Recorded per (content, seed, load, flip rate, scaffold, placement): recovery by each read; on the
grid, per-module accuracy and the measured noise law q^T Gamma q; gist (the factors decoded from the
read-out when the address is wrong); facet completion (the cue leaves out factor B, the read-out
must supply it). Criteria: docs/PREREG.md, E2; `--verdict` prints them from the result file.

    uv run --extra bench python bench/placement.py [--content factored|lifelog] [--quick] [--jobs 8]
    uv run --extra bench python bench/placement.py --verdict
"""

from __future__ import annotations

import os

for _threads in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")  # one BLAS thread per worker; parallel over seeds instead

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from loci import place
from loci.content import factored
from loci.memory import CueMaps, flip, mmse_alpha
from loci.scaffold import Scaffold, relu

sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = Path(__file__).resolve().parents[1] / "results"
DIM = 1_000
LOADS = (0.2, 0.4, 0.6, 0.8, 0.9)
FLIPS = (0.1, 0.2)
SEEDS = 20  # 0-9 for the registered criteria, 10-19 fresh for amendment 4's
PLACEMENTS = ("sequential", "random", "kmeans", "learned", "oracle")
# Added after the life-log pilot (docs/PREREG.md, amendment 4): learned on the error law. Judged on
# fresh seeds only; the registered criteria use seeds below FRESH and the registered placements.
ADDED = ("error",)
FRESH = 10
READS = ("pinv", "ridge", "stored", "nearest")
# The headline condition of the criteria.
LOAD, FLIP, READ = 0.8, 0.1, "stored"


def _content(kind: str, count: int, rng: np.random.Generator):
    if kind == "factored":
        return factored(DIM, count, rng)
    from lifelog import lifelog

    return lifelog(DIM, count, rng)


def _law(kernel: np.ndarray, phases: np.ndarray) -> list[float]:
    """Per module, the mean over occupied phases of q^T K q."""
    return [place.law(kernel, phases[:, m]) for m in range(phases.shape[1])]


def _reads(scaffold: Scaffold, maps: CueMaps, stored: np.ndarray, where: np.ndarray, cues: np.ndarray,
           alpha: float) -> dict:
    """Recovery by pinv, ridge and ridge + stored decode, and what the ridge read-out holds."""
    place_codes = scaffold.H[:, where]
    out: dict = {}
    for read, strength in (("pinv", 0.0), ("ridge", alpha)):
        h0 = relu(maps.map(place_codes, strength) @ cues)
        phases, h = scaffold.settle(h0)
        got = scaffold.index(phases)
        out[read] = float(np.mean(got == where))
        out[f"{read}_modules"] = (scaffold.phases[got] == scaffold.phases[where]).mean(axis=0).tolist()
        if read == "ridge":
            out["stored"] = float(np.mean((place_codes.T @ h0).argmax(axis=0) == np.arange(len(where))))
            out["nearest"] = float(np.mean((scaffold.H.T @ h0).argmax(axis=0) == where))
            out["_wrong"] = got != where
            out["_readout"] = np.sign(stored @ np.linalg.pinv(place_codes) @ h)
    return out


def condition(kind: str, seed: int, load: float) -> list[dict]:
    """Every flip rate, scaffold and placement for one stored set."""
    count = round(load * DIM)
    content = _content(kind, count, np.random.default_rng(10_000 * seed + count))
    stored, factors = content.patterns, content.factors
    maps = CueMaps(stored)
    grid, flat = Scaffold(seed=seed), Scaffold(periods=(60,), seed=seed)
    fixed = {
        "sequential": place.sequential(count),
        "random": place.scattered(count, grid, np.random.default_rng(seed + 1)),
        "kmeans": place.kmeans(stored, grid, np.random.default_rng(seed + 2))[0],
        "oracle": place.oracle(grid, factors),
    }
    facet = content.cue(without=1)
    knn_facet = float(np.mean(factors[(stored.T @ facet).argmax(axis=0), 1] == factors[:, 1]))
    rows = []
    for rate in FLIPS:
        alpha = mmse_alpha(count, flip_rate=rate)
        cues = flip(stored, rate, np.random.default_rng(20_000 * seed + count + int(1_000 * rate)))
        nearest = (stored.T @ cues).argmax(axis=0)
        knn_wrong = nearest != np.arange(count)
        base = {"content": kind, "seed": seed, "load": load, "P": count, "flip": rate, "alpha": alpha}
        rows.append({**base, "scaffold": "none", "placement": "knn", "stored": float(np.mean(~knn_wrong)),
                     "gist": [float(np.mean(factors[nearest[knn_wrong], f] == factors[knn_wrong, f]))
                              if knn_wrong.any() else None for f in range(3)],
                     "facet": knn_facet})
        started = time.time()
        learned, labels = place.learned(stored, grid, np.random.default_rng(seed + 3), alpha)
        error, error_labels = place.learned(stored, grid, np.random.default_rng(seed + 4), alpha, error=True,
                                            warm=labels)
        arms = {**fixed, "learned": learned, "error": error}
        kernels = {"pinv": place.gamma(stored, 0.0), "ridge": place.gamma(stored, alpha),
                   "error": place.precision(stored, alpha)}
        for name in PLACEMENTS + ADDED:
            where = arms[name]
            assert len(np.unique(where)) == count, name
            for scaffold_name, scaffold in (("grid", grid), ("flat", flat)):
                got = _reads(scaffold, maps, stored, where, cues, alpha)
                wrong, readout = got.pop("_wrong"), got.pop("_readout")
                row = {**base, "scaffold": scaffold_name, "placement": name, **got}
                if scaffold_name == "grid":
                    phases = grid.phases[where]
                    for key, kernel in kernels.items():
                        row[f"law_{key}"] = _law(kernel, phases)
                    row["wrong"] = int(wrong.sum())
                    row["gist"] = [float(np.mean(content.decode(readout[:, wrong], f) == factors[wrong, f]))
                                   if wrong.any() else None for f in range(3)]
                    _, h = grid.settle(relu(maps.map(grid.H[:, where], alpha) @ facet))
                    completed = np.sign(stored @ np.linalg.pinv(grid.H[:, where]) @ h)
                    row["facet"] = float(np.mean(content.decode(completed, 1) == factors[:, 1]))
                    if name in ("learned", "error"):
                        row["found"] = _agreement(labels if name == "learned" else error_labels, factors)
                else:
                    row = {k: v for k, v in row.items() if not k.endswith("_modules")}
                rows.append(row)
        print(f"{kind} seed {seed} P {count} flip {rate} ({time.time() - started:.1f}s)", flush=True)
    return rows


def _agreement(labels: np.ndarray, factors: np.ndarray) -> list[list[float]]:
    """Normalised mutual information of each learned module's groups with each true factor."""
    def nmi(x: np.ndarray, y: np.ndarray) -> float:
        joint = np.zeros((x.max() + 1, y.max() + 1))
        np.add.at(joint, (x, y), 1)
        joint /= joint.sum()
        px, py = joint.sum(1), joint.sum(0)
        nz = joint > 0
        mi = (joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum()
        hx, hy = -(px[px > 0] * np.log(px[px > 0])).sum(), -(py[py > 0] * np.log(py[py > 0])).sum()
        return float(2 * mi / (hx + hy))

    return [[nmi(labels[:, m], factors[:, f]) for f in range(3)] for m in range(labels.shape[1])]


def run(kind: str, seeds: int, loads: tuple[float, ...], jobs: int) -> list[dict]:
    work = [(kind, seed, load) for seed in range(seeds) for load in loads]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return [row for rows in pool.map(condition, *zip(*work, strict=True)) for row in rows]


# ---- the criteria ------------------------------------------------------------------------------


def _cell(rows, content, load, rate, scaffold, placement, key) -> dict[int, float]:
    return {r["seed"]: r[key] for r in rows if r["content"] == content and r["load"] == load and
            r["flip"] == rate and r["scaffold"] == scaffold and r["placement"] == placement}


def _paired(a: dict[int, float], b: dict[int, float], rng=None, draws: int = 10_000) -> tuple[float, float, float]:
    """Mean paired difference a - b over seeds and its bootstrap 95% interval."""
    seeds = sorted(set(a) & set(b))
    diff = np.array([a[s] - b[s] for s in seeds])
    rng = rng or np.random.default_rng(0)
    means = diff[rng.integers(0, len(diff), (draws, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(rows: list[dict], content: str) -> dict:
    """The registered criteria, on the registered seeds and placements."""
    rows = [r for r in rows if r["seed"] < FRESH and r["placement"] not in ADDED]
    out: dict = {"content": content}
    at = lambda placement, key=READ, scaffold="grid", load=LOAD, rate=FLIP: _cell(
        rows, content, load, rate, scaffold, placement, key)
    # P1 - per-module accuracy against the measured law, pooled over placements and seeds.
    p1 = {}
    for read in ("pinv", "ridge"):
        grid_rows = [r for r in rows if r["content"] == content and r["load"] == LOAD and r["flip"] == FLIP
                     and r["scaffold"] == "grid" and r["placement"] in PLACEMENTS]
        p1[read] = [float(spearmanr([r[f"law_{read}"][m] for r in grid_rows],
                                    [r[f"{read}_modules"][m] for r in grid_rows]).statistic) for m in range(3)]
    out["P1"] = {"rho": p1, "pass": all(rho <= -0.8 for v in p1.values() for rho in v)}
    # P2 - oracle over random near the cue limit; every placement alike far from it.
    gain = _paired(at("oracle"), at("random"))
    spread = [np.mean(list(at(p, load=0.2).values())) for p in PLACEMENTS]
    out["P2"] = {"oracle-random": gain, "spread@0.2": float(max(spread) - min(spread)),
                 "pass": gain[0] >= 0.10 and gain[1] > 0 and max(spread) - min(spread) <= 0.02}
    # P3 - learned over k-means.
    gain = _paired(at("learned"), at("kmeans"))
    out["P3"] = {"learned-kmeans": gain, "pass": gain[0] >= 0.05 and gain[1] > 0}
    # P4 - aligned placement's gain on the grid over its gain on the no-product control.
    grid_gain = {s: at("oracle")[s] - at("random")[s] for s in at("oracle")}
    flat_gain = {s: at("oracle", scaffold="flat")[s] - at("random", scaffold="flat")[s] for s in at("oracle")}
    gain = _paired(grid_gain, flat_gain)
    out["P4"] = {"grid-flat": gain, "flat oracle-random": _paired(at("oracle", scaffold="flat"),
                                                                   at("random", scaffold="flat")),
                 "pass": gain[0] >= 0.05 and gain[1] > 0}
    # P5 - reported, no threshold: gist from wrong read-outs, facet completion, against kNN.
    out["P5"] = {p: {"gist": _gist(rows, content, p),
                     "facet": float(np.mean(list(at(p, key="facet", scaffold="grid" if p != "knn" else "none").values())))}
                 for p in (*PLACEMENTS, "knn")}
    # Equal-synapse worth: the load at which random matches oracle's recovery at LOAD, by the P/Ns
    # collapse (E1), as a fold-increase in Ns.
    out["worth"] = {}
    for read in ("pinv", "ridge", "stored"):
        target = np.mean(list(at("oracle", key=read).values()))
        curve = [np.mean(list(at("random", key=read, load=x).values())) for x in LOADS]
        if np.any(np.diff(curve) >= 0):
            out["worth"][read] = "n/a (random's curve is not strictly falling)"
        elif target > curve[0]:
            out["worth"][read] = f"> {LOAD / LOADS[0]:.0f}x"
        elif target < curve[-1]:
            out["worth"][read] = f"< {LOAD / LOADS[-1]:.2f}x"
        else:
            x = float(np.interp(-target, [-c for c in curve], LOADS))
            out["worth"][read] = f"{LOAD / x:.2f}x"
    return out


def _synapses(load: float) -> tuple[int, int, int]:
    """Weights in each memory: W_hs and W_sh (both Nh x Ns), W_hg and W_gh (Nh x Ng each); kNN
    keeps the stored patterns themselves, P x Ns bits."""
    place_cells, grid_cells = Scaffold.place_cells, sum(p * p for p in Scaffold.periods)
    content = 2 * place_cells * DIM
    return content + 2 * place_cells * grid_cells, content + 2 * place_cells * 3_600, int(load * DIM) * DIM


def _gist(rows: list[dict], content: str, placement: str) -> list[float] | None:
    """Mean over seeds of the factors read correctly from wrong recalls; None if nothing was wrong."""
    found = [r["gist"] for r in rows if r["content"] == content and r["load"] == LOAD and r["flip"] == FLIP
             and r["placement"] == placement and r["scaffold"] in ("grid", "none") and None not in r["gist"]]
    return np.mean(found, axis=0).round(3).tolist() if found else None


def added(rows: list[dict], content: str) -> dict:
    """Amendment 4's criteria, on the fresh seeds only."""
    rows = [r for r in rows if r["seed"] >= FRESH]
    at = lambda placement, key=READ: _cell(rows, content, LOAD, FLIP, "grid", placement, key)
    grid_rows = [r for r in rows if r["content"] == content and r["load"] == LOAD and r["flip"] == FLIP
                 and r["scaffold"] == "grid"]
    rho = [float(spearmanr([r["law_error"][m] for r in grid_rows], [r["ridge_modules"][m] for r in grid_rows]).statistic)
           for m in range(3)]
    out: dict = {"content": content, "seeds": sorted({r["seed"] for r in grid_rows})}
    out["X1"] = {"rho": rho, "pass": all(x <= -0.8 for x in rho)}
    over_noise, over_kmeans = _paired(at("error"), at("learned")), _paired(at("error"), at("kmeans"))
    out["X2"] = {"error-learned": over_noise, "error-kmeans": over_kmeans,
                 "ridge error-learned": _paired(at("error", "ridge"), at("learned", "ridge")),
                 "ridge error-kmeans": _paired(at("error", "ridge"), at("kmeans", "ridge"))}
    beats_kmeans = over_kmeans[1] > 0
    if content == "lifelog":
        out["X2"]["pass"] = over_noise[0] >= 0.05 and over_noise[1] > 0 and beats_kmeans
    else:
        out["X2"]["pass"] = over_noise[1] >= -0.02 and beats_kmeans
    return out


def table(rows: list[dict], content: str) -> str:
    seeds = len({r["seed"] for r in rows})
    lines = [f"{content}: recovery at P/Ns = {LOAD}, {FLIP:.0%} flips, mean over all {seeds} seeds (grid | flat)"]
    lines.append(f"{'':12}" + "".join(f"{k:>16}" for k in READS))
    for p in PLACEMENTS + ADDED:
        cells = []
        for k in READS:
            g, f = (np.mean(list(_cell(rows, content, LOAD, FLIP, s, p, k).values())) for s in ("grid", "flat"))
            cells.append(f"{g:.3f} | {f:.3f}")
        lines.append(f"{p:12}" + "".join(f"{c:>16}" for c in cells))
    knn = np.mean(list(_cell(rows, content, LOAD, FLIP, "none", "knn", "stored").values()))
    lines.append(f"{'knn':12}{'':>32}{knn:>16.3f}")
    grid, flat, knn_bits = _synapses(LOAD)
    lines.append(f"weights: grid {grid:,}, flat {flat:,}; knn keeps {knn_bits:,} bits")
    return "\n".join(lines)


def _plain(value):
    """numpy scalars (a bool from comparing numpy floats) as plain Python for json."""
    return value.item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", choices=("factored", "lifelog"), default="factored")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true", help="2 seeds, loads 0.2 and 0.8")
    parser.add_argument("--verdict", action="store_true", help="print the criteria from the result files")
    args = parser.parse_args()
    if args.verdict:
        report = {}
        for path in sorted(RESULTS.glob("placement-*.json")):
            if path.stem.endswith("quick") or path.stem.endswith("verdict"):
                continue
            rows = json.loads(path.read_text())
            content = rows[0]["content"]
            print(table(rows, content), "\n")
            report[content] = {"registered": verdict(rows, content), "amendment 4": added(rows, content)}
            print(json.dumps(report[content], indent=1, default=_plain), "\n")
        (RESULTS / "placement-verdict.json").write_text(json.dumps(report, indent=1, default=_plain))
        return
    loads = (0.2, 0.8) if args.quick else LOADS
    rows = run(args.content, 2 if args.quick else args.seeds, loads, args.jobs)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"placement-{args.content}{'-quick' if args.quick else ''}.json"
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} rows -> {out}")
    print(table(rows, args.content))


if __name__ == "__main__":
    main()
