"""Round 2's pre-registered criteria (docs/PREREG.md: E4, T3 and E5), computed from the result files.

    uv run python bench/criteria.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOAD, FLIP = 0.8, 0.1
KINDS = ("factored", "cards", "four", "hierarchy", "lifelog")


def paired(a: dict[int, float], b: dict[int, float], draws: int = 10_000) -> tuple[float, float, float]:
    """Mean paired difference a - b over seeds, with its bootstrap 95% interval."""
    seeds = sorted(set(a) & set(b))
    diff = np.array([a[s] - b[s] for s in seeds])
    means = diff[np.random.default_rng(0).integers(0, len(diff), (draws, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def e4(rows: list[dict]) -> dict:
    def at(kind: str, placement: str, key: str = "recall") -> dict[int, float]:
        return {r["seed"]: r[key] for r in rows if r["content"] == kind and r["load"] == LOAD
                and r["flip"] == FLIP and r["placement"] == placement}

    out: dict = {}
    for kind in KINDS:
        if not at(kind, "random"):
            continue
        cell = {p: float(np.mean(list(at(kind, p).values()))) for p in
                ("random", "k-means", "encode", "encode+replay", "random+replay", "oracle", "oracle+replay")
                if at(kind, p)}
        s1 = paired(at(kind, "encode+replay"), at(kind, "k-means"))
        s3 = paired(at(kind, "encode"), at(kind, "random"))
        verdict = {"recall": cell,
                   "S1 encode+replay - k-means": s1, "S1": s1[0] >= 0.05 and s1[1] > 0,
                   "S3 encode - random": s3, "S3": s3[0] >= 0.15,
                   "ablation random+replay - encode+replay": paired(at(kind, "random+replay"),
                                                                     at(kind, "encode+replay"))}
        if at(kind, "oracle"):
            s2 = paired(at(kind, "encode+replay"), at(kind, "oracle"))
            verdict |= {"S2 encode+replay - oracle": s2, "S2": s2[1] >= -0.03}
        out[kind] = verdict
    errors = np.array([r["theory"] - r["recall"] for r in rows])
    out["T3"] = {"rows": len(errors), "MAE": float(np.abs(errors).mean()), "bias": float(errors.mean()),
                 "max": float(np.abs(errors).max()), "within 0.05": float(np.mean(np.abs(errors) <= 0.05)),
                 "pass": bool(np.abs(errors).mean() <= 0.02 and np.abs(errors).max() <= 0.06)}
    out["T3 by content"] = {kind: float(np.abs([r["theory"] - r["recall"] for r in rows
                                                if r["content"] == kind]).mean())
                            for kind in KINDS if any(r["content"] == kind for r in rows)}
    return out


def e5(rows: list[dict]) -> dict:
    """I1-I6 on bench/imagine.py's rows (seeds 40-49)."""
    def at(kind: str, placement: str, decoder: str, key: str, rate: float = FLIP) -> dict[int, float]:
        return {r["seed"]: r[key] for r in rows if r["content"] == kind and r["placement"] == placement
                and r["decoder"] == decoder and r["flip"] == rate}

    def mean(values: dict[int, float]) -> float:
        return float(np.mean(list(values.values())))

    kinds = [k for k in ("factored", "cards", "lifelog") if any(r["content"] == k for r in rows)]
    out: dict = {}
    floor = {"factored": 0.90, "cards": 0.90, "lifelog": 0.80}
    i1 = {k: {"oracle": mean(at(k, "oracle", "nearest", "construction")),
              "random": mean(at(k, "random", "nearest", "construction"))} for k in kinds}
    out["I1"] = {"construction": i1,
                 "pass": all(v["oracle"] >= floor[k] and v["random"] <= 0.05 for k, v in i1.items())}
    i2 = {f"{k} {rate}": paired(at(k, "oracle", "nearest", "recall", rate), at(k, "oracle", "stored", "recall", rate))
          for k in kinds for rate in (0.1, 0.2)}
    out["I2"] = {"oracle recall, nearest - stored": i2, "pass": all(d[0] >= -0.01 and d[1] >= -0.02 for d in i2.values())}
    i3, passes = {}, []
    for k in kinds:
        recall = paired(at(k, "encode+replay", "nearest", "recall"), at(k, "oracle", "nearest", "recall"))
        built = paired(at(k, "encode+replay", "nearest", "construction"), at(k, "oracle", "nearest", "construction"))
        i3[k] = {"recall, encode+replay - oracle": recall, "construction, encode+replay - oracle": built}
        if k in ("factored", "cards"):
            passes.append(abs(recall[0]) <= 0.02 and built[2] <= -0.4)
    out["I3"] = {**i3, "pass": all(passes)}
    anneal = {}
    for k in kinds:
        picked = [r for r in rows if r["content"] == k and r["placement"] == "encode+anneal"
                  and r["decoder"] == "nearest" and r["flip"] == FLIP]
        aligned = [r for r in picked if min(r["nmi"]) >= 0.8]
        other = [r for r in picked if min(r["nmi"]) < 0.8]
        anneal[k] = {"seeds aligned": f"{len(aligned)} of {len(picked)}",
                     "construction when aligned": float(np.mean([r["construction"] for r in aligned])) if aligned else None,
                     "construction otherwise": float(np.mean([r["construction"] for r in other])) if other else None,
                     "recall": float(np.mean([r["recall"] for r in picked]))}
    out["encode+anneal (reported)"] = anneal
    i4 = {k: paired(at(k, "oracle+replay", "nearest", "construction"), at(k, "oracle", "nearest", "construction"))
          for k in kinds}
    out["I4"] = {"construction, oracle+replay - oracle": i4, "pass": "cards" in i4 and i4["cards"][2] < 0}
    cells = {}
    for r in rows:
        if r["placement"] != "knn":
            cells.setdefault((r["content"], r["placement"], r["decoder"], r["flip"]), []).append(
                (r["construction"], r["false recall"]))
    points = np.array([np.mean(v, axis=0) for v in cells.values()])
    r_value = float(np.corrcoef(points[:, 0], points[:, 1])[0, 1])
    false_oracle = {k: mean(at(k, "oracle", "nearest", "false recall")) for k in kinds}
    out["I5"] = {"cells": len(points), "r": r_value, "oracle false recall": false_oracle,
                 "pass": r_value >= 0.9 and all(v >= 0.8 for v in false_oracle.values())}
    i6 = {k: {"recollection": mean(at(k, "oracle", "nearest", "recollection d' recombined")),
              "familiarity": mean(at(k, "oracle", "nearest", "familiarity d' recombined")),
              "familiarity, stored decoder": mean(at(k, "oracle", "stored", "familiarity d' recombined"))}
          for k in kinds}
    out["I6"] = {"d' for recombined events": i6,
                 "pass": all(i6[k]["recollection"] >= 5 and i6[k]["familiarity"] <= 4 for k in ("factored", "cards")
                             if k in i6)}
    table = {}
    for k in kinds:
        for placement in ("random", "oracle", "encode+replay", "encode+anneal", "oracle+replay", "knn"):
            for decoder in ("snap", "nearest", "stored", "knn"):
                picked = [r for r in rows if r["content"] == k and r["placement"] == placement
                          and r["decoder"] == decoder and r["flip"] == FLIP]
                if picked:
                    table[f"{k} | {placement} | {decoder}"] = {
                        key: round(float(np.mean([r[key] for r in picked])), 3)
                        for key in ("recall", "construction", "false recall", "familiarity d' recombined",
                                    "recollection d' recombined") if key in picked[0]}
    out["table, 10% flips"] = table
    return out


def e5_peak(rows: list[dict]) -> dict:
    """I7 on bench/imagine.py --nh."""
    def construction(cells: int, ridge: float) -> float:
        return float(np.mean([r["construction"] for r in rows if r["Nh"] == cells and r["readout_ridge"] == ridge
                              and r["decoder"] == "snap"]))

    curve = {f"Nh {n}": {"pinv": construction(n, 0.0), "ridge 1%": construction(n, 0.01)}
             for n in sorted({r["Nh"] for r in rows})}
    return {"construction (snap)": curve,
            "pass": construction(800, 0.0) <= 0.05 and construction(400, 0.0) >= 0.8
            and construction(1_200, 0.0) >= 0.8 and construction(800, 0.01) >= 0.9}


def main() -> None:
    report = {}
    path = RESULTS / "consolidate.json"
    if path.exists():
        report["E4"] = e4(json.loads(path.read_text()))
    if (RESULTS / "imagine.json").exists():
        report["E5"] = e5(json.loads((RESULTS / "imagine.json").read_text()))
    if (RESULTS / "imagine-nh.json").exists():
        report["E5 I7"] = e5_peak(json.loads((RESULTS / "imagine-nh.json").read_text()))
    (RESULTS / "criteria.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
