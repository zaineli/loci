"""Round 2's pre-registered criteria (docs/PREREG.md, E4 and T3), computed from the result files.

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


def main() -> None:
    report = {}
    path = RESULTS / "consolidate.json"
    if path.exists():
        report["E4"] = e4(json.loads(path.read_text()))
    (RESULTS / "criteria.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
