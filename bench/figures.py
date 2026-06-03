"""Figures for the README, from the result files. Free; matplotlib only.

    uv run --extra bench python bench/figures.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
INK, MUTED, ACCENT = "#1f1f1f", "#8a8a8a", "#b8322a"
STYLE = {"pinv": (ACCENT, "-", "pseudo-inverse (the paper)"),
         "mmse": (INK, "-", "noise-matched ridge"),
         "hebb": (MUTED, "--", "Hebbian")}


def _table(rows: list[dict]) -> dict:
    out = collections.defaultdict(list)
    for r in rows:
        out[(r["Ns"], r["P"], r["cue"], r["rule"])].append(r["recovery"])
    return out


def cliff() -> None:
    rows = json.loads((RESULTS / "cliff.json").read_text())
    t = _table(rows)
    cues = [("flip0.05", "5% of bits flipped"), ("flip0.1", "10% flipped"),
            ("flip0.2", "20% flipped"), ("mask0.5", "half the cue unknown")]
    fig, axes = plt.subplots(1, len(cues), figsize=(13, 3.2), sharey=True)
    for ax, (cue, title) in zip(axes, cues, strict=True):
        for rule, (color, dash, label) in STYLE.items():
            for dim, alpha in ((500, 0.45), (1_000, 0.7), (3_600, 1.0)):
                ps = sorted({k[1] for k in t if k[0] == dim and k[2] == cue})
                x = [p / dim for p in ps]
                y = [np.mean(t[(dim, p, cue, rule)]) for p in ps]
                ax.plot(x, y, dash, color=color, alpha=alpha, lw=1.6,
                        label=label if dim == 3_600 else None)
        ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("items stored / cue dimension  (P / Ns)")
        ax.set_xlim(0, 3.05)
    axes[0].set_ylabel("right address recovered")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("The cliff is in the cue: recall collapses exactly at P = Ns, for every Ns "
                 "(lines: Ns = 500, 1,000, 3,600; 10 seeds)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "cliff.png", dpi=160)


PLACED = {"random": (MUTED, "-", "random (the paper's sequential is the same)"),
          "kmeans": ("#555555", "--", "k-means on the patterns"),
          "learned": (ACCENT, ":", "learned, noise law"),
          "error": (ACCENT, "-", "learned, error law"),
          "oracle": (INK, "-", "oracle (the true factors)")}


def _mean(rows: list[dict], **match) -> float:
    return float(np.mean([r[match["key"]] for r in rows if all(r[k] == v for k, v in match.items() if k != "key")]))


def placement() -> None:
    kinds = [(k, t) for k, t in (("factored", "factored content"), ("lifelog", "life-log sentences (MiniLM)"))
             if (RESULTS / f"placement-{k}.json").exists()]
    fig, axes = plt.subplots(1, len(kinds) + 1, figsize=(4.4 * (len(kinds) + 1), 3.4))
    for ax, (kind, title) in zip(axes, kinds, strict=False):
        rows = json.loads((RESULTS / f"placement-{kind}.json").read_text())
        rows = [r for r in rows if r["flip"] == 0.1]
        loads = sorted({r["load"] for r in rows})
        for name, (color, dash, label) in PLACED.items():
            y = [_mean(rows, key="ridge", load=x, scaffold="grid", placement=name) for x in loads]
            ax.plot(loads, y, dash, color=color, lw=1.8, label=label)
        flat = [_mean(rows, key="ridge", load=x, scaffold="flat", placement="random") for x in loads]
        ax.plot(loads, flat, "-", color=INK, lw=0.8, alpha=0.35, label="no-product control, any placement")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("items stored / cue dimension  (P / Ns)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("right address recovered (ridge write)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    # The law: every module of every placement and seed, at the headline condition.
    ax = axes[-1]
    rows = [r for r in json.loads((RESULTS / "placement-factored.json").read_text())
            if r["load"] == 0.8 and r["flip"] == 0.1 and r["scaffold"] == "grid"]
    for name, (color, _, _) in PLACED.items():
        picked = [r for r in rows if r["placement"] == name]
        x = [1e3 * r["law_error"][m] for r in picked for m in range(3)]
        y = [r["ridge_modules"][m] for r in picked for m in range(3)]
        ax.scatter(x, y, s=9, color=color, alpha=0.7, lw=0)
    ax.set_xlabel("error law on a module's phases  (1e-3)")
    ax.set_ylabel("module read correctly")
    ax.set_title("the law, factored content (P/Ns 0.8)", fontsize=10)
    fig.suptitle("Where a memory is put decides what survives (10% of cue bits flipped; 20 seeds)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "placement.png", dpi=160)


if __name__ == "__main__":
    cliff()
    placement()
    print(f"figures -> {RESULTS}")
