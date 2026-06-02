"""Figures for the README, from the result files. Free; matplotlib only.

    uv run --extra bench python bench/figures.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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


if __name__ == "__main__":
    cliff()
    print(f"figures -> {RESULTS}")
