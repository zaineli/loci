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


GREY = "#555555"
ARMS = {  # recall-by-arm marks: identity carried by marker shape as well as ink, never shade alone
    "random": (MUTED, "o", "none", "random (the paper's order is the same)"),
    "k-means": (GREY, "s", GREY, "k-means on the patterns"),
    "encode": (GREY, "^", "none", "stored where its own recall points"),
    "encode+replay": (ACCENT, "o", ACCENT, "... then replayed: the memory files itself"),
    "oracle": (INK, "D", INK, "oracle: the true factors as the address"),
}


def theory() -> None:
    """Predicted against measured recall: the out-of-sample test, and E4's rows."""
    oos = json.loads((RESULTS / "oos" / "results.json").read_text())["rows"]
    e4 = json.loads((RESULTS / "consolidate.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax = axes[0]
    measured = [r["measured"] for r in oos]
    ax.scatter([r["E"]["address"] for r in oos], measured, s=16, facecolors="none", edgecolors=MUTED, lw=0.8,
               label="the error law alone (phase indicators)")
    ax.scatter([r["T-relu"]["address"] for r in oos], measured, s=12, color=INK, lw=0,
               label="the theory (through the scaffold's templates)")
    ax.set_title("out of sample: 156 conditions, predictions hashed first", fontsize=9.5)
    errors = np.abs([r["T-relu"]["address"] - r["measured"] for r in oos])
    ax.text(0.03, 0.97, f"theory: mean |error| {errors.mean():.3f}\n100% within 0.05, r = 0.999",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    ax = axes[1]
    ax.scatter([r["theory"] for r in e4], [r["recall"] for r in e4], s=5, color=INK, lw=0, alpha=0.5)
    errors = np.abs([r["theory"] - r["recall"] for r in e4])
    ax.set_title("E4: 1,240 rows, 5 kinds of content (720 filed by the rule)", fontsize=9.5)
    ax.text(0.03, 0.97, f"mean |error| {errors.mean():.3f}\n{np.mean(errors <= 0.05):.1%} within 0.05",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":", zorder=0)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("predicted, before any cue is drawn")
        ax.set_aspect("equal")
    axes[0].set_ylabel("measured: noisy cues that find their own address")
    fig.tight_layout()
    fig.savefig(RESULTS / "theory.png", dpi=160)


def consolidation() -> None:
    """E4: recall by arm, per kind of content (P/Ns 0.8, 10% flips, seeds 40-49)."""
    rows = [r for r in json.loads((RESULTS / "consolidate.json").read_text()) if r["load"] == 0.8 and r["flip"] == 0.1]
    kinds = [("factored", "factored: 3 factors, 9 / 16 / 5 values"), ("cards", "3 factors, 7 / 12 / 4"),
             ("four", "4 factors on 3 modules"), ("hierarchy", "a hierarchy: no product structure"),
             ("lifelog", "life-log sentences (MiniLM)")]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for y, (kind, label) in enumerate(kinds):
        ax.axhline(y, color="#e6e6e6", lw=0.6, zorder=0)
        for name, (color, marker, face, legend) in ARMS.items():
            values = [r["recall"] for r in rows if r["content"] == kind and r["placement"] == name]
            if values:
                ax.scatter(np.mean(values), y, s=46, marker=marker, facecolors=face, edgecolors=color, lw=1.3,
                           label=legend if y == 0 else None, zorder=3)
    ax.set_yticks(range(len(kinds)), [label for _, label in kinds], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 0.95)
    ax.set_xlabel("noisy cues (10% of bits flipped) that find their own address")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "consolidate.png", dpi=160)


def imagination() -> None:
    """E5: the dissociation, the one event, and the read-out peak (seeds 40-49)."""
    rows = json.loads((RESULTS / "imagine.json").read_text())
    peak = json.loads((RESULTS / "imagine-nh.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    def mean(key: str, **match) -> float:
        return float(np.mean([r[key] for r in rows if all(r[k] == v for k, v in match.items())]))

    ax = axes[0]
    names = {"random": "random", "oracle": "oracle-filed", "encode+replay": "filed by the recall rule",
             "encode+anneal": "... with annealed replay", "oracle+replay": "oracle-filed, then replayed"}
    offsets = {"random": (6, 4), "oracle": (-62, -12), "encode+replay": (6, 6), "encode+anneal": (6, -10),
               "oracle+replay": (-120, -2)}
    for name, label in names.items():
        for kind, marker in (("factored", "o"), ("cards", "s")):
            x = mean("recall", content=kind, placement=name, decoder="nearest", flip=0.1)
            y = mean("construction", content=kind, placement=name, decoder="nearest", flip=0.1)
            color = ACCENT if name.startswith("oracle") else INK if name.startswith("encode") else MUTED
            ax.scatter(x, y, s=40, marker=marker, color=color, zorder=3)
            if kind == "factored":
                ax.annotate(label, (x, y), textcoords="offset points", xytext=offsets[name], fontsize=7.8,
                            color=INK)
    ax.set_xlim(0.94, 1.004)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("recall of studied events (10% flips)")
    ax.set_ylabel("never-experienced combinations constructed")
    ax.set_title("same recall, different imagination (nearest decoder)\ncircles factored, squares cards",
                 fontsize=9.5)

    ax = axes[1]
    cells = collections.defaultdict(list)
    for r in rows:
        if r["placement"] != "knn":
            cells[(r["content"], r["placement"], r["decoder"], r["flip"])].append((r["construction"], r["false recall"]))
    points = np.array([np.mean(v, axis=0) for v in cells.values()])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":", zorder=0)
    ax.scatter(points[:, 0], points[:, 1], s=14, color=INK, lw=0)
    ax.set_xlabel("construction (a gist cue lands on an empty state)")
    ax.set_ylabel("recombined events completed to it")
    ax.set_title(f"construction and completion of recombined events\n(90 cells, r = {np.corrcoef(points.T)[0, 1]:.3f})",
                 fontsize=9.5)
    ax.set_aspect("equal")

    ax = axes[2]
    cells_nh = sorted({r["Nh"] for r in peak})
    for ridge, color, dash, label in ((0.0, ACCENT, "-", "pseudo-inverse (paper)"),
                                      (0.01, INK, "--", "1% ridge")):
        y = [np.mean([r["construction"] for r in peak if r["Nh"] == n and r["readout_ridge"] == ridge
                      and r["decoder"] == "snap"]) for n in cells_nh]
        ax.plot(cells_nh, y, dash, color=color, lw=1.8, marker="o", ms=4, label=label)
    ax.axvline(800, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("place cells, Nh (800 items stored)")
    ax.set_ylabel("construction")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("a second interpolation peak, at P = Nh", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              title="read-out", title_fontsize=7.8)
    ax.text(925, 0.52, "pseudo-inverse", fontsize=7.8, color=INK)
    ax.text(1000, 0.92, "1% ridge", fontsize=7.8, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "imagine.png", dpi=160)


if __name__ == "__main__":
    cliff()
    placement()
    theory()
    consolidation()
    imagination()
    print(f"figures -> {RESULTS}")
