"""E1, dense cues: a Vector-HaSH cued by an embedding, not by a bit-flipped copy of its store.

  gaussian  Random unit vectors in d dimensions, sign-projected to Ns bits by a fixed Gaussian
            matrix and stored at addresses 0..P-1 (the paper's placement). The cue is a paraphrase
            at cosine 0.9 - the vector mixed with a random unit vector orthogonal to it - projected
            the same way. A sign projection turns cosine c into a per-bit disagreement of
            arccos(c) / pi (14.4% at 0.9), which sets the ridge (`memory.mmse_alpha`); but the
            disagreements are not independent flips - they are the image of a d-dimensional
            perturbation. Recorded: address recovery against P, and P50, the P at which it halves,
            against d and Ns. A Hamming matched filter on the same bits is the reference.
  locomo    LoCoMo, one conversation at a time: every turn embedded (MiniLM, local), sign-projected
            to 4,096 bits and stored in dialogue order; each question's projection is the cue; hit
            = an evidence turn. Rows: VH pinv, VH ridge, VH ridge + decode to the best stored
            address, and kNN on the floats and on 1-bit codes, each with the bits of state it keeps.

What it is not: an agent-memory benchmark, or a claim that a scaffold memory should be used for
retrieval - at equal bits kNN wins, and the table says by how much. Placement is sequential here;
where to put things is E2.

    HF_HUB_OFFLINE=1 uv run --extra bench python bench/dense.py [--seeds 10] [--only locomo]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from loci.memory import CueMaps, mmse_alpha
from loci.scaffold import Scaffold, relu

OUT = Path(__file__).resolve().parents[1] / "results" / "dense.json"
DIMS = (64, 128, 384, 1_024)
WIDTHS = (2_048, 4_096, 8_192)
COSINE = 0.9
FLIP = float(np.arccos(COSINE) / np.pi)  # the per-bit disagreement a sign projection turns 0.9 into
GRID = (50, 100, 200, 300, 400, 500, 600, 800, 1_000, 1_200, 1_400, 1_600, 1_800, 2_000, 2_400,
        2_800, 3_200, 3_600)  # P <= 3,600: the scaffold's addresses
PROBE = 1_000  # items cued per point: all of them up to this many, a fixed random subset beyond

# LoCoMo (snap-research/locomo, `locomo10.json`): not shipped; point LOCOMO at a local copy.
LOCOMO = Path(os.environ.get("LOCOMO", Path(__file__).resolve().parent / "data" / "locomo10.json"))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCOMO_WIDTH = 4_096
LOCOMO_SEEDS = 3  # projection (and scaffold) seeds


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=0, keepdims=True)


def paraphrase(items: np.ndarray, cosine: float, rng: np.random.Generator) -> np.ndarray:
    """Each column mixed with a random unit vector orthogonal to it: cosine exactly `cosine`."""
    noise = rng.standard_normal(items.shape)
    noise = unit(noise - items * (items * noise).sum(axis=0))
    return cosine * items + np.sqrt(1 - cosine**2) * noise


def weights(stored: np.ndarray, cues: np.ndarray, alpha: float, gram=None, cross=None):
    """(S^T S + alpha I)^-1 S^T C - each cue's ridge weights on the stored items (the pseudo-inverse
    at alpha = 0), so that W_hs C = H_a times this. Through the P x P Gram when P <= Ns (one
    Cholesky, not the SVD `CueMaps` pays for a sweep); through S S^T when P > Ns, where the Gram
    is singular and the same map is S^T (S S^T + alpha I)^-1, the pseudo-inverse's min-norm
    solution."""
    dim, count = stored.shape
    if count <= dim:
        gram = stored.T @ stored if gram is None else gram
        cross = stored.T @ cues if cross is None else cross
        return cho_solve(cho_factor(gram + alpha * np.eye(count)), cross)
    return stored.T @ cho_solve(cho_factor(stored @ stored.T + alpha * np.eye(dim)), cues)


# ---- gaussian ----------------------------------------------------------------------------------


def gaussian(seed: int) -> list[dict]:
    scaffold = Scaffold(seed=seed)
    rows = []
    for dim in DIMS:
        # The same items and paraphrases at every Ns, so that the Ns comparison is paired.
        rng = np.random.default_rng([11, seed, dim])
        items = unit(rng.standard_normal((dim, scaffold.addresses)))
        cues = paraphrase(items, COSINE, rng)
        for width in WIDTHS:
            started = time.time()
            rows += _gaussian_condition(scaffold, items, cues, width, [12, seed, dim, width])
            print(f"gaussian seed {seed} d {dim} Ns {width} ({time.time() - started:.1f}s)",
                  flush=True)
    return [{"seed": seed, **r} for r in rows]


def _gaussian_condition(scaffold, items, cues, width, key) -> list[dict]:
    rng = np.random.default_rng(key)
    proj = rng.standard_normal((width, items.shape[0]))
    stored_all, cue_all = np.sign(proj @ items), np.sign(proj @ cues)
    gram = stored_all.T @ stored_all  # nested stored sets: every P's Gram is a corner of this
    disagree = float((stored_all != cue_all).mean())
    rows = []
    for count in GRID:
        probe = np.arange(count) if count <= PROBE else rng.choice(count, PROBE, replace=False)
        stored, cue, place = stored_all[:, :count], cue_all[:, probe], scaffold.H[:, :count]
        cross = stored.T @ cue
        base = {"d": items.shape[0], "Ns": width, "P": count, "bit_disagreement": disagree}
        rows.append({**base, "rule": "hamming",
                     "recovery": float((cross.argmax(axis=0) == probe).mean())})
        for rule, alpha in (("pinv", 0.0), ("mmse", mmse_alpha(count, FLIP))):
            w = weights(stored, cue, alpha, gram[:count, :count], cross)
            phases, _ = scaffold.settle(relu(place @ w))
            rows.append({**base, "rule": rule,
                         "recovery": float((scaffold.index(phases) == probe).mean())})
    return rows


def p50(counts, recovery) -> float | None:
    """The P at which recovery first falls through 0.5, interpolated in log P; None if it never
    does inside the grid (censored at 3,600)."""
    counts, recovery = np.asarray(counts, float), np.asarray(recovery, float)
    below = np.flatnonzero(recovery < 0.5)
    if not len(below):
        return None
    i = below[0]
    if i == 0:
        return float(counts[0])
    lo, hi = np.log(counts[i - 1]), np.log(counts[i])
    t = (recovery[i - 1] - 0.5) / (recovery[i - 1] - recovery[i])
    return float(np.exp(lo + t * (hi - lo)))


def summarise_gaussian(rows: list[dict]) -> dict:
    seeds = sorted({r["seed"] for r in rows})
    table = {}
    for dim in DIMS:
        for width in WIDTHS:
            for rule in ("pinv", "mmse", "hamming"):
                per_seed = np.array([[next(r["recovery"] for r in rows if r["seed"] == s
                                           and r["d"] == dim and r["Ns"] == width
                                           and r["rule"] == rule and r["P"] == c)
                                      for c in GRID] for s in seeds])
                seed_p50 = [p50(GRID, curve) for curve in per_seed]
                known = [v for v in seed_p50 if v is not None]
                curve = per_seed.mean(axis=0)
                table[f"{dim}/{width}/{rule}"] = {
                    "d": dim, "Ns": width, "rule": rule,
                    "recovery": curve.round(4).tolist(),
                    "recovery_sd": per_seed.std(axis=0).round(4).tolist(),
                    "P50": p50(GRID, curve),
                    "P50_seeds_mean": float(np.mean(known)) if known else None,
                    "P50_seeds_sd": float(np.std(known)) if known else None,
                    "censored_seeds": len(seed_p50) - len(known)}
                p = table[f"{dim}/{width}/{rule}"]["P50"]
                table[f"{dim}/{width}/{rule}"]["P50_over_d"] = p / dim if p else None
    verdict = {}
    for rule in ("pinv", "mmse"):
        per_d = [table[f"{d}/4096/{rule}"]["P50_over_d"] for d in DIMS]
        spread_d = max(per_d) / min(per_d) if all(per_d) else None
        spread_ns = {d: _spread([table[f"{d}/{w}/{rule}"]["P50"] for w in WIDTHS]) for d in DIMS}
        verdict[rule] = {
            "P50_over_d_at_4096": dict(zip(DIMS, per_d, strict=True)),
            "spread_over_d": spread_d,
            "P50_by_Ns": {d: {w: table[f"{d}/{w}/{rule}"]["P50"] for w in WIDTHS} for d in DIMS},
            "spread_over_Ns": spread_ns,
            # "flat in Ns" is read with the same factor as the d criterion: max / min <= 1.5.
            "pass_d": spread_d is not None and spread_d <= 1.5,
            "pass_Ns": all(v is not None and v <= 1.5 for v in spread_ns.values())}
        verdict[rule]["pass"] = verdict[rule]["pass_d"] and verdict[rule]["pass_Ns"]
    return {"grid": list(GRID), "flip_equivalent": FLIP, "table": table, "criteria": verdict,
            "pass": any(v["pass"] for v in verdict.values())}


def _spread(values) -> float | None:
    return max(values) / min(values) if all(v is not None for v in values) else None


# ---- locomo ------------------------------------------------------------------------------------


def evidence_ids(raw: list[str]) -> list[str]:
    """Evidence ids as written, normalised: a few entries hold several ids ('D8:6; D9:17') or
    typos ('D:11:26', 'D30:05'), and one is just 'D'."""
    return [f"D{m[1]}:{m[2]}" for e in raw for m in re.finditer(r"D:?(\d+):0*(\d+)", e)]


def conversations() -> list[dict]:
    """Per conversation: turn texts ('speaker: text'), and every question except category 5
    (adversarial: the answer is not in the dialogue) that has an evidence turn, with those turns'
    indices."""
    out = []
    for conv in json.loads(LOCOMO.read_text()):
        turns = [t for k, v in conv["conversation"].items()
                 if k.startswith("session_") and isinstance(v, list) for t in v]
        index = {t["dia_id"]: i for i, t in enumerate(turns)}
        questions, evidence, dropped = [], [], 0
        for qa in conv["qa"]:
            if qa.get("category") == 5:
                continue
            ev = sorted({index[e] for e in evidence_ids(qa.get("evidence", [])) if e in index})
            if not ev:
                dropped += 1
                continue
            questions.append(qa["question"])
            evidence.append(ev)
        out.append({"id": conv["sample_id"],
                    "turns": [f"{t['speaker']}: {t['text']}" for t in turns],
                    "questions": questions, "evidence": evidence, "dropped": dropped})
    return out


def embed(convs: list[dict]) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cpu")
    for conv in convs:
        for key in ("turns", "questions"):
            conv[key + "_emb"] = model.encode(conv[key], batch_size=128, convert_to_numpy=True,
                                              normalize_embeddings=True).astype(np.float32)


def recall_at(scores: np.ndarray, evidence: list[list[int]], k: int) -> float:
    """scores (P, nq): hit if any evidence turn is in a question's top k."""
    top = np.argsort(-scores, axis=0, kind="stable")[:k]
    return float(np.mean([bool(set(top[:, q]) & set(ev)) for q, ev in enumerate(evidence)]))


def locomo_conversation(conv: dict, scaffold: Scaffold, proj: np.ndarray) -> dict:
    T, Q, ev = conv["turns_emb"], conv["questions_emb"], conv["evidence"]
    count = len(T)
    stored, cues = np.sign(proj @ T.T).astype(float), np.sign(proj @ Q.T).astype(float)
    place = scaffold.H[:, :count]
    # alpha from the median question-evidence cosine read as a per-bit flip rate. It uses the
    # evidence labels, for one scalar per conversation - a mild oracle, in VH's favour.
    cos = np.median([float(T[e] @ Q[q]) for q, evs in enumerate(ev) for e in evs])
    flip_rate = float(np.arccos(np.clip(cos, -1, 1)) / np.pi)
    alpha = mmse_alpha(count, flip_rate)
    # A true pseudo-inverse here, not the Cholesky: a turn repeated word for word (conv-47 and
    # conv-48 have one each) stores an identical code, so S^T S is singular. At P <= 689 the SVD
    # is cheap.
    maps = CueMaps(stored)
    rows = {}
    for rule, a in (("VH pinv", 0.0), ("VH ridge", alpha)):
        h = relu(maps.map(place, a) @ cues)
        phases, _ = scaffold.settle(h)
        got = scaffold.index(phases)
        # Stored addresses ranked by the pre-snap place input against each one's place code:
        # the settled code has already committed to one address (possibly an empty one) and its dot
        # products with stored codes take only as many values as there are shared-phase counts.
        ranked = place.T @ h
        rows[rule] = {"R@1": float(np.mean([got[q] in e for q, e in enumerate(ev)])),
                      "R@5": recall_at(ranked, ev, 5)}
        if rule == "VH ridge":
            rows["VH ridge + stored decode"] = {"R@1": recall_at(ranked, ev, 1),
                                                "R@5": rows[rule]["R@5"]}
    # Not a VH, a diagnostic: the ridge's weights on the stored items themselves, before the
    # Nh = 400 place layer (rank 400 < P) superposes them. Where the table's loss happens.
    direct = maps.map(np.eye(count), alpha) @ cues
    rows["diagnostic: ridge weights, no place layer"] = {"R@1": recall_at(direct, ev, 1),
                                                         "R@5": recall_at(direct, ev, 5)}
    for rule, scores in (("kNN cosine, float32", T @ Q.T),
                         ("kNN Hamming, 4096 bits", stored.T @ cues),
                         ("kNN Hamming, 384 bits", np.sign(T) @ np.sign(Q).T)):
        rows[rule] = {"R@1": recall_at(scores, ev, 1), "R@5": recall_at(scores, ev, 5)}
    return {"rows": rows, "P": count, "questions": len(ev), "median_cosine": float(cos),
            "flip_equivalent": flip_rate, "alpha_over_P": alpha / count}


def bits(rule: str, count: int, dim: int) -> int:
    """State kept, in bits. VH: W_hs and W_sh, 2 x Ns x Nh float32, whatever P is (the scaffold's
    own fixed random maps are not counted). kNN: the stored vectors."""
    if rule.startswith("VH"):
        return 2 * LOCOMO_WIDTH * 400 * 32
    if rule.startswith("diagnostic"):
        return count * LOCOMO_WIDTH * 32  # the P x Ns map from a cue to item weights
    if "float32" in rule:
        return count * dim * 32
    return count * (4_096 if "4096" in rule else 384)


def locomo(seeds: int) -> dict:
    started = time.time()
    convs = conversations()
    embed(convs)
    print(f"locomo: embedded {sum(len(c['turns']) for c in convs)} turns, "
          f"{sum(len(c['questions']) for c in convs)} questions ({time.time() - started:.0f}s)",
          flush=True)
    dim = convs[0]["turns_emb"].shape[1]
    per = []
    for seed in range(seeds):
        scaffold = Scaffold(seed=seed)
        proj = np.random.default_rng([13, seed]).standard_normal((LOCOMO_WIDTH, dim))
        for conv in convs:
            per.append({"seed": seed, "conversation": conv["id"], "dropped": conv["dropped"],
                        **locomo_conversation(conv, scaffold, proj)})
        print(f"locomo seed {seed} ({time.time() - started:.0f}s)", flush=True)
    return {"per_conversation": per, "summary": summarise_locomo(per, dim)}


def summarise_locomo(per: list[dict], dim: int) -> dict:
    rules = list(per[0]["rows"])
    seeds = sorted({p["seed"] for p in per})
    first = [p for p in per if p["seed"] == seeds[0]]
    table = {}
    for rule in rules:
        entry = {}
        for k in ("R@1", "R@5"):
            macro = [np.mean([p["rows"][rule][k] for p in per if p["seed"] == s]) for s in seeds]
            pooled = [sum(p["rows"][rule][k] * p["questions"] for p in per if p["seed"] == s)
                      / sum(p["questions"] for p in per if p["seed"] == s) for s in seeds]
            entry[k] = float(np.mean(macro))
            entry[k + "_sd_over_seeds"] = float(np.std(macro))
            entry[k + "_pooled"] = float(np.mean(pooled))
        entry["bits_mean_per_conversation"] = float(np.mean([bits(rule, p["P"], dim)
                                                             for p in first]))
        table[rule] = entry
    return {"table": table, "conversations": len(first),
            "questions": sum(p["questions"] for p in first),
            "dropped_questions": sum(p["dropped"] for p in first),
            "median_cosine": [p["median_cosine"] for p in first],
            "flip_equivalent": [p["flip_equivalent"] for p in first]}


def figure(summary: dict) -> None:
    """Recovery against P at Ns = 4,096, one line per d; and P50 against d, one line per Ns."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures import ACCENT, INK, MUTED

    table = summary["table"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, dim in enumerate(DIMS):
        shade = 0.3 + 0.7 * k / (len(DIMS) - 1)
        for rule, color, dash in (("pinv", ACCENT, "-"), ("mmse", INK, "--")):
            axes[0].plot(GRID, table[f"{dim}/4096/{rule}"]["recovery"], dash, color=color,
                         alpha=shade, lw=1.5, label=f"{rule}, d = {dim}")
    axes[0].axhline(0.5, color=MUTED, lw=0.6, ls=":")
    axes[0].set(xscale="log", xlabel="items stored  (Ns = 4,096)",
                ylabel="right address recovered", title="Cue at cosine 0.9, sign-projected")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    for k, width in enumerate(WIDTHS):
        shade = 0.35 + 0.65 * k / (len(WIDTHS) - 1)
        for rule, color, dash in (("pinv", ACCENT, "-"), ("mmse", INK, "--")):
            p = [table[f"{d}/{width}/{rule}"]["P50"] for d in DIMS]
            axes[1].plot(DIMS, [np.nan if v is None else v for v in p], dash, marker="o", ms=4,
                         color=color, alpha=shade, lw=1.5, label=f"{rule}, Ns = {width:,}")
    axes[1].set(xscale="log", yscale="log", xlabel="embedding dimension d",
                ylabel="P50 (recovery halves)", title="What sets P50: d or Ns?")
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="gaussian seeds")
    parser.add_argument("--only", default="gaussian,locomo")
    args = parser.parse_args()
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.only.split(","):
        started = time.time()
        if name == "gaussian":
            rows = [row for seed in range(args.seeds) for row in gaussian(seed)]
            result[name] = {"seeds": args.seeds, "summary": summarise_gaussian(rows), "rows": rows}
        else:
            result[name] = {"seeds": LOCOMO_SEEDS, **locomo(LOCOMO_SEEDS)}
        result[name]["seconds"] = round(time.time() - started, 1)
        print(f"{name} done ({result[name]['seconds']}s)", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result))  # after each section, so one failing loses only itself
    if "gaussian" in result:
        figure(result["gaussian"]["summary"])
    if "locomo" in result:
        for rule, e in result["locomo"]["summary"]["table"].items():
            print(f"{rule:44s} R@1 {e['R@1']:.3f}  R@5 {e['R@5']:.3f}  "
                  f"bits {e['bits_mean_per_conversation']:,.0f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
