"""The out-of-sample condition grid, and how each condition's content, scaffold, placements and cues
are built. Shared by predict.py (which never runs a recall) and measure.py (which never runs the
theory). Everything is rebuilt deterministically from (family, setting, seed).

Development conditions (THEORY.md C4): factored (9,16,5), Ns 1000, Nh 400, periods (3,4,5), flips
0.1 / 0.2, ridge at the MMSE alpha, loads {0.2, 0.4, 0.6, 0.8, 0.9}, seeds 0-2. Every family below
changes at least one of those parameters, and every seed here (30-32) is fresh.
"""
import os
for _t in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_t, "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
sys.path.insert(0, "/Users/zain/work/systems/loci/src")
sys.path.insert(0, "/Users/zain/work/systems/loci/bench")
from dataclasses import dataclass
import numpy as np
from loci import place
from loci.content import CARDS, factored
from loci.memory import flip, mmse_alpha
from loci.scaffold import Scaffold

SEEDS = (30, 31, 32)
ALL = ("sequential", "random", "kmeans", "oracle", "error")


@dataclass(frozen=True)
class Setting:
    family: str
    load: float = 0.8
    content: str = "factored"
    cards: tuple = CARDS
    Ns: int = 1000
    Nh: int = 400
    periods: tuple = (3, 4, 5)
    cue: str = "flip"          # "flip" or "mask"
    rate: float = 0.1
    read: str = "ridge"        # "ridge" (MMSE alpha) or "pinv" (alpha = 0)
    placements: tuple = ALL

    @property
    def count(self) -> int:
        return round(self.load * self.Ns)


SETTINGS = [
    *[Setting("load", load=x, placements=("random", "kmeans", "oracle", "error")) for x in (0.3, 0.5, 0.7)],
    Setting("lifelog", content="lifelog"),
    Setting("cards 7/12/4", cards=(7, 12, 4)),
    *[Setting(f"Nh {n}", Nh=n, placements=("random", "oracle", "error")) for n in (200, 800)],
    Setting("periods 5,6,7", periods=(5, 6, 7), placements=("random", "kmeans", "oracle", "error")),
    Setting("periods 3,4,5,7", periods=(3, 4, 5, 7), placements=("sequential", "random")),
    Setting("Ns 500", Ns=500, placements=("random", "oracle", "error")),
    Setting("Ns 2000", Ns=2000, placements=("random", "kmeans", "oracle")),
    *[Setting(f"mask {r}", cue="mask", rate=r, placements=("random", "kmeans", "oracle", "error")) for r in (0.25, 0.5)],
    Setting("pinv read", read="pinv", placements=("random", "kmeans", "oracle", "error")),
]

# Outside the theory's derivation as written, and how they are handled (all are still predicted):
NOTES = {
    "mask": "masked cues are not symmetric flips. The theory reads the cue only through a = E[s~]/s "
            "and sigma^2 = Var(s~); the predicted accuracy depends only on sigma^2/a^2, since ReLU is "
            "positively homogeneous and the argmax scale-free. A mask at rate r (a = 1 - r, sigma^2 = "
            "r(1 - r)) is passed as the flip rate f* with the same sigma^2/a^2 = r/(1 - r), and alpha "
            "is passed explicitly. The CLT over Ns independent bits still holds.",
    "lifelog": "the content is sign-projected MiniLM, not factor-Gaussian; the theory uses the content "
               "only through K, so no assumption is broken, but it was developed on factored content only.",
    "periods 3,4,5,7": "4 modules; the frozen code needed the logged literal patch to run.",
    "Nh 200": "at 200 place cells not every address is a snap fixed point; the theory sees this only "
              "through the templates.",
}


def factored_cards(dim, count, rng, cards, detail=1.0):
    """loci.content.factored with other cardinalities (identical construction)."""
    if tuple(cards) == CARDS:
        return factored(dim, count, rng, detail)
    codes = tuple(rng.standard_normal((dim, card)) for card in cards)
    factors = np.stack([rng.integers(0, card, count) for card in cards], axis=1)
    own = detail * rng.standard_normal((dim, count))
    signal = sum(c[:, factors[:, i]] for i, c in enumerate(codes))
    from loci.content import Factored
    return Factored(np.sign(signal + own), factors, codes, own)


def build(s: Setting, seed: int):
    """(patterns, factors, scaffold, alpha_read, alpha_place)."""
    rng = np.random.default_rng(10_000 * seed + s.count + s.Ns)
    if s.content == "lifelog":
        from lifelog import lifelog
        content = lifelog(s.Ns, s.count, rng)
    else:
        content = factored_cards(s.Ns, s.count, rng, s.cards)
    scaffold = Scaffold(periods=s.periods, place_cells=s.Nh, seed=seed)
    stats = {"flip_rate": s.rate} if s.cue == "flip" else {"mask_rate": s.rate}
    alpha_mmse = mmse_alpha(s.count, **stats)
    alpha_read = 0.0 if s.read == "pinv" else alpha_mmse
    alpha_place = alpha_mmse if s.read == "ridge" else mmse_alpha(s.count, flip_rate=0.1)
    return content.patterns, content.factors, scaffold, alpha_read, alpha_place


def placements(s: Setting, seed, patterns, factors, scaffold, alpha_place):
    count, out = patterns.shape[1], {}
    for name in s.placements:
        if name == "sequential":
            out[name] = place.sequential(count)
        elif name == "random":
            out[name] = place.scattered(count, scaffold, np.random.default_rng(seed + 1))
        elif name == "kmeans":
            out[name] = place.kmeans(patterns, scaffold, np.random.default_rng(seed + 2))[0]
        elif name == "oracle":
            out[name] = place.oracle(scaffold, factors)
        elif name == "error":
            _, labels = place.learned(patterns, scaffold, np.random.default_rng(seed + 3), alpha_place)
            out[name] = place.learned(patterns, scaffold, np.random.default_rng(seed + 4), alpha_place,
                                      error=True, warm=labels)[0]
    return out


def theory_rate(s: Setting) -> float:
    """The flip rate the theory is given: the flip rate itself, or for a mask the flip rate with the
    same sigma^2 / a^2 (see NOTES["mask"])."""
    if s.cue == "flip":
        return s.rate
    x = s.rate / (1 - s.rate)                     # sigma^2 / a^2 of the mask
    return 0.5 * (1 - 1 / np.sqrt(1 + x))         # flips: sigma^2/a^2 = 1/(1-2f)^2 - 1


def cues(s: Setting, patterns, seed, draw):
    rng = np.random.default_rng(1_000_003 * (draw + 1) + 20_000 * seed + s.count + int(1_000 * s.rate))
    if s.cue == "flip":
        return flip(patterns, s.rate, rng)
    return patterns * (rng.random(patterns.shape) >= s.rate)


def row_id(s: Setting, seed, name) -> str:
    return f"{s.family}|seed {seed}|{name}"
