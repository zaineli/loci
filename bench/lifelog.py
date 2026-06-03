"""E2's second kind of content: a life-log of sentences, embedded and sign-projected.

Each entry is who (9 people) did what (5 activities) on which project (16), plus details of its own
(where, when, with what). The factor values are the same ground truth as `loci.content.factored`,
but the patterns come from a sentence encoder (all-MiniLM-L6-v2, 384 dimensions) through a random
sign projection to Ns bits, so nothing about them is Gaussian or independent by construction. The
facet cue for "which project?" is the same sentence with the project left out.

Two things measured and left in (changing them would change the registered run): a few entries
are exact duplicates (20 in 20 seeds at P = 800), which costs about 1/P on every arm alike; and
`decode` includes the item itself in its value's mean, which lifts decoding a facet cue *directly*
above chance but moves Vector-HaSH read-outs by <= 0.01.

Needs the `bench` extra; the model is read from the local Hugging Face cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

PEOPLE = ("Maya", "Omar", "Lena", "Tariq", "Sofia", "Kenji", "Amara", "Diego", "Priya")
PROJECTS = ("Heron", "Atlas", "Juniper", "Beacon", "Quartz", "Willow", "Falcon", "Meridian",
            "Cobalt", "Saffron", "Tundra", "Orchid", "Vesper", "Lantern", "Harbor", "Nimbus")
ACTIVITIES = ("drafted the budget for {}", "fixed a failing test in {}", "presented {} to the board",
              "planned the next release of {}", "interviewed two users about {}")
PLACES = ("in the library", "at the corner cafe", "from home", "in the big meeting room", "on the train",
          "in the lab", "at the airport", "in the garden office", "at a client's office", "in the hallway")
TIMES = ("on Monday morning", "late on Tuesday", "on Wednesday after lunch", "on Thursday evening",
         "early on Friday", "over the weekend", "just before the holidays", "on a rainy afternoon")
WITH = ("with a borrowed laptop", "with the old whiteboard", "with a cup of cold coffee",
        "with the new intern", "with noise-cancelling headphones", "with a printed checklist",
        "with a sketchbook", "with the team on a call")
CARDS = (len(PEOPLE), len(PROJECTS), len(ACTIVITIES))


def sentence(person: int, project: int, activity: int, detail: tuple[int, int, int], named: bool = True) -> str:
    what = f"the {PROJECTS[project]} project" if named else "the project"
    place, time, item = detail
    return f"{PEOPLE[person]} {ACTIVITIES[activity].format(what)} {PLACES[place]} {TIMES[time]}, {WITH[item]}."


@cache
def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    """Unit-norm MiniLM embeddings, (384, n)."""
    return _encoder().encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).T


@dataclass(frozen=True)
class Lifelog:
    patterns: np.ndarray  # (Ns, P) +/-1
    factors: np.ndarray  # (P, 3) person, project, activity - in the order of `content.CARDS`' roles
    texts: tuple[str, ...]
    projection: np.ndarray  # (Ns, 384)
    unnamed: np.ndarray  # (Ns, P) each entry's cue with the project left out

    def cue(self, without: int) -> np.ndarray:
        if without != 1:
            raise ValueError("the life-log only leaves out the project (factor 1)")
        return self.unnamed

    def decode(self, recalled: np.ndarray, factor: int) -> np.ndarray:
        """Which value of `factor` each recalled pattern is closest to, by the stored entries' mean
        pattern for each value - the life-log has no generative codes to decode against."""
        values = self.factors[:, factor]
        means = np.stack([self.patterns[:, values == v].mean(axis=1) for v in range(CARDS[factor])], axis=1)
        return (means.T @ recalled).argmax(axis=0)


def lifelog(dim: int, count: int, rng: np.random.Generator) -> Lifelog:
    """`count` entries in `dim` bits. Factor order matches `loci.content`: A = person (9),
    B = project (16), C = activity (5), so the same placements and oracle apply."""
    factors = np.stack([rng.integers(0, card, count) for card in CARDS], axis=1)
    details = np.stack([rng.integers(0, len(x), count) for x in (PLACES, TIMES, WITH)], axis=1)
    texts = [sentence(a, b, c, tuple(d)) for (a, b, c), d in zip(factors, details, strict=True)]
    cues = [sentence(a, b, c, tuple(d), named=False) for (a, b, c), d in zip(factors, details, strict=True)]
    projection = rng.standard_normal((dim, 384))
    patterns = np.sign(projection @ embed(texts))
    unnamed = np.sign(projection @ embed(cues))
    return Lifelog(patterns, factors, tuple(texts), projection, unnamed)
