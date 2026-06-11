# loci

Where a memory is put decides what survives: placement in scaffold memories, and a quadratic
form that ranks placements.

> **Corrections, 2026-10-01.** An internal red-team pass found four errors in the first version.
> They are fixed below and listed under [Corrections](#corrections): a decoder bias that overstated
> how much the grid's codebook costs, a paper quote attributed to the wrong document and read too
> strongly, a headline ("one matrix predicts it") the data do not support, and an undisclosed warm
> start in the learned placement.

Vector-HaSH (Chandra, Sharma, Chaudhuri & Fiete, *Nature* 2025) builds memory the way the
hippocampus is thought to: a fixed **scaffold** of grid-cell addresses (three small modules, 3,600
error-correcting fixed points), with content bound to addresses. A cue recalls by snapping to an
address and reading out what is stored there. The paper puts each item at the next address along
a fixed path. It says that for non-spatial content "the scaffold trajectory can be arbitrarily
chosen". It measures its capacity curves with clean cues, and reports that the model "avoids the
memory cliff of prior memory models".

The paper tests noisy cues at 500 stored items and in two SI figures. loci asks how noisy-cue
recall scales as the store fills, and whether it matters *where* each item goes. It reimplements the model, checks it against the upstream code bit for bit,
and runs pre-registered experiments. Every result is reported against a control, including the 8 of
the 17 pre-stated checks that failed.

```python
import numpy as np
from loci import place
from loci.content import factored
from loci.memory import Memory, flip, mmse_alpha
from loci.scaffold import Scaffold

scaffold = Scaffold()                                   # modules of period 3, 4, 5: 3,600 addresses
items = factored(1_000, 800, np.random.default_rng(0))  # 800 items, 1,000 bits, 3 hidden factors
cues = flip(items.patterns, 0.1, np.random.default_rng(1))  # every cue with 10% of its bits flipped
alpha = mmse_alpha(800, flip_rate=0.1)                  # the write matched to that noise

placements = {
    "random": place.scattered(800, scaffold, np.random.default_rng(2)),
    "learned": place.learned(items.patterns, scaffold, np.random.default_rng(3), alpha, error=True)[0],
}
for name, where in placements.items():
    memory = Memory(scaffold, rule="ridge", alpha=alpha)
    memory.store(items.patterns, where)
    print(name, (memory.recall(cues).address == where).mean())   # random 0.535, learned 0.846
```

## The result, stated plainly

**The cliff is in the cue.** With the paper's write rule, a cue with 5% of its bits flipped finds
its own address 0.1% of the time at P = Ns, where P is the number of items stored and Ns the cue
dimension. It is a dip, not a floor: recovery is 1.00 at P/Ns 0.2, 0.72 at 0.8 and 0.001 at 1.0,
and climbs back to 0.26–0.50 between P/Ns 1.1 and 1.5. That holds at every Ns tried, for flipped and masked
cues alike. Clean cues are unaffected. The paper sets Ns equal to the number of addresses (3,600),
the smallest value its clean-cue analysis allows, which puts P = Ns at the right-hand edge of every
capacity plot. A write matched to the noise keeps 85%.

**The grid's per-module snap is where noisy cues are lost.** Two addresses that share a module
phase have correlated place codes, so every item stored on a phase adds its share of a cue's noise
to that phase. With the same 400 place cells, random place codes with a nearest-code snap recover
**0.999**; the grid's snap recovers **0.455**. But decode the grid's own cue-map output to the
nearest grid code by *cosine* and it recovers 0.976 (10% flips). So at 10% flips almost all of the
loss is in the snap, which decides each module on its own; the correlated codebook costs about
0.03. At 20% flips the codebook's cost is real (0.58 against 0.99).

**Placement decides whether the pooled noise cancels.** The table below uses the same weights and
the same cues for every row; only the addresses change. Ridge write, P/Ns = 0.8, 10% flips,
20 seeds:

| placement | right address |
|---|---|
| the paper's (item j at address j) | 0.455 |
| k-means on the patterns | 0.764 |
| **learned** (no labels) | **0.850** |
| oracle (the true factors as the address) | 0.877 |

Read off the P/Ns curves, random placement (the paper's does the same) would need a **2.4×** larger
cue dimension to do as well as the oracle. That is under the ridge read; under the registered read it is 1.3×. The learned placement is
given no labels, only a group count per module, which we set to the factor cardinalities (the
life-log's counts match too). It minimises one quadratic form, and that puts each hidden factor on
its own grid module (normalised mutual information 0.92 / 0.93 / 0.93). Its search starts from 20
random balanced labellings *and* from the noise-law placement. From random starts alone it recalls
0.776 on factored content and 0.547 on sentences, below k-means (0.677); in every run the true
factors score lower on the law than anything the search found. The law is right; searching it is
hard.

**One matrix ranks the placements.** The noise a placement puts on each module phase is σ² qᵀΓq, where q
marks the items placed on that phase. Correlated items cancel each other's noise, so the best
placement groups items by *partial* correlation, not by raw similarity. The first form of this
law, which counts variance only, predicts the paper's write rule (ρ = −0.88 to −0.93). But on
sentence embeddings, its minimiser recalls *worse than random* (0.240 against 0.293). The missing
term is the ridge's bias. Bias plus variance collapses to **(SᵀS + αI)⁻¹**, the posterior covariance
of the cue's coefficients over the stored items. On seeds held back for the purpose, that
**error law** ranks per-module accuracy on the sentences (ρ ≤ −0.80), and the error-law
placement we found beats k-means on both kinds of content. Nearly all of that ρ is the difference
between placement types: a proxy that knows only which placement was used scores the same, and
within one placement the law predicts little. As an objective the error law is, up to group-size
scaling, discriminative k-means (Ye, Zhao & Wu 2007): Σ_k q_kᵀ(SᵀS + αI)⁻¹q_k =
(P − tr(Qᵀ G(G + αI)⁻¹ Q)) / α, with the ridge smoother as its kernel.

**None of this makes a better memory.** kNN over the stored patterns keeps 800,000 bits, against the
grid's 840,000 weights, and recovers 1.000. Given a cue that leaves out one factor, Vector-HaSH
fills it in 16–26% of the time; kNN does 100%. On LoCoMo conversation turns its R@1 is at most
0.037, against kNN's 0.188. This repository studies how scaffold memories work and fail. It is not
a proposal to use one.

## The cliff is in the cue

Right address recovered at P = Ns. Mean over Ns = 500, 1,000 and 3,600; 10 seeds; the paper's
placement ([`bench/cliff.py`](bench/cliff.py)).

| cue | pseudo-inverse (the paper) | **noise-matched ridge** | Hebbian | kNN, P × Ns bits |
|---|---|---|---|---|
| clean | 1.000 | 1.000 | 0.699 | 1.000 |
| 5% of bits flipped | **0.001** | **0.851** | 0.604 | 1.000 |
| 10% flipped | 0.001 | 0.668 | 0.501 | 1.000 |
| 20% flipped | 0.001 | 0.303 | 0.264 | 1.000 |
| 25% of bits unknown | 0.001 | 0.793 | 0.575 | 1.000 |
| 50% unknown | 0.000 | 0.479 | 0.386 | 1.000 |

![Recovery against P/Ns for three cue dimensions](results/cliff.png)

The pseudo-inverse's minimum sits at exactly P/Ns = 1.0 for every Ns and every kind of noise. Its
noise gain grows as P / (Ns − P − 1). Past that point the scaffold snaps to a *valid but wrong*
address, and nothing downstream compares the recall with the cue. The ridge strength that fixes it
is α = Var(ξ)·P / a² for a cue a·s + ξ; for flips, 4f(1 − f)P / (1 − 2f)². None of this is new as
mathematics: projection-rule memories lose their basins as P → N (Personnaz, Guyon & Dreyfus 1986;
Kanter & Sompolinsky 1987), the peak at P = N and its removal by weight decay are Krogh & Hertz
1992, and it is double descent (Belkin et al. 2019; Hastie et al. 2022; Nakkiran et al. 2021). What
is new is where it sits. The preprint says that from "a highly corrupted memory pattern" the
scaffold dynamics "recover the exact scaffold state … even deep in the memory continuum" (bioRxiv
v2, p. 9). The review response says noisy cues give "the correct grid and hippocampal states". The
SI shows the qualification: exact recovery from 10%-flipped cues falls with load (Fig S8, whose
caption attributes it to "overcrowding … within the sensory-to-hippocampal weights"), to 0.28 at
3,000 items. What it does not show is where the fall ends: at P = Ns. The paper's choice of
Ns = Npos = 3,600, the smallest its clean-cue analysis allows (SI D.1), puts that point at the
right-hand edge of every capacity plot. This is a double-descent dip in the cue map, not the
clean-cue memory cliff the paper's title claim is about.

Against the paper's own figures, with the published curves parsed from the SI's vector graphics
([`bench/paper_figs.py`](bench/paper_figs.py), 10 seeds):

| check | result |
|---|---|
| SI Fig S7, MI per input bit, 10% flips | reproduced; max deviation 0.023 |
| SI Fig S8, recovery vs flip rate | max deviation 0.0305 against a 0.03 bar: **fails by 0.0005**; 5 of 10 seeds pass alone |
| flips and masks on one axis, SNR = a² / Var(ξ) | collapse to within 0.016 |
| one ridge strength blind to the noise, α = cP | **fails**: c = 0.3 stays within 0.015 up to 10% flips, falls 0.074 short at 20% |

![The paper's S7 and S8, and S8 with the fix](results/paper_figs.png)

## Where a memory is put

The content has known structure. Each item is sign(A + B + C + z): three factors (think person,
project, activity) with 9, 16 and 5 values, plus the item's own detail. The second kind of content
is a life-log of templated sentences ("Maya fixed a failing test in the Heron project on the
train…"), embedded with all-MiniLM-L6-v2 and sign-projected to 1,000 bits
([`bench/lifelog.py`](bench/lifelog.py)). Ns = 1,000; 20 seeds; identical patterns and cues across
every row ([`bench/placement.py`](bench/placement.py)).

Right address recovered at P/Ns = 0.8, 10% flips:

| placement | pseudo-inverse | ridge | ridge + best stored address | no-product control |
|---|---|---|---|---|
| **factored content** | | | | |
| sequential (the paper's) | 0.138 | 0.455 | 0.849 | 0.999 |
| random | 0.137 | 0.455 | 0.833 | 0.999 |
| k-means, per module | 0.387 | 0.764 | 0.853 | 0.998 |
| learned, noise law | 0.492 | 0.830 | 0.900 | 0.998 |
| **learned, error law** | **0.553** | **0.850** | **0.908** | 0.998 |
| oracle | 0.610 | 0.877 | 0.915 | 0.999 |
| **life-log sentences** | | | | |
| sequential (the paper's) | 0.052 | 0.284 | 0.701 | 0.988 |
| random | 0.055 | 0.293 | 0.681 | 0.989 |
| k-means, per module | 0.197 | 0.668 | 0.755 | 0.989 |
| learned, noise law | 0.021 | **0.240** | 0.519 | 0.987 |
| **learned, error law** | **0.274** | **0.702** | **0.776** | 0.988 |
| oracle | 0.372 | 0.794 | 0.840 | 0.988 |
| kNN on the stored patterns | | | 1.000 / 0.999 | |

![Recovery against load for each placement, and the law](results/placement.png)

Near the cue limit, placement matters, and only there. At P/Ns = 0.2 the placements are within
0.003 of each other on the best-stored-address read (0.008 on sentences). With 20% of bits flipped
the gap widens. On factored content at P/Ns = 0.8, random recovers 0.120, the error-law placement
0.638 and the oracle 0.715. The noise-law placement collapses to 0.048: at high noise the bias it
ignores is most of the error.

**Why the error law.** The cue map writes a cue as coefficients c = (SᵀS + αI)⁻¹Sᵀs̃ over the stored
items, and a module phase receives the sum of the coefficients of the items placed on it. At the
noise-matched α, the squared error of those coefficients (bias plus variance) is
σ²(SᵀS + αI)⁻¹, the posterior covariance of Bayesian linear regression (Bishop 2006, §3.3).
[`tests/test_place.py`](tests/test_place.py) checks the identity by sampling flipped cues. At α = 0
it is the noise law. The optimal placement groups items whose coefficients are anti-correlated,
which means items that are partially correlated. The learned placement finds such groups by
pairwise swaps from 20 random balanced starts, per module, with no labels and no coupling between
modules. With k groups, the partition that cancels the most error is the one along a factor with
about k values. So which module takes which factor falls out of the law. On the sentences, the
error-law placement recovers the three factors at NMI 0.70 / 0.92 / 0.84; the noise-law placement
at 0.09 / 0.17 / 0.41.

**What survives a wrong recall.** When the address is wrong, the read-out is a blend of the items
that share its phases. With aligned placement those items share the item's factors. Factors read
off wrong recalls, factored content, seeds 0–9:

| placement | person (1 in 9) | project (1 in 16) | activity (1 in 5) |
|---|---|---|---|
| random | 0.18 | 0.14 | 0.26 |
| learned, noise law | 0.95 | 0.94 | 0.77 |
| oracle | 1.00 | 1.00 | 0.74 |

Gist survives. A missing facet does not. Give a cue with the project left out, and ask the memory
for it: Vector-HaSH answers 0.16–0.26 on factored content and 0.08–0.19 on sentences, against
kNN's 1.00 and 0.88. Aligned placement is *worse* at this (oracle 0.16, random 0.22). When the
address encodes the project, a cue without the project cannot find the address.

## Where the grid loses noisy cues

The no-product control is the same model with one module of 3,600 phases. That gives random sparse
place codes, the same 400 place cells, and a snap to the nearest of all 3,600 codes. It is
exchangeable by construction, so placement cannot matter there, and it doesn't (oracle − random
= 0.000). Through the scaffold's snap it beats the grid outright.

Two reads of the same cue-map output separate the codebook from the decoder. The grid's snap
decides each module on its own. The second read, which the grid does not have natively, takes the
nearest of all 3,600 place codes by cosine. Exploratory, not pre-registered
([`bench/budget.py`](bench/budget.py), random placement, P/Ns = 0.8, 10 seeds):

| code | weights | clean | snap, 10% / 20% flipped | nearest code (cosine), 10% / 20% |
|---|---|---|---|---|
| grid, 400 place cells (the paper's) | 840,000 | 1.000 | 0.467 / 0.129 | 0.963 / 0.591 |
| no product, 91 place cells | 837,200 | 0.908 | 0.498 / 0.245 | 0.970 / 0.743 |
| grid, 1,750 place cells | 3,675,000 | 1.000 | 0.731 / 0.262 | 0.983 / 0.725 |
| no product, 400 place cells | 3,680,000 | 1.000 | 0.998 / 0.940 | 1.000 / 1.000 |

At 10% flips the grid's codebook is nearly as good as random codes (0.963 against 0.970 at equal
weights). Almost everything the grid loses is lost in the module-by-module snap. At 20% flips the
correlated codebook costs as well: codes that share residues are confusable, and the items behind
them pool their noise. Placement is how the grid gets part of it back.

## What was registered, and what held

Criteria were written in [`docs/PREREG.md`](docs/PREREG.md) before the confirmatory runs; the
amendments are logged there, with their reasons, before the runs they apply to. E2's registered
criteria are judged on seeds 0–9; amendment 4's error law on seeds 10–19, which no pilot touched.
Bars are paired bootstrap 95% intervals over seeds. `bench/placement.py --verdict` recomputes the
E2 rows from the result files.

| | criterion | factored | life-log |
|---|---|---|---|
| E1.1 | the paper's S7 / S8 within 0.03 | S7 pass · S8 **fail** (0.0305) | |
| E1.2 | flips and masks collapse on SNR (≤ 0.05) | pass (0.016) | |
| E1.3 | one noise-blind ridge suffices | **fail** | |
| E1.4 | dense cues: recall halves at a P set by d, flat in Ns | **fail**: set by Ns | |
| P1 | module accuracy monotone in the law, ρ ≤ −0.8, pinv and ridge | **fail**: ridge's last module −0.64 | **fail**: ridge −0.48 to −0.66 |
| P2 | oracle ≥ random + 0.10 at P/Ns 0.8; equal at 0.2 | **fail**: +0.087 [0.071, 0.102] | pass: +0.156 [0.132, 0.181] |
| P3 | learned (noise law) ≥ k-means + 0.05 | pass: +0.055 [0.031, 0.076] | **fail**: −0.238 |
| P4 | aligned gain larger on the grid than the control by 0.05 | pass: +0.087 | pass: +0.158 |
| X1 | the error law predicts module accuracy, ρ ≤ −0.8 | **fail**: last module −0.72 | pass: −0.89 / −0.90 / −0.80 |
| X2 | error law ≥ noise law (by 0.05 on sentences), and > k-means | pass: +0.011; +0.049 [0.035, 0.065] | pass: +0.292; +0.059 [0.010, 0.099] |

P2–P4 are scored on the registered read, ridge + best stored address. It is the read on which
placement matters least: on the paper's pseudo-inverse the oracle's gain over random is five times
larger (+0.47). The last module is where each law is weakest, because its phase is a free slot
chosen by open addressing, not by content. **P4 could not have failed through its control**: the
control sits at ceiling, so P4 reduces to P2 with a lower bar. It says nothing about whether any
structured code would do as well as the grid.

## Corrections

Found by an internal red team after the first version was published (2026-09-30), and fixed here.

- **A decoder bias inflated the codebook's cost.** The first version decoded the grid's cue-map
  output to the nearest place code by raw dot product, found 0.695 against the control's 0.999,
  and concluded that "the loss is in the correlated codebook". A noisy cue's place activity is
  dominated by the component every code shares, so a dot product favours high-norm codes: wrong
  picks sat at the 85th–98th percentile of code norm. By cosine the same read recovers 0.963–0.976.
  The loss is in the snap. The registered "ridge + best stored address" read in the E2 tables
  also uses a dot product; by cosine it recovers about 0.99 for every placement at 10% flips,
  which is why placement matters much less on that read.
- **The paper quote.** "Even deep in the memory continuum" is from the bioRxiv preprint, not the
  review response. It concerns recovery of the *scaffold state* from a corrupted sensory cue, which
  is what this repository measures. The first version also said the paper's choice of Ns "put the
  peak just out of frame", which implied concealment. Ns ≥ P is the paper's stated condition, Ns =
  3,600 is the smallest it allows, and its S8 caption names the mechanism.
- **"One matrix predicts it."** The error law's rank correlation with accuracy is almost entirely
  between placement types. Within a placement it predicts little, and P1 and X1 failed on some
  content. It ranks placements. That is all the text now claims.
- **The learned placement's warm start.** The error-law search also started from the noise-law
  placement. The first version did not say so. From random starts alone, it falls below k-means on
  sentences (0.547 against 0.677).
- **The objective is known.** The error law is discriminative k-means (Ye, Zhao & Wu 2007), up to
  group-size scaling. The swap search is Kernighan–Lin-style partitioning.

## What did not survive

- **"Dense cues break at the embedding dimension."** A pilot claimed that the item count at which
  dense-cue recall halves tracks the embedding dimension d, not Ns. With the library's scaffold it
  goes the other way: P50 grows 2.8–3.5× from Ns = 2,048 to 8,192 at every d
  ([`bench/dense.py`](bench/dense.py)). On LoCoMo, ranking turns by the ridge's own coefficients,
  before the place layer, gets R@1 0.227, above kNN's 0.188. So the loss is in the 400 place cells
  that conversations of 369–689 turns are squeezed through.
- **The registered learned placement.** A soft Sinkhorn assignment with each module fitted to the
  previous module's residual re-found module 1's partition in module 2. The swap search replaced it
  before the confirmatory run (amendment 1).
- **The noise law as a thing to minimise.** It survives as a description of the pseudo-inverse. As
  an objective under the ridge it is misleading (amendment 4).
- From the critic's pilot, before any of this: balanced occupancy does not buy robustness by
  itself, and a wrong recall is a related stored memory only at chance (24% against 22%).

Smaller things, measured and left in because changing them would change registered runs: the
life-log has a few exact duplicate entries (≈ 1/P on every arm alike); at 400 place cells, 6 of 20
scaffold seeds have one address out of 3,600 that is not a fixed point; and the random streams of
content and cues across different seeds overlap (never within a paired comparison).

## The upstream code

The port reproduces `FieteLab/VectorHaSH` at `c71317d` exactly. Its sensory error is identical to
every printed digit (0.2076 / 0.3091 / 0.3478 at 1,001 / 2,001 / 3,001 patterns). The clean-cue
overlap follows m = erf(√(Nh / (2(P − Nh)))) to within 0.002 (0.681 / 0.434 / 0.304 measured
against 0.683 / 0.436 / 0.305).

[`docs/AUDIT.md`](docs/AUDIT.md) lists twelve things a reader of the paper would not expect from the
code, each with file:line and its measured effect. A few examples:

- grid-to-place connectivity is 0.67, not the 0.60 its comment says;
- one baseline alone gets an MI floor in the comparison plot;
- the sequence plot's Vector-HaSH curve is loaded from the item-memory file (the notebook states
  this shortcut);
- the sequence demo resets to the true state at every step. Free-running recall with its settings
  fails at step 8, but runs all 1,001 steps without the clean-up step, or at 800 place cells.

The authors released working code under MIT, and none of this study would exist without it. The
placement question is one the paper itself leaves open ("the scaffold trajectory can be arbitrarily
chosen"), and the paper already proposes routing novel inputs to fresh scaffold states.

## Reproduce

```bash
uv sync --extra bench
uv run pytest                                              # 15 tests, seconds
uv run python bench/cliff.py                               # E1
uv run python bench/paper_figs.py                          # S7 / S8, SNR, blind ridge (~5 min)
LOCOMO=path/to/locomo10.json uv run python bench/dense.py  # dense cues + LoCoMo (~6 min)
uv run python bench/placement.py --content factored        # E2 (~15 min on 8 cores)
uv run python bench/placement.py --content lifelog         # E2 on sentences (~20 min)
uv run python bench/placement.py --verdict                 # the E2 criteria, from the result files
uv run python bench/budget.py                              # equal-weight controls (seconds)
uv run python bench/figures.py
```

All randomness is seeded; the result files in [`results/`](results) are the ones quoted here.
Timings are on an Apple M5 shared with other jobs. `locomo10.json` is from snap-research/locomo;
MiniLM is read from the local Hugging Face cache.

```
src/loci/scaffold.py   the grid scaffold: addresses, place codes, the snap
src/loci/memory.py     binding content to addresses; the write rules; recall
src/loci/place.py      placements, the noise and error laws, the swap search
src/loci/content.py    factor-structured content with known ground truth
bench/                 one script per experiment, and the figures
docs/                  DESIGN, PREREG (with amendments), AUDIT
```

## Credits

Built on Vector-HaSH: S. Chandra, S. Sharma, R. Chaudhuri & I. Fiete, "Episodic and associative
memory from spatial scaffolds in the hippocampus", *Nature* 638 (2025),
doi:10.1038/s41586-024-08392-y; code at FieteLab/VectorHaSH (MIT, see [`NOTICE`](NOTICE)). This
repository is MIT-licensed.
