# Pre-registration

*Written 2026-09-30, after the reproduction (E0) and the first E1 sweep, before any E2 run. The
criteria below decide what the README may claim; a criterion that fails is reported as failed.*

## Scope, stated once
loci studies **how scaffold memories work and fail** - Vector-HaSH-style memories with a linear
cue map into a product address space. It is not a proposal for a better agent memory: at equal
bytes, an exemplar store (kNN) beats every variant here on item recall, and every table says so.

## E1 - the cliff in the cue (regime map for the paper's headline claim)
Already run (`results/cliff.json`, 10 seeds): the pseudo-inverse's minimum sits at P/Ns = 1.0 for
Ns in {500, 1,000, 3,600} and every flip (5-20%) and mask (25-50%) rate. Still to run, with the same
seeds and scaffold:
1. **The paper's own figures in its own metric.** SI Fig S7-right (MI per input bit vs P, 10%
   flips) and Fig S8 (p(correct address) vs flip rate at P in {250, 500, 1,000, 2,000, 3,000}) at
   Ns = Npos = 3,600. Pass: within 0.03 of the published curves where they can be read off.
2. **Masked and flipped cues collapse onto one SNR axis**, SNR = a^2 / Var(xi) (flips:
   (1 - 2f)^2 / 4f(1 - f); masks: (1 - r) / r). Pass: at each P/Ns, recovery as a function of SNR
   differs between the two families by <= 0.05.
3. **Ridge set blind to the noise**: alpha = c P for c in {0.1, 0.3, 1} across all noise levels,
   with its clean-cue cost. Pass: some single c stays within 0.05 of the noise-matched alpha at
   every flip rate, and costs <= 0.02 clean recovery at P/Ns <= 1.
4. **Dense cues**: sign projections (Ns = 4,096) of Gaussian vectors in d in {64, 128, 384, 1,024}
   at cue cosine 0.9; P50 (items at which recovery halves) against d. Pass: P50 / d within a
   factor 1.5 across d, and flat in Ns in {2,048, 4,096, 8,192}. And one table on LoCoMo turns
   (MiniLM): VH pinv / ridge / ridge + stored-address decode against kNN on floats and on 1-bit
   codes, per-conversation R@1 and R@5, 3 projection seeds, bits printed for every row.
The mechanism and the fix are textbook (projection-rule basins vanishing as alpha -> 1: Personnaz,
Guyon & Dreyfus 1986, Kanter & Sompolinsky 1987; the peak at alpha = 1 and its removal by weight
decay: Krogh & Hertz 1992; double descent: Belkin et al. 2019, Hastie et al. 2022, optimal ridge:
Nakkiran et al. 2021; noise as Tikhonov: Bishop 1995). The contribution is the regime map for a
Nature model's claim, the valid-but-wrong-address failure, and that the paper's Ns = Npos hid it.

## E2 - where a memory is put decides what survives
**Law under test** (derived; checked in a pilot): with the pseudo-inverse, the noise on module m's
phase k is Var(n_mk) = sigma^2 q_mk^T (S^T S)^-1 q_mk, q_mk marking the items placed on that phase
(ridge: q^T (G + a I)^-1 G (G + a I)^-1 q). Positively correlated items have negative off-diagonal
entries in (S^T S)^-1, so co-locating them *cancels* noise: optimal placement groups items by
**partial**, not raw, correlation.

**Content.** s = sign(a_A + b_B + c_C + z): factor A in 9, B in 16, C in 5, item detail z; and a
life-log of templated sentences (people x projects x activities) embedded with a local sentence
encoder and sign-projected. Ns = 1,000; P/Ns in {0.2, 0.4, 0.6, 0.8, 0.9}; 10 paired seeds (same
patterns and noise draws across arms); flips 10% and 20%; bootstrap 95% CIs on paired differences.

**Placements.** sequential (paper), random, k-means residue (per module, not learned), oracle
residue (A -> module 0, B -> module 1, C x slot -> module 2), **learned** (per-module soft
assignments minimising sum_mk q_mk^T Gamma q_mk under Sinkhorn-balanced marginals, module m+1 on the
residual of module m, open addressing in the last module), and a **no-product control** (random
sparse place codes of the same size, same decode).

**Read paths** (placement is the only thing learned): pinv; ridge; ridge + decode to the best
*stored* address.

**Criteria.**
- P1 (the law): across all placements and seeds, per-module accuracy is monotone in the measured
  q^T Gamma q (Spearman rho <= -0.8 within each module).
- P2 (placement matters near the cue limit, and only there): at P/Ns = 0.8, 10% flips, ridge +
  stored decode, oracle beats random by >= 0.10 (CI excludes 0); at P/Ns = 0.2 all placements are
  within 0.02.
- P3 (learning is not decoration): learned beats k-means by >= 0.05 at P/Ns = 0.8, 10% flips,
  ridge + stored decode, paired CI excluding 0. If it fails, "learned placement" is reported as no
  better than clustering.
- P4 (the scaffold is not decoration): aligned placement's gain over random is >= 0.05 larger on
  the grid scaffold than on the no-product control. If not, the gain is a property of any
  structured code, not of the grid.
- P5 (what survives, measured without the cue confound): gist = factors decoded from the *read-out*
  when the item is wrong; and facet completion = cue missing factor B, memory must return B.
  Reported against kNN whatever it shows (the pilot has kNN at 1.00 and VH at 0.10-0.37).
- Equal-synapse worth: the fold-increase in Ns that random placement needs to match oracle.

Collisions reading out prototypes follows from the algebra; it goes in an appendix, not a claim.

## Amendments, before the confirmatory E2 run (2026-09-30)
Logged before any E2 run beyond the seed-0 prototype and a 2-seed quick run of `bench/placement.py`.
No threshold changed.
1. **The learned optimiser.** The registered one (Sinkhorn-relaxed soft groups, module m + 1 on the
   residual of module m) failed in the prototype: module 1 re-found module 0's partition (NMI 0.81
   with factor A, 0.08 with B), because Gamma amplifies the low-variance directions a residual
   leaves. A block-balanced Sinkhorn without residuals stalled in mixtures of factors (law -153 x
   1e-3 against -235 for factor B's own partition). Replaced by pairwise-swap descent on the same
   law, per module, from 20 random balanced labellings, keeping the lowest; no residual, nothing
   coupling one module to another. Prototype: the learned groups have NMI 0.93 / 0.92 / 0.96 with
   A / B / C, found without supervision.
2. **Ambiguities, resolved.** P1 is evaluated at the headline condition (P/Ns = 0.8, 10% flips), for
   the pseudo-inverse against Gamma(0) and the ridge against Gamma(alpha); it passes only if all six
   rho <= -0.8. "Aligned placement" in P4 is the oracle. P2-P4 use ridge + stored decode, as
   registered. Every criterion is evaluated separately on each kind of content.
3. **The no-product control, as built.** A one-module scaffold of period 60: 3,600 phases, the same
   400 place cells, random sparse place codes, and a snap that is the nearest of all 3,600 codes -
   3,600 grid cells against the grid's 50. In the quick run it recovered 0.95 (pinv) and 0.998
   (ridge) where the grid recovered 0.15 and 0.51, whatever the placement. Added as exploratory, on
   both scaffolds: ridge + the nearest of all 3,600 place codes, no module snap - to tell the grid's
   codebook from its decoder.
4. **The error law, added after the life-log pilot, judged on fresh seeds.** In a 2-seed quick run on
   the life-log, the learned placement lowered the noise law below random and recovered *less*
   (ridge 0.29 against 0.33). The noise law counts only the variance cue noise adds; it leaves out
   the ridge's bias, the share of a clean cue spread onto similar items, which dense content makes
   large. Bias plus variance at the noise-matched ridge is Var(xi) q^T (G + alpha I)^-1 q on every
   phase - the posterior covariance of the cue's coefficients over the stored items - so the
   **error law** replaces Gamma with (G + alpha I)^-1; at alpha = 0 the two laws agree. A pilot on
   seeds 0-1 (P/Ns = 0.8, 10% flips) had learned-on-the-error-law at ridge 0.69 / 0.71 on the
   life-log (noise law 0.26 / 0.31, k-means 0.74 / 0.59) and 0.87 / 0.72 on factored content
   (noise law 0.87 / 0.80). The added arm, **error**, is the same search as learned on the error
   law, from 20 random starts and from the learned placement. It and these criteria are judged on
   **seeds 10-19 only**, at the headline condition, ridge + stored decode unless stated:
   - X1: per-module ridge accuracy against the measured error law, pooled over the six placements,
     Spearman rho <= -0.8 in every module, on each kind of content.
   - X2: on the life-log, error beats learned by >= 0.05 (paired CI excluding 0); on factored
     content, error is not worse than learned by more than 0.02 (lower CI bound >= -0.02); on both,
     error beats k-means (CI excluding 0).
   The registered criteria stay on seeds 0-9 and the five registered placements.

---

# Round 2 (2026-10-01)

*Written after round 2's pilots (seeds 0–2, in the lead's and agents' scratch folders) and before
any confirmatory run. Round 2 uses seeds 40–49, which no pilot has touched. This section's git
commit precedes every result file it governs.*

## E3 — a theory that predicts recall, tested out of sample
The theory (`src/loci/theory.py`, level "relu") predicts per-module and address recovery from the
stored patterns, the scaffold and the placement, with no fitted parameter and without simulating a
cue. In development it reached MAE 0.010 on 180 factored-content conditions (seeds 0–2), which is
the sampling floor of one cue per item. The out-of-sample test is run from a frozen copy of the
development code, whose sha256 hashes are recorded. Every prediction is written, hashed and
timestamped before any measurement of the new conditions:
- content: life-log, cards (7, 12, 4);
- scaffold: Nh 200 and 800, periods (5, 6, 7) and (3, 4, 5, 7);
- cue dimension: Ns 500 and 2,000;
- cues: 25% and 50% masks;
- read: the pseudo-inverse;
- loads 0.3, 0.5, 0.7;
- placements: the round-1 set;
- seeds 30–32.

- **T1:** over all out-of-sample conditions, MAE ≤ 0.02 and ≥ 90% within 0.05 of measured address
  recovery.
- **T2:** no condition family (as listed above) has MAE > 0.04. Any family that does is reported by
  name as where the theory breaks.

E4 below is a second out-of-sample test. None of its placements (encode, replay) or content kinds
other than factored were in the theory's development set:
- **T3:** over all E4 rows, MAE ≤ 0.02 and max |err| ≤ 0.06.

## E4 — a memory that files itself
`bench/consolidate.py`. Encoding stores each arriving item at the free address its own recall
prefers: argmax over free addresses of Σ_m (Q_mᵀ c)_{φ_m(a)}, with c its ridge coefficients over the
items stored so far. Capacity grows with the count so far, so the final P is never used. Replay moves
each item to where its trace-free recall prefers, which is exact coordinate descent on the error law
over realized addresses. Every module keeps its full phase count (9, 16, 25). No labels, and no group
counts matched to the content.

- **Content kinds:**
  - factored (9, 16, 5);
  - cards (7, 12, 4);
  - four factors (6, 10, 4, 8) on three modules;
  - a 5 × 4 hierarchy with no product structure;
  - the life-log.
- **Loads and noise:** P/Ns 0.4 and 0.8; 10% and 20% flips.
- **Headline condition:** P/Ns 0.8, 10% flips, the scaffold's own snap.
- **Statistics:** paired bootstrap 95% intervals over seeds 40–49.

- **S1:** encode + replay beats k-means by ≥ 0.05 on every content kind (CI excluding 0).
- **S2:** where an oracle exists (factored, cards, life-log), encode + replay is not worse than the
  oracle by more than 0.03 (CI lower bound ≥ −0.03).
- **S3:** encoding alone, one pass with no replay, beats random by ≥ 0.15 on every content kind.
- **Reported without a threshold:**
  - the ablation replay-from-random. In the pilots it came close to encode + replay, so replay does
    most of the optimising, and the encoding's value is that the memory is organised from the first
    item on;
  - oracle + replay;
  - the NMI of each module with each factor;
  - the secondary loads and noise levels.

## E5 — the price of imagination
Criteria to be added in an amendment after the composition, replay and theory pilots report, and
before the E5 confirmatory run on seeds 40–49.
