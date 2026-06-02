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
