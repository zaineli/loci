# loci - design

*2026-09-30. Written before any experiment beyond the reproduction and the team's pilots; the
pilots are marked as such.*

## What this builds on
Vector-HaSH (Chandra, Sharma, Chaudhuri & Fiete, *Nature* 2025): a fixed grid-cell scaffold gives
prod(lambda_m^2) error-correcting addresses; content is bound to them one-shot by heteroassociation
(`W_hs = H S+`, `W_sh = S H+`); a cue is recalled by snapping `ReLU(W_hs s~)` to an address and
reading `sign(W_sh h)` there. Past Nh stored items the read-out degrades smoothly - the paper's "no
memory cliff" - and we have verified it exactly: overlap `m = erf(sqrt(Nh / (2 (P - Nh))))`
(0.681 / 0.434 / 0.304 measured at P = 800 / 1,600 / 3,000 against 0.683 / 0.436 / 0.305).

In the paper, *where* each item goes is given: a fixed traversal of the scaffold, or true actions.
The authors write that for episodic memories without metric variables "the scaffold trajectory can
be arbitrarily chosen". That is the gap: whether memory should learn where to put things.

## Three facts that shape everything (pilots by the team, verified in code)
1. **The cliff is in the cue, not in the store.** With noisy or partial cues, recovery of the right
   address collapses exactly where the number stored P reaches the cue dimension Ns - the
   pseudo-inverse's peak at alpha = 1, textbook for projection-rule memories (Kanter & Sompolinsky
   1987; Krogh & Hertz 1992) and double descent's (Belkin et al. 2019; Hastie et al. 2022). The
   scaffold then snaps to a *valid but wrong* address, and nothing downstream compares the cue with
   the recall. The paper's supplement shows the noisy-cue loss (Figs S7-S8), yet the preprint and
   the review response say the scaffold recovers the exact state from corrupted cues "even deep in
   the memory continuum"; the peak itself is invisible there because Ns = Npos = 3,600 puts P = Ns
   at the right edge of every capacity axis, and the only Ns sweep (Fig S6) uses clean cues. A
   noise-matched ridge (Bishop 1995) removes the peak.
2. **Dense cues break it at the embedding dimension.** A LoCoMo question and its evidence turn sit
   at MiniLM cosine ~0.51, which is ~33% bit flips after a sign projection; Vector-HaSH's R@1 is
   <= 0.004 against kNN's 0.180 pooled over the benchmark, and the item count at which recall
   halves tracks the embedding dimension d, not Ns.
3. **Nearness on the scaffold is shared residues, not distance.** Place codes correlate only by the
   number of module phases two addresses share (0.00 / 0.26 / 0.58 / 1.00 for 0-3); positions one
   step apart share none. The scaffold is a residue number system.

## The thesis this repository tests
*Where a memory is put decides what survives.* Placement enters the model only through which place
codes the stored items get (`H_a`), and with the paper's linear cue map the noise on module m's
phase k is sigma^2 q_mk^T (S^T S)^-1 q_mk, q_mk marking the items placed there. Correlated items
have negative off-diagonal entries in (S^T S)^-1, so putting them on a shared phase cancels noise:
the best placement groups memories by **partial** correlation - conditional dependence, the
quantity a Gaussian graphical model is built on - not by raw similarity. Two first guesses did not
survive the critic's pilot and are not claims: balanced occupancy does not buy robustness by itself,
and a wrong recall is not, beyond chance, a related stored memory (most errors read a blend at an
empty address; the blend is where gist lives). The claim is about linear cue maps into product
address spaces, and near the cue limit only: at P << Ns every placement recalls perfectly.

## Experiments
- **E0 Reproduction.** The paper's capacity curves, the erf law, the fixed points, seeded.
- **E1 The cliff in the cue.** Recovery and overlap against P/Ns for flip rates 0-33% and masked
  cues, Ns in {500, 1,000, 3,600}, Nh in {200, 400, 800}, 10 seeds; the pseudo-inverse against the
  MMSE ridge write (`alpha = 4 f (1 - f) P`), Hebbian, and a matched-filter upper bound; theory
  curves from the closed-form noise variance. Dense cues: sign projections of Gaussian vectors at
  d in {64, 128, 384, 1024} and of MiniLM turns.
- **E2 Where a memory is put decides what survives (the flagship).** Factor-structured content and
  a life-log of sentences. Placements: sequential, random, k-means, oracle, **learned** (minimising
  the noise law under balanced marginals), and a no-product control; read by pinv, ridge, and
  ridge + decode to the best stored address - placement is the only thing learned, because a
  learned read path is just product quantisation. Criteria in `docs/PREREG.md`.
- **E3 (folded into E1)** One table of LoCoMo turns: the pilot already says item retrieval is not
  where scaffold memories win (R@1 0.004 against kNN's 0.18), and it is reported as such.

Every experiment states its pass and kill criteria before it runs (`docs/PREREG.md`), uses
multiple seeds, and reports against its control.
