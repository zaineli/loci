> **About this note.** This is the round-2 theory note, written by the team's theorist while the
> experiments were being designed. It keeps predictions and checks in the order they happened, and
> it separates what is proved, what is derived under stated approximations, and what is conjectured.
> Its checks ran on pilot seeds 0–2. The confirmatory numbers are in the README and `results/`,
> which supersede this note where they differ.
> - **Scripts.** The development scripts are in [`theory/`](theory), as they ran, except that their
>   absolute paths were made relative to the repository. The `i*` scripts import the composition
>   pilot's helpers (`theory/compose.py`).
> - **The theory itself** is in [`src/loci/theory.py`](../src/loci/theory.py), identical in its
>   output to `theory/t4_theory.py`'s "T-relu" level.

# loci round 2: the theory note

*Theorist, 2026-10-01. Part A is the mathematics. Part B is the predictions, written down before any
check was run. Part C is the checks: what matched and what did not. Status tags: **[proved]** means
exact algebra; **[derived]** means it follows under stated approximations; **[conjecture]** means
reasoned but not derived.*

Notation. S is the Ns x n matrix of stored patterns, K = (S^T S + alpha I)^-1, and J = sum_m J_m with
J_m = sum_k q_mk^T K q_mk (a *sum* over phases). Q_m is the n x k_m phase-indicator matrix of module
m, so J_m = tr(Q_m^T K Q_m). a = 1 - 2f is the cue gain and sigma^2 = 4f(1 - f) the cue-noise variance
per bit. The bench's alpha is alpha = sigma^2 P / a^2. H_a is the Nh x n matrix of stored place codes,
and t_mk is row (m, k) of W_gh, so t_mk = (1/Npos) sum over addresses with phi_m = k of h_a.

---------------------------------------------------------------------------------------------------

## Part 0. The price of imagination (round-2 thesis; this section supersedes the rest where they overlap)

*Written in two passes. §0.1–0.8 are derivations and predictions, fixed before any imagination check
was run. §0.9 is the check. Tags as below. The setting follows gold2/compose:*
- factor content s = sign(A[a] + B[b] + C[c] + z), with cardinalities 9, 16, 5 and unit detail;
- 16 held-out (a, b) pairs, one per b, never stored;
- aligned placement: A on module 0, B on module 1, C on a 5-slot group of module 2, with slots
  filled 0, 1, ... by open addressing;
- P = 800, Nh = 400, Ns = 1,000, and the MMSE alpha for 10% flips (450), unless stated.

Notation:
- factor cue s_f = sign(X), with X = A[a] + B[b] + C[c];
- recombination lure s_l = sign(X + z'), a new detail z' on a held-out (a, b);
- unrelated lure: a random +/-1 pattern;
- recall coefficients c(s) = K S^T s, place activity u = H_a c, h0 = ReLU(u), and C^(m) = Q_m^T c
  the phase sums;
- Stein slopes: kappa_1 = sqrt(2/pi)/2 = 0.399 for a stored item (var X + z = 4), and
  kappa_f = sqrt(2/pi)/sqrt 3 = 0.461 for the factor cue.

### 0.1 Corrections folded in from the other agents (and what they do to Parts A–C)
- **Norm bias.** A raw-dot-product nearest-code read, argmax_j h_j^T h0, favours large-norm codes.
  Use cosine. My item-5 "joint stored decode" (t5) was raw dot. The coding engineer's cosine
  numbers replace it:

  | flips | M = 3 | M = 4 | gain from the 4th module |
  |---|---|---|---|
  | 10% | 0.988 | 0.999 | +0.011 (ceiling) |
  | 20% | 0.721 | 0.916 | **+0.195** |

  The item-5 conclusion (a 4th module helps the joint decode, hurts the snap) survives at 20% flips
  only. At 10% the cosine joint decode is already at ceiling. The bench's "stored" read (0.833,
  random) is also norm-biased; cosine gives 0.988.
- **The error law is discriminative k-means [proved, critic C7].**
  J = (P - tr(Q^T G (G + alpha I)^-1 Q)) / alpha. Minimising J is kernel k-means with the ridge
  smoother G(G + alpha I)^-1 on balanced, uncentred groups (Ye, Zhao & Wu 2007). So item 2's replay
  is Hartigan-style kernel k-means. The address-level fix (2.7) is the same objective on the
  realized product of phases.
- **Rank correlations of the law are between placement types (critic C5).** Within-type rho is
  -0.14 to -0.62, and placement identity explains R^2 0.92–0.97. That applies to the error law. It
  does not apply to the level-T theory of Part C4. Within each (placement, load, flip) cell, the
  seed-to-seed deviations of measured recovery (sd 0.033) are tracked by level T with Pearson
  **0.95** (modules 0.95). The error law alone gives -0.53. So level T predicts magnitudes and
  within-type variation; the error law predicts neither.

### 0.2 Proposition I.1: the phase sums of a factor cue [derived: Stein linearisation, residual Gram ~ rho I]
Write S ~ kappa_1 F M^T + R, where F holds the 30 factor codes, M (P x 30) the factor indicators and
R the residuals (per-bit variance 1 - 3 kappa_1^2 = 0.523). Also s_f ~ kappa_f F e_abc + r_f. Take
F^T F ~ Ns I and R^T R ~ Ns (1 - 3 kappa_1^2) I, and push through (M M^T + eps)^-1 M. Then

    (I.1)   c(s_f) ~ M theta,   theta = (kappa_f / kappa_1) (N + eps I)^-1 e_abc,   N = M^T M,
            eps = ( Ns (1 - 3 kappa_1^2) + alpha ) / ( Ns kappa_1^2 )      (= 6.1 at P = 800, 4.7 at P = 400)

Under aligned placement Q_m = M_block(m), so C^(m) = [N theta]_block(m). The target phase's contrast
over the best competitor is

    (I.2)   Delta_f ~ (kappa_f / kappa_1) n_f / (n_f + eps) - h_f ,

where h_f is the held-out "hole": phase a gets none of b's mass, since (a, b) is never stored.
Module 2 decides among 25 slots, not 5 groups. Open addressing puts a C-value's items mostly in slot
0, a share omega(mu) = (1 - e^-mu)/mu, with mu the items per (a, b, c) cell. So

    (I.3)   Delta_2 ~ omega(mu) (kappa_f/kappa_1) n_C/(n_C + eps) - (A- and B-mass in the best
            competing slot) .

The numbers, computed from the counts alone (i0_inputs.py):

| P | Delta_0 | Delta_1 | Delta_2 per slot (mean) | Delta_2 (min over queries) | omega |
|---|---|---|---|---|---|
| 800 | 1.06 | 1.01 | 0.59 | 0.45 | 0.571 (measured slot-0 share 0.56–0.58) |
| 400 | 1.01 | 0.93 | 0.76 | 0.59 | 0.744 |

**Module 2 is the bottleneck, and its margin falls with load through omega.**

### 0.3 Proposition I.2: the construction condition [derived]
The snap lands on (a, b, C's group) iff, in every module,

    (I.4)   beta_m Delta_m  >  max_k [ d_mk - d_mk* ] ,

where d_m is a sum of distortions.
- (i) *Template leak.* Sum over m' of Ftilde^{mm'} C^(m') (the ANOVA of W_gh H, Part C1). Per unit
  phase sum, the rms is 0.07–0.09 at Nh = 400 and 0.05–0.07 at Nh = 800. The large C^(0)_a and
  C^(1)_b (about 1) leak into module 2.
- (ii) *The cue's own nonlinear remainder*, r_f (0.363 per bit), projected through K. This is
  deterministic per query. Its sd is about sqrt(0.363 q^T (K - alpha K^2) q), roughly 0.08 per slot
  at P = 800 (the oracle's module-2 law).
- (iii) *For flipped cues, the flip noise* (4.1): gain a = 0.8 and variance 0.36 (K - alpha K^2).

Hence the construction rate is about the product over m of P(margin_m > distortion_m):
- **Load:** the contrasts are flat in P, apart from omega(mu), which falls with load, and the
  remainder noise, which grows with the law.
- **Nh:** the template leak scales roughly as Nh^-1/2.
- **alpha:** at alpha = 0 and P -> Ns, K = (S^T S)^-1 amplifies (ii) and (iii). That is the
  cue-side interpolation peak.
- **Detail variance w^2:** kappa_1 = sqrt(2/pi)/sqrt(3 + w^2), so the ratio kappa_f/kappa_1 rises
  and eps rises. The contrasts barely move while n_f >> eps.

**Predictions (A)**, landing rate on (a, b, C-group), 3 seeds:

| P | Nh | alpha | cue | predicted |
|---|---|---|---|---|
| 800 | 400 | MMSE | clean | 0.97 +/- 0.02 |
| 800 | 400 | MMSE | 10% flips | 0.93 +/- 0.03 |
| 800 | 800 | MMSE | clean or 10% | >= 0.99 |
| 400 | 400 | MMSE | clean | >= 0.98 |
| 800 | 400 | 0 (pinv) | clean | <= 0.8 (module 2 collapses) |

Modules 0 and 1 land >= 0.995 everywhere.

### 0.4 Proposition I.3: the read-out interpolation peak [proved (a)–(c); derived (d)]
- (a) If P <= Nh and H_a has full column rank, W_sh H_a = S H_a^+ H_a = S. Stored read-outs are
  exact however ill-conditioned H_a is.
- (b) At an unstored address, W_sh h_new = S w with w = H_a^+ h_new, and
  ||w|| <= ||h_new|| / sigma_min(H_a). sigma_min -> 0 as P -> Nh (square case). For isotropic
  features, E||w||^2 = L(rho), with rho = P/Nh:

      L = rho/(1 - rho) for rho < 1,  and  L = 1/(rho - 1) for rho > 1   (Hastie et al. 2022).

  This is the **read-out-side interpolation peak at P = Nh**. It mirrors the cue-side peak at
  P = Ns, where W_hs = H_a S^+ has noise gain P/(Ns - P). It is invisible to stored-item metrics
  by (a).
- (c) Ridge: W_sh(lam) = S H_a^T (H_a H_a^T + lam I)^-1 gives ||w_lam|| <= ||h_new|| / (2 sqrt lam),
  because sigma/(sigma^2 + lam) <= 1/(2 sqrt lam), uniformly in P. The peak is removed. The cost
  is that each stored singular direction is shrunk by sigma^2/(sigma^2 + lam).
- (d) The overlap with the composite is Omega(rho) = E[sign(tau m + eta) sign X], with
  Var(eta) = 0.52 L(rho). tau = 1 for rho > 1. For rho < 1, tau lies between rho (isotropic) and 1
  (signal in the leading main-effect directions).

**Predictions (B)**, the read-out at the address the factor cue lands on, and at the most-used
empty slot (Omega is bracketed by tau = rho and tau = 1 for rho < 1):

| P/Nh | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | 2.25 |
|---|---|---|---|---|---|---|---|---|
| Omega | 0.2–0.5 | 0.25–0.5 | 0.15–0.35 | **< 0.05** | 0.3 | 0.4 | 0.49 | 0.52 |

With lam = 0.01 x the mean eigenvalue of H_a H_a^T, Omega(1.0) >= 0.4 and the stored read-out
costs <= 0.01.

### 0.5 Proposition I.4: a lure is an attenuated, noisy factor cue [derived]
Project the lure onto the factor cue:

    (I.5)   u(s~_l) = gamma' u(s_f) + xi ,   gamma' = a kappa_l / kappa_f = 0.8 x 0.866 = 0.693 (10% flips),
            xi ~ N(0, sigma_e^2 H_a K S^T S K H_a^T),   sigma_e^2 = 1 - 2 gamma' a (2/3) + gamma'^2 = 0.741 ,

using E[s_l s_f] = (2/pi) arcsin(sqrt 3 / 2) = 2/3. For comparison, the 10%-flipped factor cue has
gain 0.8 and noise 0.36. **A recombination lure is a factor cue at about 2.7x lower SNR.**

### 0.6 Proposition I.5: construction and false recall are one event [proved, given I.4]
**The family.** A decoder D chooses a state t from a state set, by maximising score(h0, t) + b_t.
- The score is positively homogeneous of degree 0 in h0. Examples: the cosine cos(h0, h_t), or the
  separable per-module sum of W_gh h0, whose argmax is scale-free.
- b_t is a fixed state prior, for example lambda on stored tuples.

**The claim.** Then

    (I.6)   P[ D(lure) = t ] = P[ D( factor cue with noise xi / gamma' ) = t ]    for every t and every b .

*Proof.* ReLU is positively homogeneous: ReLU(gamma' u_f + xi) = gamma' ReLU(u_f + xi/gamma').
The score ignores the factor gamma' > 0, and b does not depend on h0. ∎

So:
- *Construction implies false recall.* Construction at bias b with margin m against the next state
  gives false recall of the matching lure with probability >= 1 - sum_t Phi(-m / sd(xi_t - xi_T)
  / gamma'^-1). This goes to 1 as the margin grows against the lure's noise.
- *No bias separates them.* Any b that stops false recall must either remove the margin, which
  also stops construction of the clean cue, or leave a margin smaller than the lure's noise. In
  that case construction of a cue with *matched* noise fails at the same rate. Every decoder in
  the family trades construction against conjunction errors along one curve.
- *The way out.* Only information outside h0 can separate them, above all the lure's own detail
  z'. The factor cue has no detail to match, while the lure has detail the memory never stored. A
  cue-to-read-out match (recognition) uses exactly that.

*Status.* Exact given I.4. I.4 is the approximation: linear Stein projection, and xi taken as
query-independent Gaussian.

### 0.7 Proposition I.6: the frontier, and where the snap's cost really comes from [derived]
The family is D_lam(h0) = argmax over all 3,600 tuples t of [cos(h0, h_t) + lam 1(t stored)]:
- D_0 is the nearest place code among all tuples;
- lam >= 2 gives the joint cosine decode;
- the paper's snap is a separable Hebbian approximation to D_0.

Split the snap's noisy-recall deficit:

    (I.7)   joint - snap  =  (joint - D_0)          [price of admitting never-stored states]
                           + (D_0 - snap)           [price of per-module Hebbian decoding]

**Predictions (C)**, 10% flips, P = 800:

| quantity | random placement | aligned placement |
|---|---|---|
| price of admitting never-stored states | <= 0.02 | ~0 |
| price of per-module Hebbian decoding | ~0.49 | ~0.12 |
| construction (factor cue) | ~0 for every lam | ~1 at lam = 0; falls to ~0 by lam* ~ 0.05–0.15 (cosine units) |
| false recall (recombination lure) | ~0 for every lam | follows construction(lam) by (I.6), left-shifted and softer (SNR 2.7x lower) |
| noisy recall across the lam range | changes by <= 0.02 | changes by <= 0.01 |

Here lam* is the empty address's cosine margin over the best stored code.

**The price of imagination is paid in false memory, not in recall accuracy.**

### 0.8 Proposition I.7: recognition d′ [identity proved; class moments derived]
For a familiarity signal m, the studied class is a mixture:
- correct recalls, with weight rho (the recall accuracy), mean mu_ok and sd s_ok;
- failed recalls, with weight 1 - rho, mean mu_err and sd s_err.

Then

    (I.8)   d' = (rho mu_ok + (1 - rho) mu_err - mu_l) / sqrt( (s_s^2 + s_l^2) / 2 ),
            s_s^2 = rho s_ok^2 + (1 - rho) s_err^2 + rho (1 - rho)(mu_ok - mu_err)^2 .

Suppose a lure is as familiar as a failed recall (mu_l = mu_err) and the within-class spreads are
small. Then **d′ <= sqrt(2 rho / (1 - rho))**: recognition is capped by recall accuracy.

Predictions for the critic's signal cos(h0, h_snap), at P = 800 with 10% flips:
- **Random placement** (rho ~ 0.48, cap 1.36): failed recalls land on unrelated-looking tuples, and
  the lures look like failed recalls. Related and unrelated d′ are then both ~1.4–2.5 and about
  equal.
- **Aligned placement** (rho ~ 0.87, cap 3.7): a failed recall lands on a 2-phase neighbour, so
  mu_err ~ mu_ok and the mixture term is small.
  - An unrelated lure is far less familiar than any studied item: d′ >= 5.
  - A recombination lure lands on a coherent empty address, which is more familiar than a failed
    recall: d′ 2–3.
- **Under the joint cosine decoder** (rho ~ 0.99):
  - unrelated-lure d′ rises;
  - related-lure d′ does not rise, because the lure lands on a stored 2-factor neighbour with high
    cosine.
- **The cue–read-out match** cos(s~, sign W_sh h_out) separates related lures better than the h0
  signal does. The empty address carries no detail, while a studied read-out carries the item's
  own detail (overlap ~0.9 at the oracle). Predicted d′(related) ~3.

### 0.9 Checks (run after 0.1–0.8 were fixed; 3 seeds; scripts `i0`–`i4` beside this note)

#### A. Construction condition (`i0_inputs.py`, `i1_construct.py`)

Landing on (a, b, C-group) for the 80 held-out queries. The compose column is gold2/compose's
measurement; the mine column is the same stored sets, recomputed.

| P / Nh | cue | predicted (fixed beforehand) | model | compose | mine |
|---|---|---|---|---|---|
| 800 / 400 | clean | 0.97 +/- 0.02 | greedy on exact phase sums 1.000; ANOVA-template 1.000; Hebbian, no ReLU, 0.963 | 0.963 | 0.963 ✓ |
| 800 / 400 | 10% flips | 0.93 +/- 0.03 | **T-mc 0.933** | 0.921 (1 draw) | **0.933** (20 draws) ✓ |
| 800 / 800 | clean / 10% | >= 0.99 | T-mc 0.991 | 1.000 / 0.996 | 1.000 / 0.987 ✓ |
| 400 / 400 | clean / 10% | >= 0.98 | T-mc 0.956 | 0.979 / 0.954 | 0.979 / 0.958 ✓ (borderline) |
| 400 / 800 | clean / 10% | (>= 0.99) | T-mc 0.998 | 1.000 / 1.000 | 1.000 / 0.999 ✓ |
| 800 / 400 | pinv (alpha = 0), clean | <= 0.8 | | 0.704 | 0.704 ✓ |

- **Modules 0 and 1** land at >= 0.992 everywhere. Every failure is in module 2, as predicted.
- **Analytic contrasts (I.2)–(I.3)** against the exact phase sums: 1.06 / 1.01 / 0.59 against
  0.99 / 0.91 / 0.54 at P = 800 (0.95 / 0.85 / 0.69 exact at P = 400). The mean is 7–11% high,
  and the per-query correlation is only 0.2–0.56: the ridge's bulk fluctuations are not in (I.1).
- **Correction to I.2.** With the *exact* phase sums, the margin is always positive: greedy lands
  1.000 in all 12 condition-seed cells. The additive (main-effect) template leak causes no failures
  either. The failures come from the **non-additive part of the Hebbian templates**, acting on the
  coefficient mass concentrated in the 2-factor cells (a, ·, c) and (·, b, c). At P = Nh = 400 the
  ReLU adds more (Hebbian 0.992, snap 0.979). The failing queries are not the smallest-margin ones
  (margin percentile 0.42 at 800/400). Doubling Nh removes the failures.
- So the construction condition is: **phase-sum contrast > interaction crosstalk of the grid
  read-out, plus cue noise.** The contrast is fixed by the counts (I.1). The crosstalk shrinks
  with Nh. The cue noise is level T, which predicts every flipped cell within 0.004 of the mean
  of 20 real flip draws.
- The pinv collapse at P/Ns = 0.8 is the cue-side peak: module 2 falls from 0.96 to 0.71.

#### B. Read-out interpolation peak (`i2_readout.py`)

Overlap of the read-out at the landed address with the composite (Nh = 400, except the 1.00 row):

| P/Nh | measured, pinv | Omega(0.52 L_meas) | Omega(0.52 L_iso) | predicted range | stored read-out, pinv | ridge 0.01: landed / stored | ridge 0.1: landed / stored |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.488 | 0.486 | 0.663 | 0.2–0.5 ✓ | 1.000 | 0.496 / 1.000 | 0.543 / 1.000 |
| 0.50 | 0.393 | 0.367 | 0.496 | 0.25–0.5 ✓ | 1.000 | 0.456 / 1.000 | 0.594 / 0.995 |
| 0.75 | 0.266 | 0.238 | 0.333 | 0.15–0.35 ✓ | 1.000 | 0.464 / 1.000 | 0.640 / 0.962 |
| 0.90 | 0.158 | 0.142 | 0.198 | | 1.000 | 0.474 / 0.997 | 0.655 / 0.935 |
| **1.00** | **0.023** (Nh 800: 0.011) | 0.013 | 0 | **< 0.05 ✓** | **1.000** | **0.481 / 0.993** | 0.666 / 0.922 |
| 1.10 | 0.159 | 0.137 | 0.188 | | 1.000 | 0.499 / 0.985 | 0.683 / 0.909 |
| 1.25 | 0.245 | 0.214 | 0.288 | 0.3 ✗ (-0.05) | 0.995 | 0.506 / 0.971 | 0.687 / 0.890 |
| 1.50 | 0.329 | 0.309 | 0.385 | 0.4 ✗ (-0.07) | 0.967 | 0.531 / 0.939 | 0.702 / 0.857 |
| 2.00 | 0.449 | 0.426 | 0.493 | 0.49 ✓ (-0.04) | 0.902 | 0.577 / 0.884 | 0.724 / 0.816 |
| 2.25 | 0.491 | 0.470 | 0.530 | 0.52 ✓ (-0.03) | 0.878 | 0.596 / 0.863 | 0.727 / 0.801 |

- **(a) is confirmed exactly:** the stored read-out is 1.000 for every P <= Nh, including P = Nh,
  while construction collapses.
- The measured leverage at an empty address is about 2x the isotropic L. With the measured L,
  (d) tracks the whole U-curve within 0.03 (biased low by about 0.02). With L_iso it overshoots
  by 0.04–0.17.
- **(c) is confirmed:** lam = 0.01 x the mean eigenvalue lifts P = Nh from 0.02 to 0.48–0.50, at a
  stored cost of 0.007 (Nh 400) or 0.014 (Nh 800; predicted <= 0.01, ✗ marginally).
- lam = 0.1 makes composition monotone in P (0.54 -> 0.73), at a stored cost of up to 0.12. The
  read-out ridge is a gist-for-episode dial.

#### C. The frontier, the lure lemma, and I.5 (`i3_frontier.py`, `i3b_matched.py`)

- **Lemma I.4:** gain gamma' measured 0.669 aligned (predicted 0.693) and 0.607 random. The
  residual variance is 1.00 aligned and 0.98 random of the predicted sigma_e^2 trace.
- **The frontier**, P = 800, 10% flips unless "clean". Entries are measured / T-mc.

| decoder | aligned: noisy recall | aligned: construct, clean | aligned: construct, 10% | aligned: false a-b (lure) | aligned: false a-b-C | random: noisy recall | random: construct | random: false a-b |
|---|---|---|---|---|---|---|---|---|
| snap (paper) | 0.864 / 0.842 | 0.963 | 0.954 / 0.926 | 0.998 / 0.999 | 0.904 | 0.469 / 0.488 | 0.000 | 0.006 |
| D_0: nearest of all tuples | 0.987 / 0.991 | 1.000 | 1.000 / 0.998 | 1.000 / 1.000 | 0.994 | 0.963 / 0.958 | 0.000 | 0.006 |
| lam 0.075 | 0.987 | 1.000 | 0.996 / 0.996 | 0.990 / 0.985 | 0.983 | 0.990 | 0.000 | 0.006 |
| lam 0.10 | 0.987 | 0.996 | 0.963 / 0.968 | 0.925 / 0.902 | 0.921 | 0.990 | 0.000 | 0.006 |
| lam 0.15 | 0.987 | 0.708 | 0.513 / 0.484 | 0.312 / 0.313 | 0.312 | 0.990 | 0.000 | 0.006 |
| lam 0.20 | 0.987 | 0.050 | 0.012 / 0.021 | 0.015 / 0.006 | 0.015 | 0.990 | 0.000 | 0.006 |
| lam >= 0.3 = joint | 0.987 / 0.991 | 0.000 | 0.000 | 0.000 | 0.000 | 0.990 / 0.984 | 0.000 | 0.006 |

**Against the predictions (C):**

| quantity | predicted | measured |
|---|---|---|
| price of admitting never-stored states | ~0 aligned, <= 0.02 random | **0.000** aligned ✓; **0.027** random (✗ marginal) |
| price of per-module Hebbian decoding | 0.12 aligned, 0.49 random | **0.123** ✓, **0.494** ✓ |
| construction | ~1 at lam = 0 | 1.000 ✓ |
| cosine margin lam* where construction stops | 0.05–0.15 | 0.15–0.2 ✗ |
| false recall | follows construction, left-shifted and softer | ✓ at every lam |
| random placement: construction and false recall | ~0 | 0.000 and 0.006 ✓ |

T-mc tracks every aligned curve within 0.03.

**The direct, model-free test of I.5** (`i3b`). A factor cue at the lure's SNR, 18.7% flips,
against lures with 10% flips; 5 draws x 3 seeds.

| decoder | false recall (lure) | construction (matched cue) |
|---|---|---|
| lam 0 | 1.000 | 1.000 |
| lam 0.075 | 0.983 | 0.959 |
| lam 0.1 | 0.895 | 0.825 |
| lam 0.125 | 0.658 | 0.550 |
| lam 0.15 | 0.313 | 0.244 |
| lam 0.2 | 0.009 | 0.006 |
| snap | 1.000 | 0.998 |

The two curves agree within 0.11 at every lam, and exactly at the ends. A lure is falsely recalled
slightly *more* than its SNR-matched factor cue constructs. So the Gaussian-noise lemma is
conservative.

**Qualification of the corollary.** A bias can separate a *clean* factor cue from a lure (at lam
0.15: construction 0.71 against false recall 0.31). It does so only by exploiting the lure's
2.7x lower SNR, and it then fails a 10%-flipped factor cue as often (0.51). So the separation that
any lam can buy is bounded by the SNR gap between imagining and misremembering.

#### D. Recognition d′ (`i4_dprime.py`, the critic's C8 setting)

Entries are d′, measured per seed / T-mc per seed.

| placement | decoder | signal | studied vs recombination | studied vs unrelated | rho |
|---|---|---|---|---|---|
| random | snap | F1 cos(h0, h_out) | 2.15 2.20 1.81 / 2.43 1.97 1.81 | 2.42 2.34 2.23 / 2.68 2.27 2.34 | 0.46 |
| random | snap | F2 cue–read-out | 1.51 1.59 1.42 / 1.72 1.47 1.39 | 1.87 1.94 1.78 / 2.14 1.81 1.77 | 0.46 |
| random | joint | F1 | 4.17 3.96 4.03 / 4.38 3.70 3.63 | 4.39 4.08 4.31 / 4.65 3.88 4.23 | 0.99 |
| random | joint | F2 | 6.31 5.72 6.09 / 6.78 5.70 5.93 | 9.92 8.12 8.66 / 10.2 8.04 8.41 | 0.99 |
| aligned | snap | F1 | **2.50 1.76 2.97** / 2.50 1.89 2.76 | **8.04 6.26 8.42** / 7.97 6.37 8.34 | 0.91 |
| aligned | snap | F2 | 5.36 3.37 7.08 / 4.92 3.48 6.27 | 9.53 5.92 12.9 / 8.81 6.16 11.6 | 0.91 |
| aligned | D_0 (all tuples) | F1 | 3.81 3.74 3.83 / 3.91 3.76 3.64 | 10.2 9.99 9.38 / 10.4 9.78 9.37 | 1.00 |
| aligned | D_0 (all tuples) | F2 | **12.3 11.0 12.7** / 11.6 11.1 11.7 | 22.9 21.9 23.1 / 22.7 22.9 21.9 | 1.00 |
| aligned | joint | F1 | 8.08 8.32 8.13 / 8.59 8.28 7.33 | 10.2 10.4 9.50 / 10.4 9.91 9.38 | 1.00 |

- My replication reproduces the critic's numbers exactly (bold). T-mc predicts every cell within
  about 0.3 on F1, and within about 1 on the large F2 values.

**Against the predictions (I.7):**

| prediction | outcome |
|---|---|
| random: related ~ unrelated, 1.4–2.5 | ✓ |
| aligned: related 2–3, unrelated >= 5 | ✓ |
| F2 separates related lures better than F1 | ✓ |
| the joint decode does not raise related d′ | **failed.** It raises it to 8.1–8.3 (F1): refusing to construct also makes the lure look unfamiliar, since h0 is centred on an empty address that the stored 2-factor neighbour matches poorly (cos 0.70, against 0.93 for studied items) |
| the cap sqrt(2 rho/(1 - rho)) | holds where its premise does: aligned snap, mu_recomb 0.818 >= mu_err 0.805, cap 4.41 >= d′ 1.8–3.0. Under random placement the lures are *less* familiar than failed recalls (0.61 against 0.68), so d′ exceeds the "cap" of 1.31 |

**The important new number.** With the constructive decoder D_0, recognition by the cue–read-out
match F2 separates recombination lures at **d′ 11–12.7**, while the lures are still constructed
(false conjunction 1.00). By I.5, a decoder that sees only h0 cannot tell imagining from
remembering. A check that compares the cue's *detail* with the read-out can, and nearly
perfectly. That is the "way out" of I.5, measured.

### 0.10 The propositions, in one place

| # | statement | status | check |
|---|---|---|---|
| I.1 | Phase sums of a factor cue: C^(m) = [N theta]_m, theta = (kappa_f/kappa_1)(N + eps I)^-1 e | derived | mean within 7–11%; per-query variation not captured |
| I.2 | Construction <=> phase-sum contrast (module 2 diluted by omega(mu)) > non-additive template crosstalk + cue noise | derived, corrected | landing 0.963 / 0.933 (flips) / 1.000 (Nh 800) / 0.704 (pinv): all predicted within the stated bands; T-mc within 0.004 |
| I.3 | Stored read-out exact for P <= Nh; off-sample variance diverges at P = Nh; a ridge bounds it | proved (a)–(c); derived (d) | collapse 0.02 at P = Nh with stored 1.000; curve within 0.03 using measured L; ridge 0.01 restores 0.48 at a cost of 0.007 |
| I.4 | A recombination lure = a factor cue at gain 0.69 and noise 0.74 (SNR 2.7x lower) | derived | gain 0.669; residual variance ratio 1.00 |
| I.5 | For every decoder scoring h0 homogeneously plus a fixed state prior, P(false recall) = P(construction at the lure's noise) | **proved given I.4** | at matched SNR the two curves agree within 0.11 at every lam; the SNR gap is the only separation any lam can buy |
| I.6 | joint - snap = (price of never-stored states) + (price of per-module Hebbian decoding) | derived; measured | 0.000 + 0.123 aligned; 0.027 + 0.494 random |
| I.7 | Recognition d′ is a mixture over recall success (I.8); constructive decoders make recombination lures familiar | identity proved; moments derived | T-mc within 0.3 of every F1 d′; F2 with D_0 gives d′ 11–12.7 against recombination lures |

**What this does to the thesis.**
1. **The recall cost of imagination is about zero.** Take aligned placement and a cosine
   nearest-code decoder over the full attractor set. It recalls noisy cues exactly as well as the
   joint decode (0.987 against 0.987), constructs 1.000 and falsely recalls recombinations 1.000.
   The THESIS's "0.12 cost that buys construction" is the price of the paper's per-module Hebbian
   snap, not of imagination. Under random placement the 0.49 "cost" is 0.494 decoder and 0.027
   attractor set.
2. **The price of imagination is paid in false memory,** and by I.5 it is paid at par. At the
   lure's SNR, false recall equals construction for every decoder that looks only at the place
   activity.
3. **Metamemory is the escape.** A cue–read-out match (F2) rejects constructed recombinations at
   d′ 11–12 while the memory still constructs them. Imagining is cheap for recall, costly for
   source memory, and nearly free once a detail check is added.
4. **The read-out has its own interpolation peak at P = Nh,** invisible to stored-item metrics, and
   removed by a ridge of 1% of the mean eigenvalue.

This is textbook in its parts:
- ridgeless double descent: Hastie et al. 2022;
- mixture d′: signal detection;
- the ReLU homogeneity argument is elementary;
- the conjunction-error phenomenon: Reinitz et al. 1992.

What is not textbook is that one scaffold property, all tuples being fixed points, ties
construction, conjunction errors and the loss of wrong-address detection into one measurable
quantity, and that its recall price separates cleanly from the decoder's.

### 0.11 Items 1, 2, 4, 5 of the first brief, consolidated (details in Parts A–C)

1. **Recall computes where to store.**
   - ΔJ_m(k) = ||Q_m^T c - e_k||^2 / r, exact.
   - The snap agrees with the greedy choice 78–86% because of label/address mismatch and template
     crosstalk. A stored-address linear read-out is exact.
   - The 1/r gate is exact but weak.
   - Known maths: RLS, online kernel k-means.
2. **Bias = gradient.**
   - Proved. Replay is kernel k-means / DKM on the ridge smoother (Ye, Zhao & Wu 2007).
   - It stalls on labels. On realized addresses (2.7) it reaches oracle-level recall (0.856 against
     0.858) without discovering factors.
3. **Quantitative theory (item 4).**
   - Level T (K through the templates) predicts address recovery at MAE 0.010 over 180 conditions,
     and within-type seed deviations at r = 0.95.
   - The error law alone does neither.
   - It now also predicts construction (±0.004), the frontier (±0.03) and d′ (±0.3).
4. **Redundant modules (item 5).**
   - The correlation law (5.1) holds within 0.022.
   - The 4th-module benefit to the joint decode survives cosine decoding only at 20% flips (+0.195).
     At 10% it is at ceiling.
   - The snap gets worse.

---------------------------------------------------------------------------------------------------

## Part A. Mathematics

### 1. Recall computes where to store

**Block inverse [proved].** Add pattern s. Write b = S^T s, the recall coefficients c = K b, and
r = s^T s + alpha - b^T K b. The Schur complement gives

    (1.1)   K+ = [[K + c c^T / r,  -c / r],
                  [   -c^T / r,     1 / r]].

Woodbury gives I - S K S^T = alpha (S S^T + alpha I)^-1. Hence

    (1.2)   r = alpha (1 + s^T (S S^T + alpha I)^-1 s),   so alpha <= r <= alpha + |s|^2,   and 1/r = K+_ss.

So r is the residual of s's own recall. Equivalently, it is the posterior variance factor of s's
coefficient once s is stored.

**The error-law increment [proved].** Suppose s goes to phase k of module m. The old indicators gain
a zero entry, q_j -> [q_j; 0], except that q_k -> [q_k; 1]. Write C_j = (Q_m^T c)_j = sum over i in j
of c_i, the summed recall coefficient on phase j. Then

    j != k:   [q_j;0]^T K+ [q_j;0] = q_j^T K q_j + C_j^2 / r
    j  = k:   [q_k;1]^T K+ [q_k;1] = q_k^T K q_k + (C_k^2 - 2 C_k + 1) / r

    (1.3)   Delta J_m(k) = ( sum_j C_j^2  +  1 - 2 C_k ) / r  =  || Q_m^T c - e_k ||^2 / r .

The old-block change sum_j C_j^2 / r = ||Q_m^T c||^2 / r does not depend on k. It is the growth of
the old items' errors: storing s makes them all more alike. The part that depends on k is -2 C_k / r.
Read the right-hand side as a squared error: *the cost of storing s at phase k is the squared
distance between where s's own recall puts it, projected onto module m's phases, and the one-hot at
k, divided by s's prediction error.* The address is a product, J is a sum over modules, and every
phase tuple is an address (CRT). The greedy online optimum is therefore separable across modules:

    (1.4)   k*_m = argmax_k C_k   (subject to capacity, and to the tuple being free)

    (1.5)   regret(k) = Delta J_m(k) - Delta J_m(k*) = 2 (C_k* - C_k) / r ,
            benefit over a uniformly random phase = 2 (C_k* - mean_k C_k) / r .

**Prediction error gates integration [proved, with the exact sense stated].** Split s = u + v with
v orthogonal to range(S). Then S^T v = 0, so c(s) = c(u), and r(s) = r(u) + |v|^2. Hence

    (1.6)   Delta J_m(k) = || Q_m^T c(u) - e_k ||^2 / ( r(u) + |v|^2 ) .

At a fixed explained part u, the whole k-dependence (regret and benefit) scales *exactly* as 1/r:
novelty orthogonal to memory enters only the denominator, and linearly. In general the gain is
2 x (the spread of familiarity C) / r, not 1/r. The novelty lying inside range(S) also changes C,
by adding a random, sign-indefinite part to c. The gate has a floor, r >= alpha, which grows with P.
At the headline, alpha = 450 and r ranges over [450, 1450].

**When the snap equals the greedy choice [derived].** The recall of s is h0 = ReLU(W_hs s) =
ReLU(H_a c), so the grid input is x_mk = t_mk^T h0.

*Without the ReLU*, x_m = T_m c, with T_m[k, i] = t_mk^T h_{a_i}. Decompose

    (1.7)   T_m[k, i] = beta_m ( [phi_m(a_i) = k] - 1/k_m ) + gamma_i + E_m[k, i] ,

where gamma_i does not depend on k, so it adds sum_i gamma_i c_i to every phase and cancels in the
argmax. Then

    (1.8)   x_mk = const + beta_m ( C_k - mean C ) + (E_m c)_k ,

and snap = greedy exactly when argmax_k [C_k + (E_m c)_k / beta_m] = argmax_k C_k. A sufficient
condition is margin = C_(1) - C_(2) > 2 ||E_m c||_inf / beta_m. E_m holds the following:
- (i) *phase gains*: the on-phase overlap varies with k (template norms);
- (ii) off-diagonal phase-phase terms;
- (iii) *cross-module* terms. The ANOVA main effects u_mk of the place code, taken as vectors in
  R^Nh, are not orthogonal across modules (u_0k^T u_1j = O(sqrt Nh)). So module m's input picks up
  C^{m'} from the other modules' phase sums;
- (iv) interaction terms of the thresholded code.

The ReLU and the 0.5 threshold enter E only through the geometry of h (sparsity sets beta and the
size of (iii) and (iv)).

*With the ReLU*, write ReLU(u) = (u + |u|)/2. Then

    (1.9)   x_m = (1/2) T_m c + (1/2) W_gh,m |H_a c| .

The second term does not depend on the sign of c. It rewards phases carrying large coefficients of
*either* sign. A new item's c is a signed ridge combination with a lot of negative mass, so the ReLU
biases the snap towards phases where s is strongly *anti*-explained.

*Corrected snaps.*
- Dropping the ReLU leaves (1.8), and its error is E_m alone.
- Replacing W_gh by a linear read-out D with D H_a = G_a, the stored items' grid codes, gives
  x = D H_a c = G_a c = (Q_m^T c)_m. That is **exactly** the greedy choice, with no margin
  condition. D exists whenever n <= Nh, because H_a then has full column rank: D = G_a H_a^+. For
  n > Nh it is least squares, and exactness is lost in proportion to (n - Nh)/n.
- A global decoder D = G H^+ over all 3,600 addresses is exact only if the one-hot grid code lies
  in the row space of the place code. Measured on the default scaffold, it does not:
  ||G - G H^+ H|| / ||G - mean|| = 0.14 (structure.py).

### 2. The ridge's bias is the gradient

**[proved]** For a stored item, s = S e_i, so c = K S^T S e_i = K (K^-1 - alpha I) e_i:

    (2.1)   c(recall of i) = e_i - alpha K e_i .

The self part is 1 - alpha K_ii. The bias onto item j != i is b_ij = -alpha K_ij.

**Single moves are exact coordinate descent [proved].** Move i from phase a to phase b of module m:

    (2.2)   Delta J = 2 [ sum_{j in b} K_ij - sum_{j in a, j != i} K_ij ] = (2/alpha) (B_a - B_b),
            B_k = sum_{j in k, j != i} b_ij .

So "move i to the phase with the largest summed bias" is the exact minimiser of J over i's label in
module m. Each accepted move lowers J strictly, and the state space is finite, so replay converges.

**Replay is re-encoding with the item's own trace removed [proved].** Apply the block inverse to
item i against the rest. The leave-one-out coefficients are c^(-i)_j = -K_ij / K_ii, and
r_i = 1 / K_ii. So

    (2.3)   B_k = alpha K_ii C^(-i)_k ,   and (2.2) = (1.5) with r = 1/K_ii .

The replay rule is the online rule (1.4) applied to item i arriving *after* everything else. Online
encoding and replay are one operation, done at different times.

**Fixed points [proved].** A single-move fixed point (1-opt) satisfies
sum_{j in l_i \ i} K_ij <= sum_{j in k} K_ij for every i, m and admissible k. For a swap of
i (in a) with j (in b):

    (2.4)   Delta_swap(i, j) = Delta_move(i: a->b) + Delta_move(j: b->a) - 4 K_ij .

At a 1-opt point, a swap improves only if K_ij > 0, meaning i and j have *negative* partial
correlation (they are dissimilar), or if a single move is blocked by capacity. Neither fixed-point
set contains the other: swaps cannot resize groups, and single moves cannot do the two-item exchange
of (2.4). Under tight capacity (<= 1.1 P/k in replay.py), single moves are nearly frozen, so the
swap neighbourhood is the larger one.

**Why replay stalls in mixed partitions [derived, block model].** Take K_ij ~ -kappa_A [a_i = a_j]
- kappa_B [b_i = b_j] - kappa_C [c_i = c_j] for i != j, plus Wishart fluctuations. Let n_{k,v} be
the number of items with factor value v in group k. Then

    (2.5)   J_m ~ const - sum_f kappa_f sum_{k,v} n_{k,v}^2 ,
            affinity(i -> k) = kappa_A n_{k,a_i} + kappa_B n_{k,b_i} + kappa_C n_{k,c_i} .

Any partition in which each item's group holds more of its factor-mates than any other group does
is 1-opt. That includes partitions aligned to the wrong factor, and mixtures. Leaving such a state
means relabelling O(P/k) items together, so the barrier is extensive. There is a sharper cause, and
it is specific to the lead's online rule. **The greedy (1.4) is the same in every module**: C is the
same vector, all labels start at phase 0, and ties go to the lowest index. So the modules make
*identical* choices, up to capacity and to the number of groups, until a cap binds. Module 0 and
module 1 are clones until some group reaches 50 items (module 1's cap). Each module then encodes the
*total* similarity, which is dominated by the most frequently shared factor (C: 5 values, so two
items share C with probability 1/5, against 1/9 for A and 1/16 for B). Capacity is the only thing
that breaks the symmetry between modules. J is separable, so nothing in the objective asks module
0 to carry what module 2 does not.

**Which partition is the global minimum [derived, spike model].** Factor structure puts spikes in
S^T S far above the Marchenko-Pastur bulk (up to about 3,600 at P/Ns = 0.8):
- lambda_A ~ Ns x 0.161 x P/9, about 14,000, with 8 contrasts;
- lambda_B ~ 8,000, with 15 contrasts;
- lambda_C ~ 26,000, with 4 contrasts.

Here 0.161 = (2/pi) arcsin(1/4) is the bit correlation from one shared factor. With
Delta_s = 1/(lambda_bulk + alpha) - 1/(lambda_s + alpha), roughly equal for all spikes,

    (2.6)   J_m ~ P/(lambda_bulk + alpha) - sum_spikes Delta_s || Q_m^T v_s ||^2 .

- Module 0, A-aligned (9 groups): 8 x 89 = 712.
- Module 0, C-split into 9 groups: about 4 x 81 = 324 from C, plus a partial A term.
- Module 1: B-aligned gives 750, against about 360 for A-split.
- Module 2: C-aligned gives 640.

So the aligned partition is the global minimum of each module by a factor of about 2 in the spike
term. But the spikes sit at nearly equal Delta, so the high-temperature instability of all three
factor modes occurs at almost the same T.

### 3. Composition at an empty address

**The read-out is a linear smoother [proved].** Here P > Nh and H_a has full row rank, so
H_a^+ = H_a^T (H_a H_a^T)^-1 and

    (3.1)   W_sh h_new = S w,   w = H_a^T (H_a H_a^T)^-1 h_new   (the min-norm solution of H_a w = h_new).

The read-out is a weighted sum of stored contents. Equivalently, it is the OLS regression of
content on the 400 place-cell features, evaluated at the new address.

**Content model.** For each bit n, s = sign(X + z) with X = A[a] + B[b] + C[c] (var 3) and
z ~ N(0,1). Then s = g + eps, where g = erf(X / sqrt 2) = E[s | X] and Var(eps | X) = 1 - g^2. By
Stein's lemma, g's linear part is kappa_1 X with kappa_1 = sqrt(2/pi) / sqrt(1 + 3) = 0.399. The
variances are:

- E[g^2] = (2/pi) arcsin(3/4) = 0.540;
- the linear part, 3 kappa_1^2 = 0.477;
- the nonlinear remainder, 0.063;
- the noise, E[eps^2] = 0.460.

A factor's main effect is E[s | A] = erf(A / sqrt 6), with the same slope kappa_1 and a negligible
nonlinearity (0.002).

**[derived]** Under aligned placement, suppose the place-code span contains the main-effect
functions of the three phases, so that OLS is unbiased for the additive part m(a,b,c) =
sum_f erf(X_f / sqrt 6). Then, at an address never stored,

    (3.2)   yhat = m(a, b, c) + eta,   eta ~ N(0, sigma_res^2 L),   L = h_new^T (H_a H_a^T)^-1 h_new,
            sigma_res^2 ~ 1 - 3 kappa_1^2 ~ 0.52 ,

and the overlap with the composite sign(X) is

    (3.3)   Omega(v) = E[ sign(m + eta) sign(X) ],   eta ~ N(0, v) .

For isotropic features, L ~ Nh / (P - Nh - 1), which is about 1.0 at P = 800 (Hastie et al. 2022,
the variance term of ridgeless least squares). The same formula covers every linear smoother:

| read-out | v | Omega (predicted) |
|---|---|---|
| grid, aligned, P = 800 | 0.52 x L, L ~ 1 | **~0.49** |
| explicit additive model, group means over n_A ~ 89, n_B ~ 50, n_C ~ 160 items | sum_f 1/n_f = 0.038 | **~0.81** (the upper bound for a linear read-out) |
| kNN: one stored item sharing 2 of 3 factors | n/a: corr(X, X - C + C' + z) = 2/sqrt 12, (2/pi) arcsin 0.577 | **0.39** |
| a stored item sharing all 3 (not available at an empty address) | (2/pi) arcsin(3/sqrt 12) | 0.67 |
| sign of the mean stored content | corr = sqrt(0.373/3) = 0.353 | **0.23** (the chance floor for *any* composite) |
| random placement, flat control | the empty address's code carries no factor information: yhat ~ mean + noise | **<= 0.23** |

**Double descent in P [derived].** L diverges at P = Nh: the grid's composition is worst there, and
improves on both sides. By the isotropic formulas, L ~ 2 at P = 600 and 0.8 at P = 900. Below Nh
the min-norm interpolant also shrinks the signal, by about P/Nh. So Omega(P) is predicted to be
U-shaped with its minimum at P ~ 400: ~0.49 at P = 800 and ~0.52 at P = 900.

### 4. Recall under placement: a Gaussian theory

**Exact first two moments [proved].** A flipped cue is s~ = s_i o (1 - 2 flips), so E[s~] = a s_i and
Cov(s~) = sigma^2 I. Then c~ = K S^T s~ has

    (4.1)   E c~ = a (e_i - alpha K e_i),   Cov c~ = sigma^2 K S^T S K = sigma^2 (K - alpha K^2).

Averaged over items, the bias outer product alpha^2 K e_i e_i^T K / a^2 gives alpha^2 K^2 / P. At
alpha = sigma^2 P / a^2 the total error covariance of c~/a is

    (4.2)   E_i Cov_err(c~/a) = (sigma^2 / a^2) K      (the error law's origin, the Bishop identity) .

**Level E, the error law alone [derived]** (ideal templates, no ReLU): x_m ∝ Q_m^T c~, Gaussian with

    (4.3)   mean a Q_m^T (e_i - alpha K e_i),   covariance sigma^2 Q_m^T (K - alpha K^2) Q_m .

The module accuracy is P(argmax = phi_m(a_i)), averaged over items. The address accuracy is the
joint probability.

**Level T, through the templates and the ReLU [derived].** u = H_a c~ is Gaussian (CLT over Ns bits),
with mean a H_a (e_i - alpha K e_i) and covariance Sigma_u = sigma^2 H_a (K - alpha K^2) H_a^T. For
h0 = ReLU(u), take the exact per-cell mean and variance, and the off-diagonal covariance to first
order in the Hermite expansion, D Sigma_u D with D = diag Phi(mu_n / s_n). Then x = W_gh h0 is
Gaussian with mean W_gh E h0 and covariance W_gh Cov(h0) W_gh^T. Sample it jointly over all 50 grid
cells. That gives each module's accuracy, and the address accuracy with cross-module correlations
included.

**Closed form, for intuition [derived].** With iid phase noise of variance v_m ~ sigma^2 (1 - rho)
L_m, where L_m is the per-phase error law, and a true-phase margin d:

    (4.4)   acc_m ~ integral phi(t) Phi(t + d / sqrt v_m)^(k_m - 1) dt .

With random placement L_m ∝ P/k_m, the occupancy of each phase. That is why module 0, with 9
phases, is worst under random placement, and why aligned placement, which makes q^T K q small by
cancellation, fixes modules 0 and 1 completely.

### 5. Redundant modules

**Place-code correlation [derived].** Pre-activations z = sum_m w_{m,phi_m} are Gaussian over the
weights, with variance M p (p = 0.67 is the effective connectivity) and correlation q/M between two
addresses sharing q phases. Let t = theta / sqrt(M p). Expanding ReLU(sqrt(Mp) x - theta) in Hermite
polynomials gives a_1 = Phi(-t), and a_n^2 ∝ He_{n-2}(t)^2 phi(t)^2 / n! for n >= 2, so

    (5.1)   rho_h(q/M) = [ Phi(-t)^2 rho + phi(t)^2 sum_{n>=2} He_{n-2}(t)^2 rho^n / n! ] / Var(ReLU(x - t)),   rho = q/M .

rho_h is convex, with rho_h(rho) < rho on (0,1). About 61% of the variance is linear (a_1^2/Var).

**Noise pooling [derived].** Decode jointly over stored addresses by the score h_j^T H_a c~. The
crosstalk onto the true item's score from stored items sharing q phases is

    (5.2)   Pool(M) = sum_{q >= 1} N_q rho_h(q/M)^2 ,

where N_q is the number of stored items sharing exactly q phases. Under random placement,
N_1 ~ P sum_m 1/k_m. For small rho, rho_h(1/M)^2 ~ (0.61/M)^2, so Pool ~ P sum_m 1/k_m x 0.37/M^2.
It falls with M. The flat code has only the random O(P/Nh) crosstalk. Under the per-module snap, J
adds one term per module, J = sum_m J_m. Each extra module adds a factor acc_m < 1 to the address
accuracy, while leaving the others' J_m unchanged.

---------------------------------------------------------------------------------------------------

## Part B. Predictions, written before any check

P1a. (1.3) holds to machine precision against brute-force J, for every k and every module.
P1b. (1.6): at fixed u, the regret times r(u) + |v|^2 is constant in |v|, to machine precision.
P1c. The agreement between snap and greedy is limited by the ReLU (1.9) and by the phase gains (i).
Expected:
- no-ReLU with Hebbian templates: about 90%;
- no-ReLU with the stored-address linear decoder D = G_a H_a^+: 100% for n <= Nh, then
  degrading;
- the lead's snap: 78-86%.

A margin model (the agreement is P(argmax(C + xi) = argmax C), with xi set by the template
residual) should reproduce the measured agreement within about 5 points.

P1d. Online exact greedy: modules 0 and 1 are identical up to relabelling (NMI = 1) until the first
module-1 group reaches its cap. Module 0 ends up aligned with C at least as much as with A:
NMI(m0, C) >= NMI(m0, A).

P2a. (2.1) and (2.2) to machine precision.

P2b. Escapes from the online state, 3 seeds, P = 800:

| escape | predicted outcome |
|---|---|
| random-order deterministic replay (the lead's) | stalls: m0-A NMI < 0.3 |
| most-mis-placed-first replay | same fixed points, so within +/-0.03 of random-order replay; no escape |
| co-replay of pairs (swaps) from the online state | small gain (<= +0.05 recall); m0-A NMI stays < 0.3 |
| novelty-gated seeding (seed empty phases when r is high) | in this content every item is equally novel, so the gate becomes a clock: small change (< +0.05), no escape |
| annealed Gibbs replay on exp(-J/T), T from about the typical |Delta J| down to 0 | **escapes**: m0-A NMI > 0.8 on >= 2 of 3 seeds, recall within 0.03 of batch swaps or above |
| per-module independent noise at encoding (Gibbs online), then annealing | at least as good as annealing alone; the noise breaks the clone symmetry |

P3. At P = 800 and aligned placement, at unstored (a, b, c) cells:
- grid composition Omega ~ 0.49 (+/- 0.07);
- additive model ~ 0.81;
- a 2-factor kNN item 0.39;
- mean-content floor 0.23;
- random placement and the flat control <= 0.25.

The P-sweep is U-shaped with its minimum near P = 400.

P4. Level T should get the ordering of placements right, and the headline magnitudes within about
0.05. Level E (error law alone) should get the ordering but overestimate the accuracy of random and
sequential placement, because it has no template or ReLU crosstalk.

P5. (5.1) reproduces 0 / 0.26 / 0.58 / 1 within 0.03. For a 4th module (period 7) at Nh = 400:
- the joint stored-address decode improves under noise, because Pool falls from about 15 to
  about 8. The flat code's 0.999 is the ceiling, and the grid's random-placement stored decode at
  M = 3 is 0.83;
- the per-module snap's address accuracy does not improve: one more factor acc < 1.

### Addendum to Part B (written after the diagnosis in t2b_diag.py, before the address-level run)

The t2 and t2b runs on seed 0 showed that J on *labels* is not the law recall feels. Open addressing
overrides the labels whenever a (p0, p1) cell is crowded, and it is crowded exactly when two modules
encode the same factor. Worst-first and swaps put modules 0 and 1 both on B (NMI(m0, m1) 0.72-0.74).
That gives 38% overflow in module 1, 43-51% in module 2, and recall 0.68 despite batch-level
label-J. J on realized phases, with module 2 at 25 phases, tracks recall across arms (Spearman
-0.88). The correct objective is (1.3) on *realized* phases. The address constraint couples the
modules: a tuple holds one item, so redundancy between modules is penalised automatically. The
exact moves are:

    (2.7)   online:  a* = argmax over free addresses a of  sum_m C^(m)_{phi_m(a)}  (phase-level sums, module 2 at 25 phases)
            replay:  item i -> argmin over free a of  sum_m Delta J_m(phi_m(a_i) -> phi_m(a)),  Delta J_m from (2.2)

P2c. Predictions for address-level descent (3 seeds):
- address-level replay beats label-level replay in recall by >= 0.05 and keeps NMI(m0, m1) < 0.1;
- annealed address-level replay reaches batch or better;
- because it also shapes module 2's 25 phases, which the oracle leaves random within C-groups
  (module 2 is the oracle's bottleneck at 0.88), it may exceed the oracle. This is *conjecture*: I
  give it about 1 in 3.

---------------------------------------------------------------------------------------------------

## Part C. Checks: what matched and what did not

All runs used seeds 0-2, P <= 900, Ns = 1,000 and VECLIB_MAXIMUM_THREADS = 2, with the bench's
content, scaffold, alpha, cues and placement RNGs. Nothing below was tuned to a measured recall.
Where a result is post hoc, it says so.

### C1. Recall computes where to store (`t1_online.py`, `t1b_margin.py`, `t1c_gate.py`)

| claim | status | result |
|---|---|---|
| (1.3) Delta J_m(k) = \|\|Q^T c - e_k\|\|^2 / r | **proved, matches** | max relative error against brute force 5e-12 (3 seeds, n = 50/200/500, every phase of every module). K grown by 800 block updates (1.1) matches the direct inverse to 9e-15 |
| (1.6) regret x r is constant at a fixed explained part | **proved, matches** | constant to 4 digits (e.g. 0.8590) while \|v\|^2 goes 0 -> 900 and r 619 -> 1519 |
| P1c: snap vs greedy is limited by the ReLU and the phase gains | **failed** | Dropping the ReLU changes agreement by <= 1 point; gain correction by <= 2. The causes are below |
| P1c: the stored-address linear decoder G_a H_a^+ with no ReLU is exact for n <= Nh | **proved, matches** | 1.000 in every module and seed for n <= 400; 0.96-0.99 for n > 400 |
| P1d: modules 0 and 1 are clones until a module-1 group hits its cap | **matches** | NMI(m0, m1) = 1 until n = 52 / 52 / 117 on the exact trajectory |
| P1d: NMI(m0, C) >= NMI(m0, A) | **failed / moot** | 2 of 3 seeds, and all NMIs <= 0.15: online greedy finds no factor at all |

**Why the snap agrees 78-86% (explained, not predicted a priori).** There are two separate causes.

1. *Labels are not addresses.* Open addressing overrides the label whenever a (p0, p1) cell is
   crowded. That happens to 16-22% of items in module 1 and 44-50% in module 2, averaged over the
   run. Against the greedy computed on *actual* phases, the lead's snap agrees
   0.867 / 0.792 / 0.749 (mean of 3 seeds). Against labels it agrees 0.867 / 0.775 / 0.747.
2. *Hebbian template crosstalk.* The template overlap T_m[k, a] has an exact ANOVA over the
   product of phases. Its main effects explain 95.8 / 94.4 / 92.7% of the template variance. The
   own-module deviation from beta (Q - 1/k) is 6.5-8.8% rms, and the leak from each other module
   is 6.3-9.1% rms. Built from the scaffold alone plus the recall's phase sums, the additive model
   x_m = sum_m' F^{mm'} C^(m') reproduces the Hebbian agreement:

| seed | ANOVA model, modules 0 / 1 / 2 | actual Hebbian, modules 0 / 1 / 2 |
|---|---|---|
| 0 | 0.859 / 0.830 / 0.801 | 0.874 / 0.817 / 0.768 |
| 1 | 0.879 / 0.812 / 0.777 | 0.877 / 0.809 / 0.706 |
| 2 | 0.855 / 0.768 / 0.826 | 0.865 / 0.757 / 0.793 |

   That is within 0.03, except module 2 (0.03-0.07), which carries the most interaction variance.
   With own-module templates alone the model gives 0.89 / 0.86 / 0.89. So the cross-module leak
   costs 3-9 points and the own-module non-uniformity about 10.

**Corrected snaps**, agreement with the address-greedy choice (mean of 3 seeds):

| read-out | module 0 | module 1 | module 2 |
|---|---|---|---|
| Hebbian, with ReLU (the lead's) | 0.867 | 0.792 | 0.749 |
| Hebbian, no ReLU | 0.872 | 0.794 | 0.756 |
| global decoder G H^+, no ReLU | 0.942 | 0.917 | 0.938 |
| stored-address decoder G_a H_a^+, with ReLU | 0.951 | 0.922 | 0.934 |
| **stored-address decoder, no ReLU** | **0.991** | **0.980** | **0.994** (1.000 while n <= Nh) |

The ReLU matters only once the templates are right (it costs about 5 points there). The exact
condition is (1.8) with E_m = the ANOVA terms: the snap equals the greedy iff the margin in
C^(m) exceeds the leak sum_m' F^{mm'} C^(m') / beta_m plus the interaction term.

**The gate [proved exact; empirically weak].** Benefit = 2 (C_max - mean C) / r. It is exactly 1/r
at a fixed explained part. But r is bounded: alpha <= r <= alpha + Ns, from 450 to 1,450 at the
headline. So 1/r has a dynamic range of at most 1 + a^2 Ns / (sigma^2 P) = 3.2 there, and over the
online run it only moves from 1,363 (the first 20 items) to 811 (the last 100). With LOO benefits
at the final state (`t1c_gate.py`):

- *factored content*: var(log r) is 0.000 of var(log benefit), which is 0.005;
- *mixed content* (half the items pure noise): var(log r) = 0.019 of var(log benefit) = 0.63. Novel
  items have 1.31x the r of familiar ones but 4.2x less benefit. The familiarity numerator carries
  about 70% of the variance directly, with corr(log r, log spread) = -0.92.

"Prediction error gates integration" is therefore right in direction. But the gating is done
almost entirely by the familiarity spread, not by the 1/r factor.

**The online greedy fills groups one at a time [derived post hoc, matches].** The mean recall
coefficient is positive, because factored items are positively correlated on average. So
C_k ~ n_k cbar + fluctuation, and the largest non-full group wins. Measured: module 0 chooses the
largest non-full group on 55-62% of arrivals, against 11% by chance. That, plus the clones, is why
online greedy ends near sequential placement (0.543, against sequential's 0.484 on these seeds and 0.455 over the bench's 20).
Pricing group size, score C_k - n_k (1^T c)/n, cuts this to 34%.

### C2. The ridge's bias is the gradient (`t2_replay.py`, `t2b_diag.py`, `t2c_address.py`, `t2d_fixedpoints.py`)

**Identities [proved, match].**
- (2.1): max error 3e-15.
- (2.2): relative error 3e-12.
- (2.3), the LOO form: 2e-15.
- (2.4): with the 1.1 P/k caps, every improving swap at a single-move fixed point has K_ij > 0 or
  a capacity-blocked single move (0 unexplained, 9 module-seeds). Without caps, the single-move
  fixed points had *no* improving swaps.

**The escapes** (recall at P/Ns 0.8, 10% flips, ridge; NMI of module 0 with factor A):

| method | seed 0 | seed 1 | seed 2 | mean | predicted | verdict |
|---|---|---|---|---|---|---|
| online greedy (labels) | 0.605 | 0.529 | 0.495 | 0.543 | | |
| + replay, 8 sweeps | 0.796 | 0.666 | 0.666 | 0.709 | stalls, m0-A < 0.3 | matches (m0-A 0.07/0.15/0.07) |
| + replay to convergence | 0.754 | 0.655 | 0.684 | 0.698 | | lower J, *worse* recall than 8 sweeps |
| + worst-first to convergence | 0.691 | 0.642 | 0.682 | 0.672 | within +/-0.03 of replay | ~matches (-0.026) |
| + pair swaps | 0.682 | 0.728 | 0.684 | 0.698 | <= +0.05, no escape | mixed: seed 1 found A (0.93) |
| + annealed Gibbs (label J) | 0.770 | 0.630 | 0.645 | 0.682 | escapes on >= 2/3 seeds | **failed**: m0 lands on B (0.23/0.83/0.83) |
| novelty-seeded online, + replay | 0.761 | 0.693 | 0.771 | 0.742 | small change | online +0.08, a little over the prediction; best label-level |
| centred online, + replay | 0.792 | 0.719 | 0.713 | 0.741 | (not predicted) | |
| Gibbs online, + anneal | 0.670 | 0.604 | 0.626 | 0.633 | >= annealing | failed |
| batch swaps, 20 starts | 0.801 | 0.709 | 0.812 | 0.774 | | |
| oracle | 0.879 | 0.836 | 0.859 | 0.858 | | |
| **ADDRESS online greedy (2.7)** | 0.769 | 0.715 | 0.665 | 0.716 | | +0.17 over label online |
| **ADDRESS online + replay (2.7)** | **0.880** | **0.877** | **0.811** | **0.856** | beats label replay by >= 0.05; NMI(m0, m1) < 0.1 | **matches**: +0.15; NMI(m0, m1) 0.09/0.07/0.05 |
| ADDRESS + annealed Gibbs | 0.864 | 0.785 | 0.844 | 0.831 | >= batch | matches (+0.06 over batch) |
| ADDRESS centred online + replay | 0.904 | 0.819 | 0.853 | 0.859 | | |
| oracle + ADDRESS replay | 0.895 | 0.868 | 0.899 | 0.887 | may exceed oracle (1 in 3) | **matches**: +0.03 over oracle, via module 2 |

**Why label-level replay stalls [derived post hoc, checked].** Replay descends J on *labels*. But
recall feels J on *realized* phases, and the two diverge when modules become redundant.
- Deeper label descent (worst-first, swaps) drove modules 0 and 1 both onto B: NMI(m0, m1) 0.72-0.74.
- The (p0, p1) cells then overflow: 38% of items are re-homed in module 1 and 43-51% in module 2.
  Realized J rises even as label J falls.
- Realized J orders the 8 arms by recall (Spearman -0.88, seed 0).

J is separable over modules. Only the address constraint (one item per tuple) couples them, and
label-level descent ignores it. Descent on the realized law, (2.7), respects it automatically. It
also shapes module 2's 25 phases, which labels leave random within C-groups.

**Factor discovery is not needed for recall.** Address-level replay matches the oracle's recall
(0.856 against 0.858) with NMI(m0, A) of only 0.20 / 0.33 / 0.11. It trades module 0 and 1
perfection for module 2 accuracy: [0.97, 0.97, 0.94] against the oracle's [1.00, 1.00, 0.88]. The
address accuracy is the product, so balancing errors across modules wins.

**Caveat on "by recall alone".** The moves need the phase sums of the coefficient vector c (online)
and of -alpha K e_i (replay). The Hebbian snap delivers those only about 80% of the time (C1). The
stored-address decoder delivers them exactly, but it is a pinv over the stored *place codes*. That
is not a Gram matrix of contents, but it is a global operation.

### C3. Composition (`t3_compose.py`, `t3b_ridge_readout.py`)

P = 800, 3 seeds x 150 empty (a, b, c) cells:

| read-out | measured | predicted |
|---|---|---|
| grid, aligned, empty address in a *random* used slot of group c | **0.440** | 0.49 isotropic; 0.396 with the measured leverage L = 1.87 |
| grid, aligned, empty address in the *most-used* slot | **0.569** | 0.516 with L = 0.85 |
| random placement | 0.066 | <= 0.23: matches |
| flat control | 0.059 | <= 0.23: matches |
| sign of the mean stored content | 0.231 | 0.229: matches |
| kNN: one stored item sharing 2 factors | 0.393 | 0.392: matches |
| kNN: mean of all stored items sharing 2 factors | 0.727 | (not predicted; a strong exemplar baseline) |
| explicit additive factor model | 0.843 | 0.828: matches |

**The P-sweep** (random slot / most-used slot):

| P | 200 | 400 | 600 | 800 | 900 |
|---|---|---|---|---|---|
| composition | 0.347 / 0.410 | **0.018 / 0.023** | 0.324 / 0.450 | 0.440 / 0.569 | 0.477 / 0.603 |

It is U-shaped, with the collapse exactly at P = Nh, as predicted (the double descent of ridgeless
least squares).

The leverage theory (3.2)-(3.3) orders every condition correctly. Given the measured L, it is
biased low by 0.02-0.06. The likely reason is that I count the erf nonlinearity (0.063) as noise,
though it is partly signal for a sign target.

**The empty address's slot matters (post hoc).** Open addressing fills slots 0, 1, ... in order,
so slot 0's phase is shared by many more items. The main effect of a phase is estimated from the
items that share it, and L falls from 1.87 to 0.85.

**Ridge read-out (post hoc, new).** W_sh = S H_a^T (H_a H_a^T + lambda I)^-1 trades the leverage
variance for bias:

| P | lambda, per mean eigenvalue | composition | stored item's own read-out |
|---|---|---|---|
| 800 | 0 | 0.568 | 0.905 |
| 800 | 0.3 | 0.799 | 0.772 |
| 400 | 1 | 0.781 (from 0.025) | |

At P = 800 the stored read-out also moves towards the item's composite (0.695 -> 0.808). The
read-out regulariser sets the balance between episode and gist.

### C4. Recall under placement (`t4_theory.py`, `t4_summary.py`, `t4b_noise.py`, `t4c_mean.py`; rows in `t4_results.json`)

Coverage: 180 rows (3 seeds x 5 loads x 2 flip rates x 6 placements, including learned and error).
The theory sees only K, H_a, W_gh and the placement: no cue is simulated. The measured values are
my recomputation, which equals the repo's JSON for the same seeds.

| level | address MAE | mean bias | max \|err\| | within 0.05 | module MAE | Kendall tau (placement order) |
|---|---|---|---|---|---|---|
| E: error law, (4.3) | 0.132 | +0.131 | 0.367 | 29% | 0.069 | 0.67 |
| T-lin: templates, no ReLU | 0.012 | -0.005 | 0.057 | 98% | 0.009 | 0.85 |
| **T-relu: templates + moment-matched ReLU** | **0.010** | +0.005 | 0.063 | **99%** | 0.007 | 0.87 |
| T-mc: Gaussian c~, exact ReLU | 0.010 | 0.000 | 0.035 | 100% | 0.007 | 0.87 |

Sampling noise from one cue draw over P = 800 items is about 0.018, so T sits at the noise floor.

**Headline** (load 0.8, 10% flips, mean of 3 seeds, measured vs T-relu):

| placement | measured | T-relu |
|---|---|---|
| sequential | 0.484 | 0.497 |
| random | 0.482 | 0.494 |
| k-means | 0.732 | 0.748 |
| learned | 0.826 | 0.831 |
| error | 0.839 | 0.832 |
| oracle | 0.868 | 0.867 |

T-relu also predicts the noise-law learner's collapse at 20% flips: 0.052 against 0.043 measured
at load 0.8. Address accuracy equals the product of module accuracies to 0.003 (measured) and
0.002 (theory).

**Where the error law alone fails, and why** (`t4b`, `t4c`). E is structurally right but sits on
the steep part of the accuracy curve.
- Under random placement, the bias -alpha K e_i already cuts the true-phase margin from 0.8 beta
  to about 0.26 beta.
- The templates then remove another 15-30% of the margin against the *best* competitor, from the
  per-address deviations of +/-11% of beta.
- They add about 20% phase-difference noise variance. The cross-module leak and interaction terms
  are 0.0006 and 0.0003, against an own-module 0.0068, in units of beta^2.
- They leave 2.5-5.5% of items on a wrong phase even with no noise.

At margin/noise of about 2, those corrections move module accuracy from 0.94 to 0.80. E overstates
modules 1 and 2 most, because they have the fewest items per phase. So the error law's
per-module ranking survives, but its magnitudes do not. **To get magnitudes you must project K
through the templates, T_m = W_gh,m H_a, not through the phase indicators.**

### C5. Redundant modules (`structure.py`, `t5_modules.py`)

**The correlation law (5.1) [derived, matches within 0.022].**

| M | q | predicted rho_h | measured |
|---|---|---|---|
| 3 | 1 / 2 | 0.242 / 0.564 | 0.254 / 0.586 |
| 4 | 1 / 2 / 3 | 0.178 / 0.397 / 0.664 | 0.193 / 0.418 / 0.678 |

The small upward bias comes from per-cell heterogeneity in connectivity: rho_h is convex, so
Jensen pushes the measured values up.

**Pool (5.2)**, random placement, P = 800. The measured q >= 1 sum is 13.9-15.0 at M = 3 and 8.4-9.1
at M = 4. Eq 5.2 gives 12.0 and 7.3. The random q = 0 part is about 1.6 at both.

**A 4th module (3, 4, 5, 7; 176,400 addresses; Nh = 400)**, random placement, P = 800:

| read | flips | M = 3 | M = 4 | flat |
|---|---|---|---|---|
| joint decode over stored place codes | 10% | 0.836 | **0.950** | 0.999 |
| | 20% | 0.411 | **0.618** | 0.940 |
| per-module snap | 10% | 0.481 | 0.323 | |
| | 20% | 0.128 | 0.066 | |
| summed grid inputs at stored phases | 10% | 0.668 | 0.759 | |

The prediction held: the joint decode improves, and the snap gets worse. The snap got worse by more
than the one extra factor. Each module's own snap accuracy also fell, for example module 0 from
0.80 to 0.66-0.76. My **conjecture** for why: a module's share of the pre-activation variance falls
as 1/M, while its template crosstalk grows with M - 1.

---------------------------------------------------------------------------------------------------

## Verdicts, and what is textbook

- **(1) Recall computes where to store: strong, with a correction.**
  - Strong: (1.3) is exact, and the neat form is ||Q^T c - e_k||^2 / r. (2.1)-(2.3) are exact:
    replay is re-encoding with the item's own trace removed.
  - The correction: it has to run on *realized addresses* (2.7). Then online encoding plus replay
    reaches oracle-level recall (0.856 against 0.858) and beats batch swaps (0.774).
  - Weak: "discovers the factors" is not needed, and it did not happen.
  - Weak: "prediction error gates integration" is exact but carries little of the variance.
  - Textbook: the block update is recursive least squares. The LOO identity is the
    precision-matrix / partial-correlation identity of Gaussian graphical models. Single-item
    replay is ICM (Besag 1986) on a Potts-type energy, with its known stalls, for which split-merge
    moves are the standard escape (Jain & Neal 2004).
- **(2) Composition: real but weak as it stands.**
  - Aligned 0.44-0.57, against 0.06 for random placement and the flat control, and 0.39 for a
    2-factor exemplar.
  - It loses to averaging the 2-factor neighbours (0.73) and to an additive model (0.84).
  - It is fully explained as ridgeless least squares on place features (Hastie et al. 2022),
    including the collapse at P = Nh.
  - A ridge read-out lifts it to 0.80, at a cost in episode fidelity.
  - Still rigged: the factors are aligned to the modules by construction.
- **(3) Redundant modules: strong for the joint decode** (+0.11 at 10% flips, +0.21 at 20%), **dead
  for the per-module snap**, and still far from the flat code. The error-correcting view is
  Sreenivasan & Fiete 2011. Eq 5.2 is standard kernel crosstalk.
- **(4) Quantitative theory: strong.**
  - A Gaussian theory with no free parameters predicts address recovery to MAE 0.010 over 180
    conditions.
  - The error law alone does not, although it orders placements moderately well.
  - The honest summary: the error law is the right *object* (K), projected through the wrong
    *map* (Q instead of W_gh H_a).
