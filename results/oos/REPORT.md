# Out-of-sample test of the zero-fit recall theory: report

**Result: PASS.** On 156 new conditions, over 12 condition families and fresh seeds, the frozen
T-relu theory predicts noisy-cue address recovery with:

| measure | value |
|---|---|
| mean absolute error | **0.009** |
| RMSE | 0.012 |
| worst row | 0.043 |
| rows within 0.05 | 100% |
| correlation with measured recovery | 0.9992 |
| Kendall τ, placement order within each (family, seed) | 0.99 |

Measured recovery ranges from 0.12 to 1.00. The pre-stated bars (MAE ≤ 0.03 and ≥ 90% of rows within 0.05) were written before any measurement. **No family was flagged**: none has a family MAE above 0.05.

## Protocol, in the order it happened (hashes and times in FROZEN.txt and PREDICTIONS.txt)
1. **Theory frozen** at 20:54:34Z.
   - `gold2/theorist/t4_theory.py` and `common.py` were copied verbatim and hashed.
   - One permitted patch was then applied and logged with its diff. The code hard-coded 3 modules and 50 grid cells, so it could not run on the (5,6,7) or (3,4,5,7) scaffolds. Only those literals changed.
2. **Grid defined** in `grid.py`: 14 settings × their placements × seeds 30–32, 156 rows. Every setting changes at least one parameter from the 180 development conditions:
   - loads 0.3 / 0.5 / 0.7;
   - life-log (MiniLM) content;
   - factor cardinalities (7, 12, 4);
   - Nh 200 and 800;
   - periods (5, 6, 7) and (3, 4, 5, 7);
   - Ns 500 and 2,000;
   - masked cues at 25% and 50%;
   - the pseudo-inverse read (α = 0).

   Placements are the repo's `place.py`: sequential, random, k-means, oracle, and error-law learned, wherever each is defined for the scaffold.

   Conditions outside the derivation as written are listed in `grid.NOTES`, and all were still predicted:
   - **Masks** are passed as the flip rate with the same σ²/a². The prediction depends only on that ratio, because ReLU is positively homogeneous and the argmax is scale-free.
   - **Four modules** needed the literal patch.
   - **Nh 200** includes addresses that are not fixed points.
3. **Predictions written** (`predict.py`, theory only, no recall). The sha256 of `predictions.json` and of the saved placements was recorded at **21:02:06Z**, before any measurement. The protocol scripts were hashed next.
4. **A bookkeeping bug**, fixed and logged at 21:03:18Z before any measurement had been seen. The first `measure.py` run crashed and wrote nothing.
   - Row ids omitted the load, so the three "load" settings shared ids, and the saved-placement file kept only load 0.7's arrays.
   - `predictions.json` was not touched; `measure.py` re-checks its hash.
   - Measurement now keys rows by (family, load, seed, placement) and rebuilds every placement deterministically.
   - **132 of 156 rebuilt placements were asserted equal to their saved copies.** The other 24 are the load 0.3 and 0.5 rows, whose saved copies the collision overwrote. For those, determinism is inferred from the other 132.
5. **Measured** (`measure.py`) with the repo's own read, as in bench/placement.py's `_reads`:
   - the cue map `CueMaps(S).map(H_a, α)`, then ReLU, then `Scaffold.settle`;
   - scored against the predicted addresses;
   - on 3 independent cue draws per row. The prediction is an expectation over cues, so it is compared with the mean over draws.
   - The between-draw SD is 0.009 per draw, which sets a sampling floor of about 0.005–0.008 for a 3-draw mean.

## Results by family
| family | rows | MAE | bias | max \|err\| | within 0.05 | module MAE | MAE, error law alone (E) | sampling floor |
|---|---|---|---|---|---|---|---|---|
| load 0.3 / 0.5 / 0.7 | 36 | 0.007 | +0.005 | 0.026 | 100% | 0.003 | 0.096 | 0.007 |
| life-log (MiniLM) | 15 | 0.007 | +0.007 | 0.017 | 100% | 0.005 | 0.206 | 0.009 |
| cardinalities 7/12/4 | 15 | 0.008 | +0.008 | 0.021 | 100% | 0.006 | 0.274 | 0.009 |
| Nh 200 | 9 | 0.004 | +0.002 | 0.011 | 100% | 0.004 | 0.366 | 0.009 |
| Nh 800 | 9 | 0.008 | +0.005 | 0.034 | 100% | 0.004 | 0.096 | 0.007 |
| periods 5,6,7 | 12 | 0.009 | +0.006 | 0.023 | 100% | 0.004 | 0.205 | 0.008 |
| periods 3,4,5,7 (4 modules) | 6 | 0.007 | +0.000 | 0.012 | 100% | 0.007 | 0.450 | 0.010 |
| Ns 500 | 9 | 0.007 | +0.005 | 0.020 | 100% | 0.004 | 0.140 | 0.010 |
| Ns 2,000 | 9 | 0.010 | +0.009 | 0.022 | 100% | 0.005 | 0.247 | 0.006 |
| mask 25% | 12 | 0.006 | +0.004 | 0.017 | 100% | 0.003 | 0.142 | 0.007 |
| mask 50% | 12 | 0.004 | +0.002 | 0.011 | 100% | 0.004 | 0.202 | 0.009 |
| pseudo-inverse read | 12 | **0.029** | **+0.029** | 0.043 | 100% | 0.023 | 0.127 | 0.009 |
| **all** | **156** | **0.009** | **+0.007** | **0.043** | **100%** | 0.006 | 0.187 | 0.008 |

The full table and the worst 10 conditions are in `results.md`. The raw data is `predictions.json`, `measured.json` and `results.json`.

## Failures and weaknesses, named
- **The pseudo-inverse read is the theory's weakest regime.** Its MAE is 0.029, with a consistent +0.029 bias: the theory is optimistic. 9 of the worst 10 rows are pseudo-inverse rows (errors +0.025 to +0.043). It still passes the family bar. A likely cause, not tested: at α = 0 the noise gain P/(Ns − P) is large, so u = H_a c̃ has heavier tails than the Gaussian the theory assumes.
- **A small optimistic bias everywhere** (+0.007 overall, +0.005 without the pseudo-inverse). It is about the size of the sampling floor, but its sign is consistent.
- **The error law alone fails out of sample, as it did in development.** Projecting K only through the phase indicators (level E) gives MAE 0.187 with bias +0.187; it reaches 0.45 on four modules and 0.37 at Nh 200. The magnitudes come from the templates, W_gh H_a, not from the error law. That is the claim the development note made, now confirmed on new conditions.
- **The theory is semi-analytic, not closed form.** It computes exact moments, a CLT over cue bits, and a moment-matched ReLU. It then takes the argmax probabilities by Monte Carlo over a *Gaussian*, 400 draws per item. It never simulates a cue or runs the network.
- **24 of the 156 placements** were rebuilt without a saved copy to check them against (the load 0.3 and 0.5 rows).

## What it means
From K, the placement and the scaffold alone, with nothing fitted, the theory predicts how often a noisy cue finds its own address, to within about 0.01. That holds for content it was not developed on (sentence embeddings, mismatched factor counts), scaffolds it was not developed on (other periods, a fourth module, half and double the place cells), other cue dimensions, a different kind of noise (masks), and new loads. It also ranks placements essentially perfectly (τ 0.99).

The one regime it handles less well is the paper's own write rule, the pseudo-inverse, where it is optimistic by about 0.03.
