# Reproduction audit: FieteLab/VectorHaSH @ c71317d

This note records what we found while reproducing Chandra, Sharma, Chaudhuri & Fiete, "Episodic and associative memory from spatial scaffolds in the hippocampus", *Nature* 638 (2025), from the public code at `FieteLab/VectorHaSH`, commit `c71317dbf4dfb207c7a6cb061d2c6a776a380e55`.

The authors released working code, figure notebooks and supplementary material under the MIT licence. This study would not have been possible without that. Almost everything below is about how the notebooks and plotting scripts compute or present a number. None of it changes the core model code.

**How to read this note**
- **Where the numbers come from.** Every number is in [`results/upstream-reference.json`](../results/upstream-reference.json); the relevant key is given for each one. It was produced by running the upstream code itself, unmodified, not this repository's reimplementation. Runs used 3 seeds unless stated. Machine: Apple M5, numpy 2.5.3, Python 3.12.
- **Line numbers.** They refer to the commit above. Notebook locations are given as `cell [i] line j`: `i` is the 0-based index in the `.ipynb` cell list and `j` is the line within that cell.
- **"Ran" vs "inferred".** "Ran" means we executed code and measured the result. "Inferred" means the statement follows from reading the code, but we did not execute that specific path.

## Reproduced exactly

- **Our code matches upstream bit for bit.** A direct port of the recall path gives the same sensory L1 error as the upstream `senstrans_gs_vectorized_patts`, to every printed digit: 0.2076165501165501 / 0.30914445555000275 / 0.3478373875374875 at 1,001 / 2,001 / 3,001 stored patterns (seed 0, clean cues). See `crosschecks.ours_vs_upstream_clean_L1_seed0`. Ran.
- **Clean-cue memory continuum (Fig 3c).**
  - Configuration: the exact upstream call from `Full_model_testing_Fig_3.ipynb` cells [1]–[3], with λ={3,4,5}, Nh=400, Ns=3600 and random ±1 patterns.
  - Recall is perfect up to 401 patterns. Overlap is 0.681 / 0.435 / 0.305 at 801 / 1,601 / 3,001 patterns.
  - MI per input bit is 0.367 / 0.141 / 0.068.
  - Seed-to-seed std is ≤ 0.0004.
  - This agrees with SI Fig S8's caption: "perfect sensory recovery up to 400 patterns".
  - See `curves.fig3_upstream`. Ran.
- **Closed-form law for the continuum.** For ±1 patterns with a sign readout, the overlap follows m = erf(√(Nh / (2(N − Nh)))) for N > Nh.
  - Predicted: 0.683 / 0.436 / 0.305 at N = 800 / 1,600 / 3,000.
  - Measured: 0.682 / 0.435 / 0.305.
  - We derived this formula; it is consistent with SI Sec. D.3 (recall reconstructs a projection onto an Nh-dimensional subspace). See the headline rows `fig = "3c theory"`. The formula is derived; the match is ran.
- **Scaffold fixed points (Fig 2).**
  - **All states fixed with enough hippocampal cells.** With Wgp trained Hebbian-style on all states:
    - λ={3,4,5}: all 3,600 states are fixed points at every tested Nh ≥ 500 (3/3 seeds).
    - λ={2,3,5,7}: all 44,100 states are fixed points at Nh = 800 and 1,000 (3/3 seeds).
  - **Notebook capacity rule.** Using the rule from `Scaffold_testing_Fig_2.ipynb`, capacity is the full 3,600 for every Nh ≥ 250.
  - **Generalization counts match upstream.** Our counts are identical to the `num_correct` returned by upstream `capacity_gcpc_vectorized`. See `crosschecks.fig2C_upstream_num_correct_equals_ours = true` and headline rows `fig = "2d/ED1"`, `"2d"`, `"2f/ED2"`. Ran.
- **Noisy cues (SI Figs S7/S8).** With 10% bit flips at 3,000 stored patterns, the correct scaffold state is recovered for 28% of cues. We read ≈ 0.25 off SI Fig S8 by eye. See headline rows `fig = "3c/S8"`. Ran; the comparison with the figure is visual.

## Items

### 1. The style sheet is loaded by a relative path when modules are imported
- **Where:** `src/data_utils.py:6`. It is imported by `src/senstranspose_utils.py:3`, `src/sensory_utils.py:3`, `src/sensgrid_utils.py:3`, `src/sens_sparseproj_utils.py:3` and `src/capacity_utils.py:2`.
- **What the code does:** it calls `plt.style.use('./src/presentation.mplstyle')` at import time.
- **What a reader would assume:** that `src` imports from any working directory.
- **Measured:** importing `src.data_utils` from outside the repository root fails with `OSError: './src/presentation.mplstyle' is not a valid package style`. From the root it succeeds. See `gotcha_checks.import_data_utils_outside_repo_root` and `gotcha_checks.import_data_utils_at_repo_root`.
- **Status:** ran.

### 2. The two main modules import each other, so import order matters
- **Where:** `src/assoc_utils_np.py:9` (`from src.assoc_utils_np_2D import module_wise_NN_2d`) and `src/assoc_utils_np_2D.py:5` (`from . import assoc_utils_np`).
- **What the code does:** the circular import works only if `assoc_utils_np` is imported first. The notebooks happen to do that.
- **What a reader would assume:** that either module can be imported on its own.
- **Measured:** importing `src.assoc_utils_np_2D` first raises `ImportError: cannot import name 'module_wise_NN_2d' from partially initialized module`. Importing `src.assoc_utils_np` first works. See `gotcha_checks.import_assoc_utils_np_2D_first` and `gotcha_checks.import_assoc_utils_np_first`.
- **Status:** ran.

### 3. Grid→hippocampus connectivity is about 0.67, not the 0.60 in the comment
- **Where:** `src/senstranspose_utils.py:312-315`. The same pattern appears at `src/assoc_utils_np.py:281-283`, `333-335`, `398-400` and `455-457`, among others.
- **What the code does:** it sets `c = 0.60  # connection probability`, then zeroes `int((1-c)*Np*Ng)` (row, column) pairs. The pairs are drawn independently with replacement, so duplicates are common. The pruned fraction is about 1 − e^(−0.4) ≈ 0.33.
- **What a reader would assume:** that 60% of grid→hippocampus synapses are present.
- **Measured:** the effective connection probability is 0.671 (Nh=400, Ng=50), 0.672 (Nh=275, Ng=38) and 0.670 (Nh=400, Ng=99). The analytic value is 0.670. See `gotcha_checks.Wpg_effective_connection_prob`.
- **Effect on results:** not measured. All reference numbers use the upstream construction, so they already include it.
- **Status:** ran.

### 4. The returned "grid error" is actually the sensory error divided by Ng
- **Where:** `src/senstranspose_utils.py:107` (`g_l2_err = np.linalg.norm(sout-strue,axis=(1,2))/(Ng)`), returned as `errgc` at `:118`. The continuous-pattern variant does the same at `:187` and `:198`, dividing by Ns.
- **What the code does:** the second output (`err_gc`) of `capacity(senstrans_gs_vectorized_patts, …)`, as called in `Full_model_testing_Fig_3.ipynb` cell [2], is computed from the sensory output.
- **What a reader would assume:** that `err_gc` measures the grid-state error.
- **Measured (seed 0, clean cues):**
  - The returned `err_gc` is 34.60 at 1,001 patterns and 77.54 at 3,001.
  - Both equal the sensory L2 error divided by Ng, while the true grid-state error is 0.0.
  - See `gotcha_checks.err_gc_is_sensory_error`.
- **Scope:** the notebook's figures use `err_sensl1`, so this matters only to someone reading `err_gc`.
- **Status:** ran.

### 5. Scaffold capacity is judged by mean error, which can hide a few unstable states
- **Where:**
  - `src/assoc_utils_np.py:570` averages the per-pattern error over patterns.
  - `Scaffold_testing_Fig_2.ipynb` cell [4] line 4 sets `errthresh = 0.001`. Capacity is the last number of stored patterns before the *mean* error exceeds the threshold.
  - Separately, `src/assoc_utils_np.py:551` uses the test-noise parameter `pflip` as the standard deviation of training noise too.
- **What a reader would assume:** that "capacity" means every trained state is a fixed point, and that `pflip` controls only test noise.
- **Measured:**
  - **Notebook configuration (λ={3,4,5}, Nh=350, Niter=2).** The notebook rule gives capacity 3,600 for 3/3 seeds. Requiring every trained state to be fixed gives 3,600 / 201 / 901. Up to 5 of 3,600 states are not fixed points.
  - **Wgp trained on all states.** Having all 3,600 states fixed needs Nh = 400–500. With the notebook rule and prefix training, full capacity already appears at Nh = 250.
  - See headline rows `fig = "2 (notebook)"`, `"2d/ED1"` and `"2d"`.
  - **Double use of `pflip`:** not measured; all our scaffold runs use `pflip = 0`.
- **Status:** ran for the capacity rule; inferred from code for the `pflip` double use.

### 6. Only the standard-Hopfield baseline gets an MI floor in the item-memory plot
- **Where:** `Plots_for_baseline_item.py:21`. It zeroes the MI wherever `MI < 2/np.sqrt(708)`, and `:25` uses the result for the standard-Hopfield curve. No other curve in the script gets this floor. The sequence script does not apply it either (`Plots_for_baseline_seq.py:24`).
- **What a reader would assume:** that every curve in the information-per-synapse panel is computed the same way.
- **Measured:**
  - The threshold is 2/√708 = 0.075 bits, which corresponds to an overlap of about 0.32.
  - For the standard Hebbian Hopfield net (N=708, clean cues), MI per synapse at the largest load (α = 1.10) is 0.0535 raw and 0.000 with the floor. See headline row `fig = "3d"`, config "hopfield_hebbian_N708 clean, with Plots_for_baseline_item.py:21 floor".
  - Arithmetic, not a run: the same floor would also zero Vector-HaSH at 3,000 stored patterns with λ={3,4,5} (MI 0.068 < 0.075).
- **Status:** ran for the Hopfield number; inferred (arithmetic) for the Vector-HaSH comparison.

### 7. Cue noise differs between models in the information-per-synapse comparison
- **Where:** `Plots_for_baseline_item.py:18, 27, 35, 60, 69` and `Plots_for_baseline_seq.py:18, 26, 34, 59, 68`. The result filenames encode the cue noise: `noise=0.05` for pseudo-inverse Hopfield and `noise=0.0` for standard, sparse-connectivity, sparse and bounded Hopfield.
- **What a reader would assume:** that all models in the panel were cued under the same condition.
- **Measured (about 500k synapses per model, 3 runs):**
  - **Pseudo-inverse Hopfield (N=708)**, peak MI per synapse: 0.990 with clean cues, 0.536 with 5% flips and 0.453 with 10% flips.
  - **Vector-HaSH (λ={2,3,5}, Nh=275, Ns=900)**, peak MI per synapse: 0.503 clean, 0.447 at 5% and 0.371 at 10%.
  - **Vector-HaSH at the largest load (α = 1.55):** 0.282 clean, 0.000 at 5% flips.
  - See headline rows `fig = "3d"` and `curves.fig3d`.
- **Not recorded:** the cue condition behind the published Vector-HaSH curve, because its data file is not shipped (item 8).
- **Status:** ran for the numbers; inferred for which pairing the published figure uses.

### 8. The Vector-HaSH curve in the sequence plot is the item-memory curve
- **Where:** `Plots_for_baseline_item.py:77` and `Plots_for_baseline_seq.py:76` both load `./MI_235_Np275.npy`, with the same pattern counts `np.arange(1,901,10)` (`:79` and `:78`). `Sequence_results_VH_and_baselines_Fig_5.ipynb` cell [0] lines 50–52 explain that the sequence quantification "does not include scaffold sequence dynamics", on the grounds that the scaffold dynamics factor out.
- **What a reader would assume:** that the Vector-HaSH sequence curve comes from simulating sequence recall end to end.
- **Measured:** not measured. `MI_235_Np275.npy`, `continuum_results/` and `continuum_results_seq/` are not in the repository; see `gotcha_checks.missing_data_files`. That the two plots use the same file is read from the code. Item 9 shows that end-to-end sequence recall depends on scaffold settings that this shortcut skips.
- **Status:** inferred from code.

### 9. The MLP sequence demo resets to the true state at every step
- **Where:** `VectorHASH_minimal_MLP_seq_Fig_5.ipynb`:
  - cell [10] lines 23–24: `s = sbook_flattened[:,i]`, `g = gbook_flattened[:,i]`;
  - cell [10] line 13: the position is decoded from the shifted `g`;
  - cell [3] line 13: `thresh = 2`.
- **What the code does:** each step starts from the true grid and sensory state of step i, predicts one action, and shifts the *true* grid state. The decoded path is therefore a one-step-ahead prediction, so errors cannot accumulate.
- **What a reader would assume:** that the plot titled "Sequence recalled by Vector-HASH" (cell [11]) shows free-running recall from the start state.
- **Measured (seed 0, notebook configuration):**
  - **The notebook's loop:**
    - The MLP fits its training data perfectly (accuracy 1.0).
    - The loop decodes 53.5% of positions correctly (465 of 1,001 wrong).
    - With `thresh = 2`, only 74.0% of path states are scaffold fixed points.
  - **Free-running recall (our addition):**
    - With the notebook's cleanup step, recall fails at step 8 (0.8% of positions correct).
    - Without the cleanup step, all 1,001 steps are correct.
    - With Nh = 800 and `thresh = 0.5`, where 100% of path states are fixed points, all 1,001 steps are correct even with cleanup.
  - See headline rows `fig = "5 (notebook)"` and `fig = "5"`.
  - Inferred from code: each wrong step lands within one lattice step of the correct position, because the shift is applied to the true state. This would make the saved plot look correct at its scale.
- **Status:** ran; the explanation of the plot's appearance is inferred.

### 10. Saved notebooks were executed out of order and need unshipped data
- **Where:**
  - `Full_model_testing_Fig_3.ipynb` has execution counts `[1, 10, 11, 15, 24, 25, 16, 23, 27, 28, 31, 29]`.
  - Its cell [4] (lines 6–7, 18, 20) uses `pbook`, `Wgp` and `Wpg`. No earlier cell defines these at top level, because cell [2] builds them inside `senstrans_gs_vectorized_patts`. They are defined in cells [6]–[7].
  - Cell [6] line 20 loads `../../BW_miniimagenet_3600_60_60_full_rank.npy` and cell [7] line 19 loads `miniimagenet_3600_84_84_3.npy`. So does `Sequence_results_VH_and_baselines_Fig_5.ipynb` cell [0] line 35. None of these files is shipped.
  - `Scaffold_testing_Fig_2.ipynb` has execution counts `[1, 2, 3, 8, 5, 9, 39]`.
- **What a reader would assume:** that "Run All" reproduces the saved outputs.
- **Measured:**
  - After cells [0]–[3], `pbook`, `Wgp` and `Wpg` are undefined; see `gotcha_checks.fig3_cell4_names_defined_by_cells_0_to_3`, all `false`.
  - The data files are absent; see `gotcha_checks.missing_data_files`.
  - Execution counts: `gotcha_checks.notebook_execution_counts`.
- **Status:** ran as a static check, by executing cell [0] and parsing cells [1]–[3]. We did not run the notebooks top to bottom, so the resulting `NameError` is inferred.

### 11. MI per bit is computed from the mean overlap, not averaged per pattern
- **Where:** `Full_model_testing_Fig_3.ipynb` cell [3]. Line 2 takes the mean L1 over patterns (`normlizd_l1 = err_sensl1`), line 7 converts it to an overlap (`m = 1 - (2*normlizd_l1)`), and line 16 computes MI from it (`MI = 1 - S`).
- **What a reader would assume:** "MI per input bit" is the average information recovered per pattern.
- **Measured (λ={3,4,5}, Nh=400, 3,000 stored patterns):**
  - With clean cues the two definitions agree: 0.068 either way.
  - With 10% bit flips they differ by a factor of two: 0.0106 from the mean overlap vs 0.0219 as the mean of per-pattern MI.
  - The cause: noisy recall is bimodal. 28% of cues recover the correct scaffold state (overlap 0.307); the rest recover about 0.
  - See `curves.fig3_cues["flip=0.1"]` (`MI_of_mean_overlap`, `mean_MI`, `p_grid_correct`).
- **Status:** ran.

### 12. Minor code-hygiene items
- **`is` comparisons with string literals.**
  - Where: `Full_model_testing_Fig_3.ipynb` cell [8] line 28 (`if connectivity is 'standard':`), and `Sequence_results_VH_and_baselines_Fig_5.ipynb` cell [1] lines 24 and 65 (`learning is 'bounded_hebbian'`).
  - This relies on CPython string interning. Python 3.12 emits `SyntaxWarning: "is" with 'str' literal`; see `gotcha_checks.fig3_cell8_compile_warnings`. The code still behaves as intended in current CPython.
  - Status: ran.
- **Two helpers were not ported.**
  - Where: `src/assoc_utils_np.py:1-2` notes that `corrupt_p01()` and `topk()` "haven't been converted to numpy". They exist only in `src/assoc_utils.py:99` and `:33`.
  - Measured: `hasattr` returns false for both on `assoc_utils_np`; see `gotcha_checks.assoc_utils_np_has`.
  - Status: ran.
- **Basin-trial count differs between code and saved output.**
  - Where: `Scaffold_testing_Fig_2.ipynb` cell [6] line 27 sets `basin_runs = 100`, but the saved progress bar for that cell shows `10/10`. See `gotcha_checks.fig2_cell6_basin_runs`.
  - The saved basin figure was therefore probably made with 10 trials.
  - Status: ran (read from the notebook file); which count produced the published figure is inferred.
