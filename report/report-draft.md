# Draft Report Structure

This draft is an outline for the final report. Figures should be inserted as PDF files from `report/figures/`; PNG copies are generated for quick preview.

## 1. Experimental Setup And Metrics

Introduce the five experiment groups:

- Euclidean prox on nonsmooth benchmark objectives.
- Weighted prox geometry on ill-conditioned benchmark objectives.
- Entropy prox on simplex-constrained benchmark objectives.
- Euclidean prox on lasso logistic regression datasets.
- Stochastic simple averages versus weighted SDA on synthetic lasso logistic datasets.

Define the main plotted metrics:

- True objective gap for benchmark objectives: `objective_gap_x_hat = f(x_hat_k) - f(x^*)`.
- Lasso objective gap to sklearn: `objective_value_x_hat - sklearn_train_loss`, where the sklearn SAGA solution is matched by `(dataset, lambda, seed)`.
- Runtime-vs-gap plots use final true objective gap for benchmark objectives and final train objective gap to sklearn for lasso.
- SDA certificate gap `final_gap` is a stopping certificate, not necessarily the true objective gap when the prox-radius assumptions are violated.

## 2. Euclidean Prox On Benchmark Objectives

Use these figure groups from `report/figures/nonsmooth_objective_grid/`:

- `nonsmooth_default_unrestricted.pdf`
- `nonsmooth_default_restricted.pdf`
- `nonsmooth_restriction_norm_gap.pdf`
- `nonsmooth_gamma_sweep.pdf`
- `nonsmooth_gamma_d_grid.pdf`
- `nonsmooth_method_comparison.pdf`
- `nonsmooth_runtime_vs_final_objective_gap.pdf`

Each trajectory figure is one multi-panel plot, with one subplot per objective. Objective ids:

- `abs_a2`
- `ill_conditioned_l1_shift_8d`
- `ill_conditioned_max_affine_8d`
- `linf_shift_2d`
- `max_affine_1d`
- `max_affine_4d`
- `weighted_l1_shift_1d`
- `weighted_l1_shift_2d`
- `weighted_l1_shift_4d`

Draft analysis comments:

- Start with `nonsmooth_default_unrestricted_*`: these show the intended SDA usage, but also expose the failure mode when the minimizer is outside the prox ball. In that case the normalized certificate can be small, or even negative, while the true objective gap remains large.
- Then compare with `nonsmooth_default_restricted_*`: the restricted version keeps iterates inside `F_D`, so the minimization problem effectively becomes the problem on `F_D`. This satisfies the radius condition but solves a less general problem than the unrestricted objective over the larger domain.
- Use `nonsmooth_restriction_norm_gap` to compare `objective_gap_x` and `objective_gap_x_hat` for restricted and unrestricted SDA. This illustrates how the current iterate and averaged iterate can behave differently, and why the restricted and unrestricted variants can diverge even under the same `D` and `gamma_mult`.
- Use `nonsmooth_gamma_sweep` to discuss tuning. Nesterov's proposed `gamma_mult=1` is a baseline, but the plots show it is not always the fastest empirical choice.
- Use `nonsmooth_gamma_d_grid` to discuss the interaction between step scaling and radius. A gamma value that works well for one `D` can be poor for another, so the practical parameter is the pair `(gamma, D)`.
- Use `nonsmooth_method_comparison` to compare SDA simple, weighted SDA, and projected subgradient under matched default parameters.

## 3. Prox Geometry Comparison

Use these figure groups from `report/figures/prox_geometry_comparison/`:

- `prox_geometry.pdf`
- `prox_runtime_vs_final_objective_gap.pdf`

The trajectory figure is one multi-panel plot, with one subplot per objective.

Draft analysis comments:

- This section isolates the effect of prox geometry. The key comparison is Euclidean prox versus weighted Euclidean prox under matched `D=32`, `gamma_mult=1`, and unrestricted runs.
- Weighted geometry should help when the coordinate weights match the objective's anisotropy, because the prox model measures movement in coordinates at the scale used by the subgradients.
- The inverse or mismatched profile is useful as a negative control: it can slow convergence or worsen final objective gap if the geometry emphasizes the wrong coordinates.
- Compare simple and weighted dual averaging separately from the prox profile; the figure should make clear whether the improvement comes from the prox geometry or from the averaging rule.

## 4. Entropy Geometry On Simplex Objectives

Use these figure groups from `report/figures/simplex_entropy_geometry/`:

- `simplex_entropy.pdf`
- `simplex_runtime_vs_final_objective_gap.pdf`

The trajectory figure is one multi-panel plot, with one subplot per simplex objective.

Draft analysis comments:

- Entropy prox is tested only on simplex-constrained objectives, where the natural primal geometry is `L1` and the dual geometry is `L_infinity`.
- The trajectory plots compare simple and weighted entropy SDA for `D in {2, 3}` and `gamma_mult in {0.1, 0.5, 1}`.
- Discuss whether the larger entropy radius improves the final gap or mostly changes early-iteration behavior.
- Use the runtime plot to separate convergence quality from computational overhead.

## 5. Lasso Logistic Sparsity

Use these figure groups from `report/figures/lasso_logistic_sparsity/`:

- `lasso_d_comparison.pdf`
- `lasso_parameter_sweep.pdf`
- `lasso_parameter_sweep_train_objective.pdf`
- `lasso_norms.pdf`
- `lasso_runtime_vs_final_objective_gap_to_sklearn.pdf`

Each lasso trajectory figure is one multi-panel plot, with one subplot per dataset. Dataset stems:

- `iris_csv`
- `synthetic_logistic_small_5d_csv`
- `synthetic_logistic_sparse_20d_csv`
- `synthetic_logistic_noisy_20d_csv`
- `synthetic_logistic_imbalanced_10d_csv`
- `breast_cancer_wdbc_csv`
- `banknote_authentication_csv`
- `spambase_csv`

Draft analysis comments:

- The lasso plots use sklearn SAGA as an empirical optimum reference. For every custom run, the reference is matched by `(dataset, lambda, seed)`.
- The trajectory metric is the objective gap at the averaged iterate: `objective_value_x_hat - sklearn_train_loss`.
- Start with `lasso_d_comparison`: compare SDA simple, weighted SDA, and projected subgradient at `lambda=1`, `seed=0`, and default step parameters across all `D`.
- Then use `lasso_parameter_sweep`: fix `D=32` and show how gamma or alpha tuning changes the gap trajectory.
- Use `lasso_parameter_sweep_train_objective` with the same configuration to show the absolute train objective values and the matched sklearn train objective reference.
- Use `lasso_norms` to compare iterate norms against `||x_sklearn||`. These figures should help explain whether a method approaches the sklearn solution scale even when objective gap decreases slowly.
- Use the runtime-vs-final-gap grid as the summary view for optimization quality relative to computational cost.

## 6. Stochastic Simple Averages On Synthetic Lasso Logistic Datasets

Use these figure groups from `report/figures/lasso_logistic_sparsity_ssa_comparison/`:

- `lasso_ssa_runtime_vs_final_objective_gap_to_sklearn.pdf`
- `lasso_ssa_parameter_sweep.pdf`
- `lasso_ssa_parameter_sweep_train_objective.pdf`

Each trajectory figure is one multi-panel plot, with one subplot per synthetic dataset. Dataset stems:

- `synthetic_logistic_small_5d_csv`
- `synthetic_logistic_sparse_20d_csv`
- `synthetic_logistic_noisy_20d_csv`
- `synthetic_logistic_imbalanced_10d_csv`
- `synthetic_logistic_big_sparse_50d_csv`
- `synthetic_logistic_big_noisy_80d_csv`

Draft analysis comments:

- This section isolates the stochastic oracle effect by comparing weighted SDA against stochastic simple averages (SSA) on lasso logistic objectives. The normal lasso section keeps the original dataset suite; this comparison uses only synthetic datasets, including the two larger generated problems.
- SSA samples a training example at each iteration and forms a stochastic subgradient of the empirical logistic-lasso objective. This makes each iteration cheaper than the full-gradient weighted SDA step, but the trajectory is noisier and the final objective gap must be interpreted relative to runtime as well as iteration count.
- The runtime plot includes sklearn SAGA as the matched empirical reference. The sklearn point has zero final train objective gap by construction after matching on `(dataset, lambda, seed)`, so it should be read as a runtime/reference baseline, not as another SDA-style trajectory.
- Use `lasso_ssa_parameter_sweep` to compare how the `gamma_mult` choice affects final objective-gap decay for weighted SDA and SSA. Both axes use log scaling where appropriate, so multiplicative changes in objective gap are easier to compare.
- Use `lasso_ssa_parameter_sweep_train_objective` to check absolute train objective values against the sklearn train objective reference. This helps distinguish genuine optimization progress from cases where all methods are close in objective but differ visibly in gap scale.
- In the large synthetic datasets, the main question is whether SSA's cheaper stochastic steps can compensate for higher variance. A useful discussion is whether SSA reaches a comparable objective gap faster in wall-clock time, or whether weighted SDA's full-gradient stability dominates despite higher per-iteration cost.

## 7. Cross-Experiment Runtime And Gap Discussion

Use the runtime-vs-gap figures:

- `nonsmooth_objective_grid/nonsmooth_runtime_vs_final_objective_gap.pdf`
- `prox_geometry_comparison/prox_runtime_vs_final_objective_gap.pdf`
- `simplex_entropy_geometry/simplex_runtime_vs_final_objective_gap.pdf`
- `lasso_logistic_sparsity/lasso_runtime_vs_final_objective_gap_to_sklearn.pdf`
- `lasso_logistic_sparsity_ssa_comparison/lasso_ssa_runtime_vs_final_objective_gap_to_sklearn.pdf`

Draft analysis comments:

- The benchmark experiments use known optima, so their runtime plots show true final objective gaps.
- The lasso experiments use sklearn SAGA as the reference, so the final gap is empirical rather than a mathematical certificate.
- Compare the algorithms by both final gap and runtime. A lower final gap at much higher runtime should be discussed separately from genuine efficiency improvements.
- Highlight cases where the stopping certificate and the true objective gap disagree; these motivate the radius-condition discussion from the benchmark section.
