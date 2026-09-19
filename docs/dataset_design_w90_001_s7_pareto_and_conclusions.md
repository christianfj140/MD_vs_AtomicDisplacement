# DATASET-DESIGN-W90-001-S7 -- Pareto frontier, figures, conclusions

Cites `docs/dataset_design_w90_001_s1_convention_manifest.md` (S1), and reads S4/S5/S6 outputs verbatim (`Comparison/results/dataset_design_w90_001_s{4,5,6}/`), never recomputing them. Produced by `Comparison/scripts/dataset_design_w90_001_s7_pareto_figures_and_conclusions.py`.

## 1. Best sampler at fixed N_train

Ranking source: S5's `common_validation` set (one predeclared validation distribution shared by every training family, disjoint from S5's three frozen tests -- never used to name a winner from a frozen test, per S1 section 12).

- N_train=16: best family = **angular_shell** (mean H-MAE = 0.3776 eV); full ranking: angular_shell=0.3776, sobol_sparse=0.3951, random_cartesian=0.3975.
- N_train=32: best family = **angular_shell** (mean H-MAE = 0.3579 eV); full ranking: angular_shell=0.3579, sobol_sparse=0.3783, random_cartesian=0.3876.
- N_train=64: best family = **random_cartesian** (mean H-MAE = 0.3603 eV); full ranking: random_cartesian=0.3603, sobol_sparse=0.3677, angular_shell=0.3933.

## 2. Best sampler at fixed SIESTA cost

Ranking source: S5's `common_validation` set (one predeclared validation distribution shared by every training family, disjoint from S5's three frozen tests -- never used to name a winner from a frozen test, per S1 section 12). SIESTA cost for a training point is `N_train + s4.VAL_COUNT` unique SIESTA single-point calculations (S4's `select_split` now draws one *fixed* validation pool per (family, dim, k, seed) group, reused -- not re-drawn -- across every nested N_train, so `train16 subset train32 subset train64`; the marginal cost of the next N_train point is only its newly-added samples, but the absolute unique-run count backing a point still includes the shared validation pool). Because every family pays the same fixed VAL_COUNT offset, the fixed-cost ranking at cost=N_train+VAL_COUNT is identical to conclusion 1's fixed-N ranking; no separate cost-sharing effect was observed or assumed.

## 3. Contribution of 1D -> 2D -> 3D displacement space

Mean H-MAE by dimensionality, restricted to dims where *every* family with any `ok` S4 data also has data (2D_in, 3D); excluded from this pooling because coverage is not shared across families: 1D_z, 1D_in.

- 2D_in: 0.3596 eV
- 3D: 0.3671 eV

Mean H-MAE increases from 2D_in to 3D across the dims common to every family; see `fig6_dimensionality_comparison.png` for the per-family breakdown (the pooled trend is not necessarily monotonic per family). This does not quantify 1D_z, 1D_in coverage, which only exists for a subset of families.

## 4. k=2 vs k=1

Mean H-MAE: k=1 -> 0.3582 eV, k=2 -> 0.3476 eV (pooled over family, dim, N_train, seed; see `fig5_collectivity_locality.png` for the per-family split). This pilot never trained or evaluated k>2 -- no claim is made beyond k in {1, 2}.

## 5. Required displacement amplitudes

S3/S4 trained only at amplitudes (0.03, 0.08) Ang. S5's `common_displacement` frozen test (amplitudes drawn continuously within that same 0.03-0.08 Ang range) gives mean H-MAE 0.3669 eV across training families, vs 0.5696 eV on `ood_displacement` (amplitude 0.12 Ang, outside the training grid). Error grows outside the trained amplitude range, i.e. amplitude coverage up to the intended deployment displacement scale is required -- 0.03-0.08 Ang alone does not certify accuracy at 0.12 Ang.

## 6. Saturation point

- family=angular_shell, dim=3D, k=1: Mean H_MAE barely moves across both nested increments: N=16->32 gain=3.2%, N=32->64 gain=0.7% (both below the 5% threshold); H_MAE(16,32,64) = (0.4046, 0.3915, 0.3886) eV. Not pruned outright (still informs the saturation-point conclusion), but additional N_train beyond 64 is not justified by this evidence.
- family=random_cartesian, dim=1D_in, k=2: Mean H_MAE barely moves across both nested increments: N=16->32 gain=-2.1%, N=32->64 gain=4.1% (both below the 5% threshold); H_MAE(16,32,64) = (0.3397, 0.3468, 0.3324) eV. Not pruned outright (still informs the saturation-point conclusion), but additional N_train beyond 64 is not justified by this evidence.
- family=random_cartesian, dim=3D, k=2: Mean H_MAE barely moves across both nested increments: N=16->32 gain=1.7%, N=32->64 gain=4.0% (both below the 5% threshold); H_MAE(16,32,64) = (0.3925, 0.3856, 0.3700) eV. Not pruned outright (still informs the saturation-point conclusion), but additional N_train beyond 64 is not justified by this evidence.
- family=sobol_sparse, dim=3D, k=1: Mean H_MAE barely moves across both nested increments: N=16->32 gain=4.1%, N=32->64 gain=-0.1% (both below the 5% threshold); H_MAE(16,32,64) = (0.3802, 0.3645, 0.3650) eV. Not pruned outright (still informs the saturation-point conclusion), but additional N_train beyond 64 is not justified by this evidence.

## 7. Generalization to real MD frames

Every S4 model was evaluated on the 7-frame `md_frozen` decorrelated MD test (S5). Best-generalizing training family: **angular_shell** (mean H-MAE = 0.3620 eV over 60 models); worst: **random_cartesian** (mean H-MAE = 0.3675 eV). Caveat: only 7 MD frames are available (S6), all from one trajectory region -- this is a small-sample, single-trajectory generalization estimate, not a paper-grade MD test.

## 8. Can a designed displacement dataset replace MD training data?

**No non-inferiority claim is made.** S6 explicitly could not fix a non-inferiority margin (`delta_ni_fixed: false` in `non_inferiority_margin.json`): no MD-trained baseline model exists at all (every N_train in {16,32,64} is `insufficient_md_training_data`, 13 available MD training frames vs >=16 needed), so there is nothing to be non-inferior to. Per S6's own instruction to later steps, this conclusion is limited to the observed error/cost and Pareto front above (sections 1-4, 9): designed displacement samplers reach H-MAE 0.2324-0.4987 eV at N_train in {16,32,64} on their own validation split, and generalize to `md_frozen` with the errors reported in section 7 -- but whether that is 'as good as MD training data' is undefined without an MD baseline to compare against.

## 9. Pareto-optimal set

Non-dominated (family, dim, k, N_train, seed) points (H-MAE vs SIESTA-run cost):

- cost=32: random_cartesian/1D_in/k=2/seed=1, H-MAE=0.3218 eV
- cost=48: angular_shell/2D_in/k=1/seed=3, H-MAE=0.3008 eV
- cost=80: random_cartesian/2D_in/k=1/seed=4, H-MAE=0.2804 eV

Full table: `pareto_table.csv` (300 points, 3 on the frontier).

## 10. 6x6 candidate grid / k>2 extrapolation

This pilot defined 5 sampler families (angular_shell, axial_radial, local_pair_modes, random_cartesian, sobol_sparse), of which only 3 (angular_shell, random_cartesian, sobol_sparse) produced any `ok` S4 training point -- `axial_radial` and `local_pair_modes` never had enough S3 samples (see pruning log). Coverage actually exercised: 3 usable families x 4 dimensionalities x 2 k values (1, 2) x 3 N_train x 2 seeds. **No 6x6 family/dimensionality grid was constructed, and k was never trained or evaluated above 2.** Every conclusion and figure above is scoped to this tested space; extending any of them to a 6x6 grid or to k>2 would be extrapolation beyond the evidence and is explicitly not done here.
