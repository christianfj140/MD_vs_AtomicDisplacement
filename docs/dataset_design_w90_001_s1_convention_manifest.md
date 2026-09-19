# DATASET-DESIGN-W90-001-S1 — Convention manifest

Audit only. No sampler, SIESTA job, or training code was added in this step.
Every later step in this design (S2+) must read this file first and cite it
when it reuses a function or asserts a convention holds.

## 1. SIESTA basis / mesh / energy convention for the W90 graphene system

The repo runs **two incompatible SIESTA basis regimes**. The W90 graphene
family (`materials/graphene*`, including `graphene_5x5`) uses regime A. Do
not mix it with regime B (legacy H2O `AtomDisplacement` pipeline).

**Regime A — graphene family (`materials/graphene/RUN.fdf`,
`materials/graphene_5x5/RUN.fdf`, `graphene_5x2`, `graphene_5x5_vacancy`,
`graphene_hBN_AA`, `bilayer_graphene_AA`, `twisted_bilayer_graphene_*`):**
- Basis: explicit `%block PAO.Basis` per species (single-zeta radii, e.g. C
  `n=2 0 1` r=4.088 Bohr, `n=2 1 1` r=4.870 Bohr). **No `PAO.BasisSize` and
  no `PAO.EnergyShift` keyword is set** — the basis is fully pinned by the
  explicit cutoff radii instead.
- `MeshCutoff 600.0 Ry` — consistent across every graphene-family template
  checked.
- `XC.functional GGA`, `XC.authors PBE`.
- `LatticeConstant 1.46700 Ang` (i.e. positions given as fractions of the
  C–C-derived lattice constant, in Angstrom).
- Wannierization: `materials/graphene/RUN.fdf` carries a `%block
  Wannier.Manifolds` / `Wannier.Manifold.entangled` block (SIESTA's built-in
  Wannier90-library interface — there is no standalone `wannier90.x`/`.win`
  file anywhere in the repo). Supercell templates (`graphene_5x5`, etc.)
  deliberately omit this block and consume the raw SIESTA H/S output
  instead — see `audit_bundle_v1/03B_material_inputs.md:873`.

**Regime B — legacy `AtomDisplacement/{base,relaxed}/RUN.fdf` (H2O), and
its config mirror `Comparison/config/shared_siesta_settings.yaml`:**
`PAO.BasisType split`, `PAO.BasisSize DZP`, `PAO.EnergyShift 0.03 eV`,
`MeshCutoff 200 Ry`. Rendered by `md_common_settings()` in
`shared/siesta_run_fdf.py:39-64`.

**Convention to use for any new W90 graphene work: Regime A.** Do not
inherit `PAO.BasisSize`/`PAO.EnergyShift`/`MeshCutoff 200 Ry` defaults from
`shared_siesta_settings.yaml` or `md_common_settings()` — those are the H2O
convention and are physics-irrelevant noise for a graphene comparison.

### Known mismatch (fatal — flag before any H-MAE comparison)

`Comparison/config/graphene_5x5_vacancy_pipeline_config.yaml` (`md:` block,
~lines 17-29) declares `basis_size: SZP`, `energy_shift: 0.275 eV`,
`mesh_cutoff: 250 Ry`, `xc_functional: LDA`/`xc_authors: PZ`. None of this
matches the actual `materials/graphene_5x5_vacancy/RUN.fdf` it points at
(`MeshCutoff 600.0 Ry`, explicit PAO.Basis, GGA/PBE, no PAO.EnergyShift).
These YAML keys appear vestigial/unused — the real template is copied
verbatim and only the MD dynamics block is overlaid
(`render_md_layer` in `shared/siesta_run_fdf.py`). **Severity: fatal if
anyone reads `basis_size`/`mesh_cutoff` from this YAML to justify a
comparison — the actual RUN.fdf disagrees with it.** Action for S2+: read
basis/mesh/XC from the rendered `RUN.fdf` (or via
`Comparison/scripts/siesta_settings.py`), never from this YAML's `md:`
block.

The existing mismatch-severity classifier already exists and should be
reused rather than reimplemented:
`Comparison/scripts/siesta_settings.py::COMMON_KEYS` /
`PHYSICS_RELEVANT_KEYS` (lines 28-77) and `siesta_mismatch_severity()`
(line ~89) — classifies `PAO.BasisType/BasisSize/EnergyShift`,
`MeshCutoff`, `XC.functional/authors`, `SolutionMethod`, DM convergence
keys as `"severe"`, everything else `"warning"`.

## 2. Geometry units, PBC, atom ordering — random_cartesian vs MD datasets

Both generators share the same internal geometry model, so they are
compatible:

- **Units: Angstrom**, uniformly. `shared/fdf_materialization.py`'s
  `Structure` dataclass stores `lattice_vectors_ang` (line 51) and exposes
  `positions_ang` (line 65); fdf `LatticeConstant` units (`Ang` or `Bohr`)
  are normalized to Angstrom at parse time (line ~150).
  `AtomDisplacement/scripts/generate_random_cartesian_dataset.py` writes
  metadata fields named `max_displacement_ang`, `sigma_ang`,
  `displacements_ang`, `min_distance_ang` (lines 1446-1457) — no Bohr
  values leak into metadata.
- **PBC**: both generators carry the full `lattice_vectors_ang` through to
  each generated sample (`generate_random_cartesian_dataset.py:1423`
  passes `lattice_vectors_ang=candidate.lattice_vectors_ang` into
  `materialize_sample_fdf`) — lattice vectors are never dropped or
  replaced with a cluster/open-BC geometry.
- **Atom ordering**: neither generator reorders atoms. Order is whatever
  the source material's `RUN.fdf`
  `AtomicCoordinatesAndAtomicSpecies` block defines (for `graphene_5x5`,
  a raster order over the supercell, not grouped by sublattice).
  `generic_random_moving_indices()`
  (`generate_random_cartesian_dataset.py:1248`) selects displaced atoms by
  species filter over `range(len(reference.atom_species))` — the array
  index space is the material's native order, unchanged.

**One caveat for the MD generator**: `MD/scripts/generate_md_dataset.py`
round-trips geometry through SIESTA's native `.XV` restart format
(`parse_xv_geometry()` line 642, `rewrite_run_fdf_from_xv()` line 736).
`.XV` is natively Bohr; verify the Bohr→Angstrom conversion at that
boundary before trusting `positions_ang` equality between an MD sample and
a random_cartesian sample of the same nominal geometry — this was not
independently re-verified in this audit step (flagged as **warning**, not
fatal, since the repo-wide convention is consistently Angstrom on both
sides of that boundary and the conversion is centralized in one function).

## 3. `graphene_5x5` as large-supercell + active-subset reference

`materials/graphene_5x5/` (50-atom pristine supercell, same pseudos/basis
as `materials/graphene/`) plus
`materials/graphene_5x5_vacancy/` (49-atom, one vacancy, test-only OOD
target) already implement exactly the "large supercell + active-subset
selection without N=2 assumptions" pattern:
`Comparison/scripts/build_graphene_5x5_vacancy_target.py` builds the
49-atom target from pristine 5x5 snapshots using
`extract_fdf_structure`/`materialize_fdf_text`
(`shared/fdf_materialization.py`) and `run_siesta` from
`Comparison/scripts/run_hamiltonian_derivative_siesta_references.py`, none
of which assume a 2-atom primitive cell. **This infrastructure is directly
reusable for W90 large-supercell + active-subset work; no new subset
selector is needed.**

## 4. Reusable functions by role (file : symbol)

| Role | File : symbol |
|---|---|
| Displacement generation | `AtomDisplacement/scripts/generate_random_cartesian_dataset.py :: generic_random_displacement_field`, `bounded_gaussian_vector` (1259), `generic_random_moving_indices` (1248) |
| Systematic FC displacement (fdf layer) | `shared/siesta_run_fdf.py :: render_fc_layer` (172) |
| SIESTA job submission | `MD/scripts/generate_md_dataset.py :: run_siesta_with_venv` (435); `Comparison/scripts/run_hamiltonian_derivative_siesta_references.py :: run_siesta` |
| Pipeline orchestration | `Comparison/scripts/g2m_deeph_runner.py` (`_prepare_md_dataset_sweep_config`, 3681) |
| Graph2Mat training | `Comparison/scripts/write_graph2mat_configs.py`, `Comparison/scripts/g2m_deeph_training_sweep.py`, `Comparison/scripts/g2m_deeph_end_to_end_pipeline.py` |
| DeepH gates / certification | `Comparison/scripts/g2m_deeph_gate_check.py`, `g2m_deeph_derivative_gate_check.py`, `deeph_split_audit.py`, `deeph_raw_global_equivalence_preflight.py` |
| Split construction (frozen test) | `MD/scripts/generate_md_dataset.py :: _split_blocked_with_gap` (544), `write_split_manifests` (1318), `write_split_summary` (1371), `write_excluded_gap_manifest` (1214); `AtomDisplacement/scripts/generate_random_cartesian_dataset.py :: random_cartesian_split_group` (1018), `write_split_manifests` (1495) |
| Metric computation | `Comparison/scripts/g2m_deeph_metrics.py :: H_MAE_METRIC_GROUP`, `H_RMSE_METRIC_GROUP` (42, 48); `Comparison/scripts/ml_vs_siesta/viewer.py:72` (MAE on Hamiltonian error arrays) |
| Basis/mesh mismatch severity | `Comparison/scripts/siesta_settings.py :: COMMON_KEYS`, `PHYSICS_RELEVANT_KEYS`, `siesta_mismatch_severity` |
| Figures | `Comparison/scripts/export_graph2mat_matrix_error_plot.py`, `plot_hamiltonian_derivative_metrics.py`, `g2m_deeph_plot_winner_bands.py` |

## 5. Mismatch summary

| Mismatch | Severity | Note |
|---|---|---|
| Regime A (graphene, no BasisSize/EnergyShift, MeshCutoff 600 Ry, GGA/PBE) vs Regime B (H2O, DZP, EnergyShift 0.03 eV, MeshCutoff 200 Ry) | **fatal if conflated** | Never apply `shared_siesta_settings.yaml` / `md_common_settings()` defaults to graphene W90 work. |
| `graphene_5x5_vacancy_pipeline_config.yaml`'s `md:` basis/mesh/XC keys vs actual `RUN.fdf` | **fatal if read as ground truth** | Config keys are vestigial; read basis/mesh from the rendered fdf via `siesta_settings.py`, not this YAML. |
| `.XV` (Bohr) → `positions_ang` (Angstrom) round-trip in MD generator | warning | Conversion is centralized and the surrounding convention is Angstrom-consistent; not independently re-verified this pass. |

## Referenced by

S2 (dataset/sampler design) must cite this file and use Regime A conventions
plus the function inventory in §4 instead of re-deriving them.
