# Dossier 1C — Muestreo cartesiano aleatorio

## Objeto de revisión

Auditar distribuciones y amplitudes, transformaciones geométricas, rechazo de estructuras, reproducibilidad, agrupación familiar y aislamiento de splits.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `AtomDisplacement/pipeline_config.yaml`

SHA-256: `7904a92df49cf33a117252e6371c051b404b2c8924c1a759c956551f7ae5abed`

```yaml
00001 | paths:
00002 |   base_dir: base
00003 |   relaxed_dir: relaxed
00004 |   dataset_dir: dataset
00005 |   samples_dir: dataset/samples
00006 |   collected_dir: dataset/collected
00007 |   training_dir: training
00008 |   run_fdf_name: RUN.fdf
00009 |   run_out_name: RUN.out
00010 |   training_config_name: config.yaml
00011 |   runs_json_name: runs.json
00012 |   samples_manifest_name: samples_manifest.json
00013 |   run_summary_name: run_summary.json
00014 |   collected_json_name: water_atom_displacement_dataset.json
00015 |   collected_csv_name: water_atom_displacement_summary.csv
00016 |   venv_activate: ${REPO_ROOT}/.venv/bin/activate
00017 | material:
00018 |   preset: h2o
00019 | atomic_displacement:
00020 |   recipe: generic_cartesian
00021 |   amplitude_ang: 0.03
00022 |   selected_species: null
00023 |   include_base: true
00024 |   overwrite: false
00025 | commands:
00026 |   graph2mat: graph2mat
00027 |   siesta: siesta
00028 |   shell: bash
00029 |   python: python
00030 | generation:
00031 |   sample_id_format: sample_{index:04d}
00032 | structure:
00033 |   molecule_name: H2O
00034 |   relaxation:
00035 |     system_name: H2O molecule relaxation
00036 |     system_label: h2o_relax
00037 |   single_point:
00038 |     system_name_template: H2O {sample_id}
00039 |     title: Force-constant calculation for H2O atom displacements
00040 |   force_constants:
00041 |     enabled: true
00042 |     displacement: 0.05 Ang
00043 |     target_count: null
00044 |     include_reference: true
00045 |     expand_amplitudes: false
00046 |     subsampling:
00047 |       method: spread
00048 |       seed: 0
00049 |     random_seed: 42
00050 |     combination_mode: aligned
00051 |     max_datasets: 100
00052 |     displacements:
00053 |       0.02 Ang:
00054 |       - 5
00055 |       - 7
00056 |       - 9
00057 |       0.03 Ang:
00058 |       - 4
00059 |       - 6
00060 |       - 8
00061 |       0.05 Ang:
00062 |       - 2
00063 |       - 10
00064 |       - 13
00065 |     allow_missing_matrix: true
00066 |     first_atom: 1
00067 |     last_atom: null
00068 |     lua_script: md_store.lua
00069 |     save_tshs: true
00070 |     save_tsde: true
00071 |     save_dhs: true
00072 |     dHdR_tolerance: -1 Ry/Bohr
00073 |     dSdR_tolerance: -1 1/Bohr
00074 |   random_cartesian:
00075 |     enabled: false
00076 |     recipe: legacy_components
00077 |     n_structures: 100
00078 |     seed: 1234
00079 |     distribution: gaussian
00080 |     sigma_ang: 0.03
00081 |     uniform_range_ang: 0.05
00082 |     move_atoms: all
00083 |     species_filter: []
00084 |     min_distance_ang: 0.65
00085 |     max_attempts_per_structure: 100
00086 |     remove_center_of_mass_translation: true
00087 |     components:
00088 |       atom_displacement:
00089 |         enabled: true
00090 |         distribution: gaussian
00091 |         sigma_ang: 0.03
00092 |         uniform_range_ang: 0.05
00093 |         move_atoms: all
00094 |         species_filter: []
00095 |         remove_center_of_mass_translation: true
00096 |       bond_displacement:
00097 |         enabled: false
00098 |         distribution: gaussian
00099 |         sigma_ang: 0.01
00100 |         uniform_range_ang: 0.02
00101 |         min_delta_ang: null
00102 |         max_delta_ang: null
00103 |         min_bond_ang: 0.70
00104 |         max_bond_ang: 1.30
00105 |         bonds: h2o_oh
00106 |       angle_displacement:
00107 |         enabled: false
00108 |         distribution: gaussian
00109 |         sigma_deg: 3.0
00110 |         uniform_range_deg: 5.0
00111 |         min_delta_deg: null
00112 |         max_delta_deg: null
00113 |         min_angle_deg: 80.0
00114 |         max_angle_deg: 130.0
00115 |         angles: h2o_hoh
00116 |     validation:
00117 |       min_distance_ang: 0.65
00118 |       max_rmsd_from_reference_ang: null
00119 |       max_attempts_per_structure: 100
00120 |   lattice_constant:
00121 |     value: 1.0
00122 |     unit: Ang
00123 |   lattice_vectors:
00124 |   - - 15.0
00125 |     - 0.0
00126 |     - 0.0
00127 |   - - 0.0
00128 |     - 15.0
00129 |     - 0.0
00130 |   - - 0.0
00131 |     - 0.0
00132 |     - 15.0
00133 |   coordinates_format: Ang
00134 |   species:
00135 |   - index: 1
00136 |     atomic_number: 8
00137 |     symbol: O
00138 |   - index: 2
00139 |     atomic_number: 1
00140 |     symbol: H
00141 |   atoms:
00142 |   - label: O
00143 |     species_index: 1
00144 |     position:
00145 |     - 7.5
00146 |     - 7.5
00147 |     - 7.619262
00148 |   - label: H
00149 |     species_index: 2
00150 |     position:
00151 |     - 7.5
00152 |     - 8.263239
00153 |     - 7.022953
00154 |   - label: H
00155 |     species_index: 2
00156 |     position:
00157 |     - 7.5
00158 |     - 6.836839
00159 |     - 7.022953
00160 |   kgrid_monkhorst_pack:
00161 |   - - 1
00162 |     - 0
00163 |     - 0
00164 |     - 0.0
00165 |   - - 0
00166 |     - 1
00167 |     - 0
00168 |     - 0.0
00169 |   - - 0
00170 |     - 0
00171 |     - 1
00172 |     - 0.0
00173 |   siesta:
00174 |     ForceAuxCell: T
00175 |     MeshCutoff: 200 Ry
00176 |     PAO.BasisType: split
00177 |     PAO.BasisSize: DZP
00178 |     PAO.EnergyShift: 0.03 eV
00179 |     XC.functional: GGA
00180 |     XC.authors: PBE
00181 |     MaxSCFIterations: 200
00182 |     SolutionMethod: diagon
00183 |     DM.MixingWeight: 0.02
00184 |     DM.NumberPulay: 3
00185 |     DM.Tolerance: 1.d-5
00186 |     DM.Require.Energy.Convergence: T
00187 |     DM.Energy.Tolerance: 1.e-5 eV
00188 |     SCF.MixAfterConvergence: F
00189 |     SpinPolarized: F
00190 |     FixSpin: F
00191 |     NonCollinearSpin: F
00192 |     DM.InitSpinAF: F
00193 |     ON.UseSaveLWF: T
00194 |     DM.UseSaveDM: T
00195 |     UseSaveData: T
00196 |     LongOutput: T
00197 |     WriteCoorXmol: T
00198 |     WriteCoorStep: T
00199 |     WriteForces: T
00200 |     Save.HS: T
00201 |     XML.Write: T
00202 |   relaxation_md:
00203 |     MD.TypeOfRun: CG
00204 |     MD.NumCGsteps: 200
00205 |     MD.MaxForceTol: 0.04 eV/Ang
00206 |     MD.MaxCGDispl: 0.03 Bohr
00207 |     MD.VariableCell: F
00208 |     MD.ConstantVolume: T
00209 |     MD.UseSaveXV: T
00210 |     MD.UseSaveCG: T
00211 |     WriteMDHistory: T
00212 |   single_point_overrides:
00213 |     DM.UseSaveDM: F
00214 | splits:
00215 |   train: 0.8
00216 |   validation: 0.1
00217 |   test: 0.1
00218 | single_points:
00219 |   limit: null
00220 |   rerun: false
00221 |   # Scientific hardening default: references are reusable only when RUN.out
00222 |   # proves job completion and SCF convergence. Use the CLI debug flag only for
00223 |   # unsafe local recovery/debug workflows.
00224 |   allow_unvalidated_matrices: false
00225 |   workers: 1
00226 | training:
00227 |   torch_float32_matmul_precision: null
00228 |   data:
00229 |     out_matrix: hamiltonian
00230 |     symmetric_matrix: true
00231 |     sub_point_matrix: false
00232 |     matrix_component_policy: h_only
00233 |     n_matrix_components: 1
00234 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00235 |     runs_json: runs.json
00236 |     val_runs: ../dataset/validation_samples/*/RUN.fdf
00237 |     batch_size: 10
00238 |     store_in_memory: true
00239 |   model:
00240 |     num_interactions: 1
00241 |     correlation: 1
00242 |     max_ell: 2
00243 |     hidden_irreps: 10x0e + 10x1o + 10x2e
00244 |     loss: graph2mat.metrics.block_type_mae
00245 |     optim_lr: 0.005
00246 |   trainer:
00247 |     accelerator: cpu
00248 |     logger:
00249 |       class_path: TensorBoardLogger
00250 |       init_args:
00251 |         name: atom_displacement_model
00252 |         save_dir: lightning_logs
00253 |     max_epochs: 200
00254 |   fit_args:
00255 |   - models
00256 |   - mace
00257 |   - main
00258 |   - fit
00259 |   - -c
00260 |   - config.yaml
00261 |   min_completed_samples: 2
00262 |   tensorboard_hint: tensorboard --logdir lightning_logs
00263 | checkpoint:
00264 |   path: null
00265 |   auto_best: true
00266 |   search_glob: lightning_logs/**/checkpoints/best-*.ckpt
00267 |   selection: latest_version
00268 | testing:
00269 |   sample_index: 0
00270 |   data:
00271 |     out_matrix: hamiltonian
00272 |     symmetric_matrix: true
00273 |     sub_point_matrix: false
00274 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00275 |     store_in_memory: true
00276 |     matrix_component_policy: h_only
00277 |     n_matrix_components: 1
00278 |   callbacks:
00279 |     plot_matrix_error: true
00280 |     show_plot: false
00281 |     store_plot_in_logger: false
00282 |     samplewise_metrics_logger: true
00283 |     output_file: sample_metrics.csv
00284 | prediction:
00285 |   data:
00286 |     out_matrix: hamiltonian
00287 |     symmetric_matrix: true
00288 |     sub_point_matrix: false
00289 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00290 |     predict_structs: ../dataset/FC_steps/*/RUN.fdf
00291 |     store_in_memory: true
00292 |     matrix_component_policy: h_only
00293 |     n_matrix_components: 1
00294 |   callbacks:
00295 |     matrix_writer: true
00296 |     output_file: ML_prediction.HSX
00297 | pipeline:
00298 |   skip_model_test: false
00299 |   steps:
00300 |   - render_inputs
00301 |   - generate_atom_displacement_dataset
00302 |   - run_single_points
00303 |   - normalize_fc_steps
00304 |   - collect_atom_displacement_dataset
00305 |   - run_atdisp_training
00306 |   - run_atdisp_testing
00307 |   - run_atdisp_prediction
00308 | performance:
00309 |   max_parallel_siesta_jobs: 1
00310 |   max_parallel_dataset_jobs: 1
00311 |   max_parallel_prediction_jobs: 1
00312 |   max_parallel_evaluation_jobs: 1
00313 |   max_parallel_metric_jobs: 1
00314 |   omp_num_threads: null
00315 |   mkl_num_threads: null
00316 |   openblas_num_threads: null
00317 |   numexpr_num_threads: null
00318 |   torch_num_threads: null
00319 |   compute_accelerator: cpu
00320 |   batch_size: null
00321 |   store_in_memory: null
00322 |   reuse_validated_siesta_outputs: true
00323 |   enable_experiment_cache: false
00324 |   error_policy: fail_fast
00325 |   preset: null
00326 |   torch_float32_matmul_precision: null
```

## `tests/test_generic_random_cartesian.py`

SHA-256: `5f6f07dcbfb857af285d45dc9a477f1689c489f462896ff89591b1fb44cc696e`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import importlib.util
00004 | import json
00005 | import sys
00006 | import tempfile
00007 | import unittest
00008 | from pathlib import Path
00009 | 
00010 | 
00011 | REPO_ROOT = Path(__file__).resolve().parents[1]
00012 | SHARED_DIR = REPO_ROOT / "shared"
00013 | if str(SHARED_DIR) not in sys.path:
00014 |     sys.path.insert(0, str(SHARED_DIR))
00015 | 
00016 | from fdf_materialization import extract_fdf_structure  # noqa: E402
00017 | 
00018 | 
00019 | def load_random_cartesian_module():
00020 |     scripts_dir = REPO_ROOT / "AtomDisplacement" / "scripts"
00021 |     if str(scripts_dir) not in sys.path:
00022 |         sys.path.insert(0, str(scripts_dir))
00023 |     spec = importlib.util.spec_from_file_location(
00024 |         "generate_random_cartesian_dataset_generic_test",
00025 |         scripts_dir / "generate_random_cartesian_dataset.py",
00026 |     )
00027 |     assert spec and spec.loader
00028 |     module = importlib.util.module_from_spec(spec)
00029 |     sys.modules[spec.name] = module
00030 |     spec.loader.exec_module(module)
00031 |     return module
00032 | 
00033 | 
00034 | def synthetic_fdf_text() -> str:
00035 |     return "\n".join(
00036 |         [
00037 |             "SystemName synthetic crystal",
00038 |             "SystemLabel synthetic",
00039 |             "NumberOfSpecies 2",
00040 |             "NumberOfAtoms 4",
00041 |             "%block ChemicalSpeciesLabel",
00042 |             " 1 14 Si",
00043 |             " 2 6 C",
00044 |             "%endblock ChemicalSpeciesLabel",
00045 |             "LatticeConstant 1.0 Ang",
00046 |             "%block LatticeVectors",
00047 |             " 6.0 0.0 0.0",
00048 |             " 0.0 6.0 0.0",
00049 |             " 0.0 0.0 6.0",
00050 |             "%endblock LatticeVectors",
00051 |             "AtomicCoordinatesFormat Ang",
00052 |             "%block AtomicCoordinatesAndAtomicSpecies",
00053 |             " 0.0 0.0 0.0 1",
00054 |             " 2.0 0.0 0.0 2",
00055 |             " 0.0 2.0 0.0 1",
00056 |             " 0.0 0.0 2.0 2",
00057 |             "%endblock AtomicCoordinatesAndAtomicSpecies",
00058 |             "MeshCutoff 200 Ry",
00059 |             "",
00060 |         ]
00061 |     )
00062 | 
00063 | 
00064 | class GenericRandomCartesianTests(unittest.TestCase):
00065 |     def setUp(self) -> None:
00066 |         self.tmp = tempfile.TemporaryDirectory()
00067 |         self.root = Path(self.tmp.name)
00068 |         self.module = load_random_cartesian_module()
00069 |         self.write_material()
00070 | 
00071 |     def tearDown(self) -> None:
00072 |         self.tmp.cleanup()
00073 | 
00074 |     def write_material(self) -> None:
00075 |         material_root = self.root / "materials" / "sic"
00076 |         material_root.mkdir(parents=True)
00077 |         (material_root / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
00078 |         pseudo_dir = material_root / "pseudos"
00079 |         pseudo_dir.mkdir()
00080 |         (pseudo_dir / "Si.psf").write_text("si pseudo\n", encoding="utf-8")
00081 |         (pseudo_dir / "C.psml").write_text("c pseudo\n", encoding="utf-8")
00082 |         basis_dir = material_root / "basis"
00083 |         basis_dir.mkdir()
00084 |         (basis_dir / "Si.ion.xml").write_text("<ion />\n", encoding="utf-8")
00085 | 
00086 |     def config(self, **random_overrides) -> dict:
00087 |         random_config = {
00088 |             "recipe": "generic_cartesian_noise",
00089 |             "n_structures": 4,
00090 |             "max_displacement_ang": 0.04,
00091 |             "selected_species": None,
00092 |             "min_interatomic_distance_ang": 0.5,
00093 |             "remove_center_of_mass_translation": False,
00094 |             "seed": 12345,
00095 |             "variants_per_family": 1,
00096 |             "max_attempts_per_structure": 20,
00097 |         }
00098 |         random_config.update(random_overrides)
00099 |         return {
00100 |             "material": {
00101 |                 "label": "sic",
00102 |                 "fdf": "materials/sic/RUN.fdf",
00103 |                 "pseudopotential_dir": "materials/sic/pseudos",
00104 |                 "basis_dir": "materials/sic/basis",
00105 |                 "structure_type": "crystal",
00106 |             },
00107 |             "random_cartesian": random_config,
00108 |         }
00109 | 
00110 |     def run_generator(self, config: dict, output_name: str = "RandomCartesian_steps") -> dict:
00111 |         return self.module.generate_dataset(
00112 |             config,
00113 |             output_dir=self.root / "dataset" / output_name,
00114 |             material_base_dir=self.root,
00115 |         )
00116 | 
00117 |     def test_non_h2o_generic_random_cartesian_generates_materialized_samples(self) -> None:
00118 |         manifest = self.run_generator(self.config())
00119 |         dataset_root = self.root / "dataset" / "RandomCartesian_steps"
00120 | 
00121 |         self.assertEqual(manifest["recipe"], "generic_cartesian_noise")
00122 |         self.assertEqual(manifest["generated_structures"], 4)
00123 |         self.assertEqual(manifest["material"]["label"], "sic")
00124 |         self.assertTrue((dataset_root / "sample_000001" / "Si.psf").exists())
00125 |         self.assertTrue((dataset_root / "sample_000001" / "C.psml").exists())
00126 |         self.assertTrue((dataset_root / "basis" / "Si.ion.xml").exists())
00127 |         self.assertTrue((dataset_root / "dataset_manifest.json").exists())
00128 |         self.assertTrue((dataset_root / "split_manifest_summary.json").exists())
00129 |         structure = extract_fdf_structure(dataset_root / "sample_000001" / "RUN.fdf")
00130 |         self.assertEqual(structure.atom_count, 4)
00131 |         self.assertEqual([species.label for species in structure.species], ["Si", "C"])
00132 | 
00133 |     def test_fixed_seed_is_deterministic(self) -> None:
00134 |         first = self.run_generator(self.config(), output_name="first")
00135 |         second = self.run_generator(self.config(), output_name="second")
00136 | 
00137 |         self.assertEqual(first["siesta_input_hashes"], second["siesta_input_hashes"])
00138 |         self.assertEqual(
00139 |             first["deterministic_hashes"]["sample_family_hashes"],
00140 |             second["deterministic_hashes"]["sample_family_hashes"],
00141 |         )
00142 | 
00143 |     def test_species_filter_only_displaces_selected_species(self) -> None:
00144 |         manifest = self.run_generator(
00145 |             self.config(selected_species=["C"], n_structures=2),
00146 |         )
00147 | 
00148 |         self.assertEqual([atom["atom_index"] for atom in manifest["selected_atoms"]], [2, 4])
00149 |         dataset_root = self.root / "dataset" / "RandomCartesian_steps"
00150 |         for sample in manifest["samples"]:
00151 |             metadata = json.loads(
00152 |                 (dataset_root / sample["sample_id"] / "metadata.json").read_text(encoding="utf-8")
00153 |             )
00154 |             displacements = metadata["displacements_ang"]
00155 |             self.assertEqual(displacements[0], [0.0, 0.0, 0.0])
00156 |             self.assertEqual(displacements[2], [0.0, 0.0, 0.0])
00157 |             self.assertNotEqual(displacements[1], [0.0, 0.0, 0.0])
00158 |             self.assertNotEqual(displacements[3], [0.0, 0.0, 0.0])
00159 | 
00160 |     def test_min_distance_guard_fails_when_constraints_are_impossible(self) -> None:
00161 |         with self.assertRaisesRegex(
00162 |             RuntimeError,
00163 |             "could not generate a valid structure",
00164 |         ):
00165 |             self.run_generator(
00166 |                 self.config(
00167 |                     n_structures=1,
00168 |                     min_interatomic_distance_ang=10.0,
00169 |                     max_attempts_per_structure=2,
00170 |                 )
00171 |             )
00172 | 
00173 |     def test_group_metadata_keeps_variants_in_same_split(self) -> None:
00174 |         manifest = self.run_generator(
00175 |             self.config(n_structures=4, variants_per_family=2),
00176 |         )
00177 |         summary = manifest["split_summary"]
00178 | 
00179 |         self.assertEqual(summary["group_count"], 2)
00180 |         self.assertEqual(summary["counts"], {"train": 2, "validation": 2, "test": 0})
00181 |         self.assertIn("split_group_id", summary["split_group_keys_used"])
00182 |         for group in summary["groups"]:
00183 |             self.assertEqual(group["sample_count"], 2)
00184 |         group_to_split: dict[str, str] = {}
00185 |         for split_name in ("train", "validation", "test"):
00186 |             split_payload = json.loads(
00187 |                 (
00188 |                     self.root
00189 |                     / "dataset"
00190 |                     / "RandomCartesian_steps"
00191 |                     / f"split_manifest_{split_name}.json"
00192 |                 ).read_text(encoding="utf-8")
00193 |             )
00194 |             for sample in split_payload["samples"]:
00195 |                 previous = group_to_split.setdefault(sample["split_group_id"], split_name)
00196 |                 self.assertEqual(previous, split_name)
00197 | 
00198 |     def test_legacy_h2o_components_remain_separate(self) -> None:
00199 |         config = self.module.random_cartesian_config(
00200 |             {
00201 |                 "structure": {
00202 |                     "random_cartesian": {
00203 |                         "recipe": "legacy_components",
00204 |                         "n_structures": 1,
00205 |                         "components": {
00206 |                             "atom_displacement": {"enabled": False},
00207 |                             "bond_displacement": {"enabled": True, "bonds": "h2o_oh"},
00208 |                             "angle_displacement": {"enabled": False},
00209 |                         },
00210 |                     }
00211 |                 }
00212 |             }
00213 |         )
00214 | 
00215 |         self.assertEqual(config["recipe"], "legacy_components")
00216 |         self.assertTrue(config["components"]["bond_displacement"]["enabled"])
00217 |         self.assertEqual(config["components"]["bond_displacement"]["bonds"], "h2o_oh")
00218 | 
00219 | 
00220 | if __name__ == "__main__":
00221 |     unittest.main()
```

## `AtomDisplacement/scripts/generate_random_cartesian_dataset.py` — extractos seleccionados

SHA-256 del archivo completo: `a5ea4280fcc5a7c5c2f1134d6ab571cbeb71cdf390841f21b64a3e6d146fabb4`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Generate random Cartesian perturbation samples for SIESTA single points."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import copy
00008 | import hashlib
00009 | import json
00010 | import math
00011 | import random
00012 | import re
00013 | import shutil
00014 | import sys
00015 | from pathlib import Path
00016 | from typing import Any
00017 | 
00018 | ATOM_ROOT = Path(__file__).resolve().parents[1]
00019 | REPO_ROOT = ATOM_ROOT.parent
00020 | SHARED_DIR = REPO_ROOT / "shared"
00021 | if str(SHARED_DIR) not in sys.path:
00022 |     sys.path.insert(0, str(SHARED_DIR))
00023 | 
00024 | from fdf_materialization import extract_bundle_structure, materialize_sample_fdf  # noqa: E402
00025 | from material_bundle import BASIS_EXTENSIONS, ValidatedMaterialBundle  # noqa: E402
00026 | from material_presets import resolve_material_bundle  # noqa: E402
00027 | 
00028 | from atom_displacement_utils import (
00029 |     BASE_DIR,
00030 |     DATASET_DIR,
00031 |     PIPELINE_CONFIG,
00032 |     PIPELINE_PATHS,
00033 |     RANDOM_CARTESIAN_STEPS_DIR_NAME,
00034 |     RELAXED_DIR,
00035 |     Structure,
00036 |     angle_degrees,
00037 |     compute_water_geometry_metrics,
00038 |     distance,
00039 |     ensure_dir,
00040 |     load_reference_structure,
00041 |     structure_with_positions,
00042 |     write_json,
00043 | )
00044 | from pipeline_config_utils import render_single_point_fdf
00045 | 
00046 | 
00047 | DEFAULT_RANDOM_CARTESIAN_CONFIG: dict[str, Any] = {
00048 |     "enabled": False,
00049 |     "recipe": "legacy_components",
00050 |     "n_structures": 100,
00051 |     "seed": 1234,
00052 |     "components": None,
00053 |     "distribution": "gaussian",
00054 |     "sigma_ang": 0.03,
00055 |     "uniform_range_ang": 0.05,
00056 |     "max_displacement_ang": None,
00057 |     "move_atoms": "all",
00058 |     "species_filter": [],
00059 |     "selected_species": None,
00060 |     "min_distance_ang": 0.65,
00061 |     "min_interatomic_distance_ang": None,
00062 |     "max_rmsd_from_reference_ang": None,
00063 |     "max_attempts_per_structure": 100,
00064 |     "remove_center_of_mass_translation": True,
00065 |     "variants_per_family": 1,
00066 |     "validation": {},
00067 | }
00068 | 
00069 | BOHR_TO_ANG = 0.529177210903
00070 | SCIENTIFIC_WARNING = "This is a constrained local non-MD perturbation method, not a thermodynamic ensemble."
00071 | GENERIC_RANDOM_CARTESIAN_RECIPE = "generic_cartesian_noise"
00072 | LEGACY_RANDOM_CARTESIAN_RECIPE = "legacy_components"
00073 | COMPONENT_NAMES = ("atom_displacement", "bond_displacement", "angle_displacement")
00074 | SPLIT_NAMES = ("train", "validation", "test")
00075 | RANDOM_CARTESIAN_SPLIT_STRATEGY = "grouped_family_round_robin"
00076 | RANDOM_CARTESIAN_FAMILY_FIELDS = (
00077 |     "base_geometry_hash",
00078 |     "distribution",
00079 |     "sigma_ang",
00080 |     "uniform_range_ang",
00081 |     "seed_family",
00082 |     "move_atoms",
00083 |     "species_filter",
00084 |     "recipe_id",
00085 |     "block_id",
00086 | )
00087 | 
```

### `normalize_validation_config` — líneas 289–314

```py
00289 | def normalize_validation_config(config: dict[str, Any]) -> dict[str, Any]:
00290 |     raw_validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
00291 |     min_distance_source = config.get("min_interatomic_distance_ang")
00292 |     if min_distance_source in (None, ""):
00293 |         min_distance_source = config.get("min_distance_ang", 0.65)
00294 |     validation = {
00295 |         "min_distance_ang": float(raw_validation.get("min_distance_ang", min_distance_source)),
00296 |         "max_rmsd_from_reference_ang": raw_validation.get(
00297 |             "max_rmsd_from_reference_ang",
00298 |             config.get("max_rmsd_from_reference_ang"),
00299 |         ),
00300 |         "max_attempts_per_structure": int(
00301 |             raw_validation.get("max_attempts_per_structure", config.get("max_attempts_per_structure", 100))
00302 |         ),
00303 |     }
00304 |     if validation["min_distance_ang"] < 0:
00305 |         raise RuntimeError("random_cartesian.validation.min_distance_ang no puede ser negativo.")
00306 |     if validation["max_attempts_per_structure"] <= 0:
00307 |         raise RuntimeError("random_cartesian.validation.max_attempts_per_structure debe ser mayor que cero.")
00308 |     if validation["max_rmsd_from_reference_ang"] not in (None, ""):
00309 |         validation["max_rmsd_from_reference_ang"] = float(validation["max_rmsd_from_reference_ang"])
00310 |         if validation["max_rmsd_from_reference_ang"] < 0:
00311 |             raise RuntimeError("random_cartesian.validation.max_rmsd_from_reference_ang no puede ser negativo.")
00312 |     else:
00313 |         validation["max_rmsd_from_reference_ang"] = None
00314 |     return validation
```

### `normalize_component_config` — líneas 317–339

```py
00317 | def normalize_component_config(config: dict[str, Any], *, explicit_components: bool | None = None) -> dict[str, dict[str, Any]]:
00318 |     explicit = has_explicit_component_config(config) if explicit_components is None else explicit_components
00319 |     atom_source = component_source(config, "atom_displacement")
00320 |     bond_source = component_source(config, "bond_displacement")
00321 |     angle_source = component_source(config, "angle_displacement")
00322 |     components = {
00323 |         "atom_displacement": normalize_atom_component(
00324 |             config,
00325 |             atom_source,
00326 |             default_enabled=(not explicit) or (bool(atom_source) and "enabled" not in atom_source),
00327 |         ),
00328 |         "bond_displacement": normalize_bond_component(
00329 |             bond_source,
00330 |             default_enabled=False,
00331 |         ),
00332 |         "angle_displacement": normalize_angle_component(
00333 |             angle_source,
00334 |             default_enabled=False,
00335 |         ),
00336 |     }
00337 |     if not enabled_component_names(components):
00338 |         raise RuntimeError("random_cartesian necesita al menos un componente habilitado.")
00339 |     return components
```

### `normalize_generic_random_cartesian_config` — líneas 357–407

```py
00357 | def normalize_generic_random_cartesian_config(config: dict[str, Any]) -> dict[str, Any]:
00358 |     generic = copy.deepcopy(config)
00359 |     generic["recipe"] = GENERIC_RANDOM_CARTESIAN_RECIPE
00360 |     generic["n_structures"] = int(generic["n_structures"])
00361 |     generic["seed"] = int(generic["seed"])
00362 |     generic["distribution"] = validate_distribution(generic.get("distribution", "uniform"), label="random_cartesian")
00363 |     max_displacement = generic.get("max_displacement_ang", generic.get("max_displacement"))
00364 |     if max_displacement in (None, ""):
00365 |         max_displacement = generic.get("amplitude_ang", generic.get("uniform_range_ang", 0.05))
00366 |     generic["max_displacement_ang"] = numeric_value_with_unit(max_displacement)
00367 |     if generic["max_displacement_ang"] <= 0:
00368 |         raise RuntimeError("random_cartesian.max_displacement_ang debe ser mayor que cero.")
00369 |     if generic["distribution"] == "gaussian":
00370 |         sigma = generic.get("sigma_ang")
00371 |         generic["sigma_ang"] = float(sigma) if sigma not in (None, "") else generic["max_displacement_ang"] / 3.0
00372 |         if generic["sigma_ang"] < 0:
00373 |             raise RuntimeError("random_cartesian.sigma_ang no puede ser negativo.")
00374 |     else:
00375 |         generic["uniform_range_ang"] = generic["max_displacement_ang"]
00376 |     selected_species = generic.get("selected_species", generic.get("species_filter"))
00377 |     if selected_species in (None, "", "all"):
00378 |         generic["selected_species"] = None
00379 |     elif isinstance(selected_species, str):
00380 |         values = [item.strip() for item in selected_species.split(",") if item.strip()]
00381 |         if not values:
00382 |             raise RuntimeError("random_cartesian.selected_species no puede estar vacio; usa null para todas.")
00383 |         generic["selected_species"] = values
00384 |     elif isinstance(selected_species, (list, tuple, set)):
00385 |         values = [str(item).strip() for item in selected_species if str(item).strip()]
00386 |         if not values:
00387 |             raise RuntimeError("random_cartesian.selected_species no puede estar vacio; usa null para todas.")
00388 |         generic["selected_species"] = values
00389 |     else:
00390 |         raise RuntimeError("random_cartesian.selected_species debe ser null, string o lista.")
00391 |     generic["remove_center_of_mass_translation"] = parse_bool(
00392 |         generic.get("remove_center_of_mass_translation"),
00393 |         True,
00394 |     )
00395 |     generic["variants_per_family"] = int(generic.get("variants_per_family", 1) or 1)
00396 |     if generic["variants_per_family"] <= 0:
00397 |         raise RuntimeError("random_cartesian.variants_per_family debe ser mayor que cero.")
00398 |     generic["validation"] = normalize_validation_config(generic)
00399 |     generic["min_distance_ang"] = float(generic["validation"]["min_distance_ang"])
00400 |     generic["min_interatomic_distance_ang"] = generic["min_distance_ang"]
00401 |     generic["max_rmsd_from_reference_ang"] = generic["validation"]["max_rmsd_from_reference_ang"]
00402 |     generic["max_attempts_per_structure"] = int(generic["validation"]["max_attempts_per_structure"])
00403 |     if generic["n_structures"] <= 0:
00404 |         raise RuntimeError("random_cartesian.n_structures debe ser mayor que cero.")
00405 |     if generic.get("blocks"):
00406 |         raise RuntimeError("random_cartesian.recipe=generic_cartesian_noise no soporta blocks en esta fase.")
00407 |     return generic
```

### `random_cartesian_config` — líneas 410–437

```py
00410 | def random_cartesian_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
00411 |     root = config or PIPELINE_CONFIG
00412 |     raw = raw_random_cartesian_config(root)
00413 |     merged = copy.deepcopy(DEFAULT_RANDOM_CARTESIAN_CONFIG)
00414 |     merged.update(raw)
00415 |     recipe = str(merged.get("recipe") or LEGACY_RANDOM_CARTESIAN_RECIPE).strip()
00416 |     if recipe == GENERIC_RANDOM_CARTESIAN_RECIPE:
00417 |         return normalize_generic_random_cartesian_config(merged)
00418 |     if recipe not in {LEGACY_RANDOM_CARTESIAN_RECIPE, "legacy", "components", "h2o_components"}:
00419 |         raise RuntimeError(f"random_cartesian.recipe no soportado: {recipe!r}.")
00420 |     merged["recipe"] = LEGACY_RANDOM_CARTESIAN_RECIPE
00421 |     merged["n_structures"] = int(merged["n_structures"])
00422 |     merged["seed"] = int(merged["seed"])
00423 |     merged["distribution"] = str(merged["distribution"]).strip().lower()
00424 |     apply_random_cartesian_amplitude(merged)
00425 |     merged["sigma_ang"] = float(merged["sigma_ang"])
00426 |     merged["uniform_range_ang"] = float(merged["uniform_range_ang"])
00427 |     merged["distribution"] = validate_distribution(merged["distribution"], label="random_cartesian")
00428 |     merged["components"] = normalize_component_config(merged)
00429 |     merged["validation"] = normalize_validation_config(merged)
00430 |     merged["min_distance_ang"] = float(merged["validation"]["min_distance_ang"])
00431 |     merged["max_rmsd_from_reference_ang"] = merged["validation"]["max_rmsd_from_reference_ang"]
00432 |     merged["max_attempts_per_structure"] = int(merged["validation"]["max_attempts_per_structure"])
00433 |     if merged["n_structures"] <= 0:
00434 |         raise RuntimeError("random_cartesian.n_structures debe ser mayor que cero.")
00435 |     if merged["sigma_ang"] < 0 or merged["uniform_range_ang"] < 0:
00436 |         raise RuntimeError("Las amplitudes random_cartesian no pueden ser negativas.")
00437 |     return merged
```

### `moving_atom_indices` — líneas 518–534

```py
00518 | def moving_atom_indices(structure: Structure, config: dict[str, Any]) -> list[int]:
00519 |     species_filter = [str(item) for item in (config.get("species_filter") or [])]
00520 |     allowed_species = set(species_filter)
00521 |     move_atoms = config.get("move_atoms", "all")
00522 |     if move_atoms in (None, "", "all"):
00523 |         indices = list(range(len(structure.atom_species)))
00524 |     elif isinstance(move_atoms, list):
00525 |         indices = [int(item) - 1 for item in move_atoms]
00526 |     else:
00527 |         raise RuntimeError("random_cartesian.move_atoms debe ser 'all' o una lista de indices 1-based.")
00528 |     if any(index < 0 or index >= len(structure.atom_species) for index in indices):
00529 |         raise RuntimeError("random_cartesian.move_atoms contiene indices fuera de rango.")
00530 |     if allowed_species:
00531 |         indices = [index for index in indices if structure.symbols[index] in allowed_species]
00532 |     if not indices:
00533 |         raise RuntimeError("random_cartesian no tiene atomos movibles tras aplicar species_filter.")
00534 |     return indices
```

### `sample_displacement_vector` — líneas 537–542

```py
00537 | def sample_displacement_vector(rng: random.Random, config: dict[str, Any]) -> list[float]:
00538 |     if config["distribution"] == "gaussian":
00539 |         sigma = float(config["sigma_ang"])
00540 |         return [rng.gauss(0.0, sigma) for _axis in range(3)]
00541 |     half_width = float(config["uniform_range_ang"])
00542 |     return [rng.uniform(-half_width, half_width) for _axis in range(3)]
```

### `displacement_field` — líneas 545–564

```py
00545 | def displacement_field(
00546 |     structure: Structure,
00547 |     config: dict[str, Any],
00548 |     rng: random.Random,
00549 | ) -> list[list[float]]:
00550 |     moving = moving_atom_indices(structure, config)
00551 |     displacements = [[0.0, 0.0, 0.0] for _atom in structure.atom_species]
00552 |     for index in moving:
00553 |         displacements[index] = sample_displacement_vector(rng, config)
00554 |     if bool(config.get("remove_center_of_mass_translation", True)):
00555 |         mean = [
00556 |             sum(displacements[index][axis] for index in moving) / len(moving)
00557 |             for axis in range(3)
00558 |         ]
00559 |         for index in moving:
00560 |             displacements[index] = [
00561 |                 displacements[index][axis] - mean[axis]
00562 |                 for axis in range(3)
00563 |             ]
00564 |     return displacements
```

### `positions_with_displacements` — líneas 567–574

```py
00567 | def positions_with_displacements(
00568 |     structure: Structure,
00569 |     displacements: list[list[float]],
00570 | ) -> list[list[float]]:
00571 |     return [
00572 |         [position[axis] + displacements[index][axis] for axis in range(3)]
00573 |         for index, position in enumerate(structure.positions_ang)
00574 |     ]
```

### `minimum_pair_distance` — líneas 577–582

```py
00577 | def minimum_pair_distance(positions_ang: list[list[float]]) -> float:
00578 |     min_distance = math.inf
00579 |     for left in range(len(positions_ang)):
00580 |         for right in range(left + 1, len(positions_ang)):
00581 |             min_distance = min(min_distance, distance(positions_ang[left], positions_ang[right]))
00582 |     return min_distance
```

### `apply_bond_displacement` — líneas 698–730

```py
00698 | def apply_bond_displacement(
00699 |     reference: Structure,
00700 |     positions: list[list[float]],
00701 |     rng: random.Random,
00702 |     component: dict[str, Any],
00703 | ) -> tuple[list[list[float]], list[dict[str, float]]]:
00704 |     oxygen, h1, h2 = water_atom_indices(reference)
00705 |     updated = copy.deepcopy(positions)
00706 |     deltas: list[dict[str, float]] = []
00707 |     for label, hydrogen in (("oh_1", h1), ("oh_2", h2)):
00708 |         ref_length = distance(reference.positions_ang[oxygen], reference.positions_ang[hydrogen])
00709 |         delta = sample_bounded_scalar(
00710 |             rng,
00711 |             component,
00712 |             sigma_key="sigma_ang",
00713 |             min_key="min_delta_ang",
00714 |             max_key="max_delta_ang",
00715 |         )
00716 |         target_length = ref_length + delta
00717 |         direction = vector_sub(updated[hydrogen], updated[oxygen])
00718 |         if vector_norm(direction) <= 1e-14:
00719 |             direction = vector_sub(reference.positions_ang[hydrogen], reference.positions_ang[oxygen])
00720 |         updated[hydrogen] = vector_add(updated[oxygen], vector_scale(vector_unit(direction), target_length))
00721 |         deltas.append(
00722 |             {
00723 |                 "bond": label,
00724 |                 "atom_indices_0_based": [oxygen, hydrogen],
00725 |                 "reference_length_ang": ref_length,
00726 |                 "delta_ang": target_length - ref_length,
00727 |                 "target_length_ang": target_length,
00728 |             }
00729 |         )
00730 |     return updated, deltas
```

### `apply_angle_displacement` — líneas 740–786

```py
00740 | def apply_angle_displacement(
00741 |     reference: Structure,
00742 |     positions: list[list[float]],
00743 |     rng: random.Random,
00744 |     component: dict[str, Any],
00745 | ) -> tuple[list[list[float]], dict[str, float]]:
00746 |     oxygen, h1, h2 = water_atom_indices(reference)
00747 |     updated = copy.deepcopy(positions)
00748 |     ref_angle = angle_degrees(
00749 |         reference.positions_ang[h1],
00750 |         reference.positions_ang[oxygen],
00751 |         reference.positions_ang[h2],
00752 |     )
00753 |     delta = sample_bounded_scalar(
00754 |         rng,
00755 |         component,
00756 |         sigma_key="sigma_deg",
00757 |         min_key="min_delta_deg",
00758 |         max_key="max_delta_deg",
00759 |     )
00760 |     target_angle = ref_angle + delta
00761 |     pivot = updated[oxygen]
00762 |     v1 = vector_sub(updated[h1], pivot)
00763 |     v2 = vector_sub(updated[h2], pivot)
00764 |     r1 = vector_norm(v1)
00765 |     r2 = vector_norm(v2)
00766 |     if r1 <= 1e-14 or r2 <= 1e-14:
00767 |         raise RuntimeError("No se puede aplicar angle_displacement con un enlace O-H de longitud cero.")
00768 |     current_angle = angle_degrees(updated[h1], updated[oxygen], updated[h2])
00769 |     axis = vector_cross(v1, v2)
00770 |     if vector_norm(axis) <= 1e-12:
00771 |         axis = perpendicular_axis(v1)
00772 |     delta_rad = math.radians(target_angle - current_angle)
00773 |     new_v1 = rotate_vector(vector_scale(vector_unit(v1), r1), axis, -0.5 * delta_rad)
00774 |     new_v2 = rotate_vector(vector_scale(vector_unit(v2), r2), axis, 0.5 * delta_rad)
00775 |     updated[h1] = vector_add(pivot, new_v1)
00776 |     updated[h2] = vector_add(pivot, new_v2)
00777 |     final_angle = angle_degrees(updated[h1], updated[oxygen], updated[h2])
00778 |     return updated, {
00779 |         "angle": "h2o_hoh",
00780 |         "atom_indices_0_based": [h1, oxygen, h2],
00781 |         "reference_angle_deg": ref_angle,
00782 |         "current_angle_before_deg": current_angle,
00783 |         "delta_deg": target_angle - ref_angle,
00784 |         "target_angle_deg": target_angle,
00785 |         "final_angle_deg": final_angle,
00786 |     }
```

### `apply_atom_displacement` — líneas 789–807

```py
00789 | def apply_atom_displacement(
00790 |     reference: Structure,
00791 |     positions: list[list[float]],
00792 |     rng: random.Random,
00793 |     component: dict[str, Any],
00794 | ) -> tuple[list[list[float]], list[list[float]]]:
00795 |     component_config = {
00796 |         "distribution": component["distribution"],
00797 |         "sigma_ang": component["sigma_ang"],
00798 |         "uniform_range_ang": component["uniform_range_ang"],
00799 |         "move_atoms": component.get("move_atoms", "all"),
00800 |         "species_filter": component.get("species_filter") or [],
00801 |         "remove_center_of_mass_translation": component.get("remove_center_of_mass_translation", True),
00802 |     }
00803 |     displacements = displacement_field(reference, component_config, rng)
00804 |     return [
00805 |         [position[axis] + displacements[index][axis] for axis in range(3)]
00806 |         for index, position in enumerate(positions)
00807 |     ], displacements
```

### `remove_mean_translation_from_reference` — líneas 810–824

```py
00810 | def remove_mean_translation_from_reference(
00811 |     reference: Structure,
00812 |     positions: list[list[float]],
00813 | ) -> tuple[list[list[float]], list[float]]:
00814 |     if len(reference.positions_ang) != len(positions):
00815 |         return positions, [0.0, 0.0, 0.0]
00816 |     translation = [
00817 |         sum(positions[index][axis] - reference.positions_ang[index][axis] for index in range(len(positions)))
00818 |         / len(positions)
00819 |         for axis in range(3)
00820 |     ]
00821 |     return [
00822 |         [position[axis] - translation[axis] for axis in range(3)]
00823 |         for position in positions
00824 |     ], translation
```

### `build_geometry_metrics` — líneas 836–842

```py
00836 | def build_geometry_metrics(reference: Structure, candidate: Structure) -> dict[str, Any]:
00837 |     metrics: dict[str, Any] = {
00838 |         "minimum_pair_distance_ang": minimum_pair_distance(candidate.positions_ang),
00839 |         "rmsd_from_reference_ang": rmsd_from_reference(reference, candidate),
00840 |     }
00841 |     metrics.update(water_geometry_metrics_or_none(candidate))
00842 |     return metrics
```

### `generate_candidate` — líneas 845–890

```py
00845 | def generate_candidate(
00846 |     reference: Structure,
00847 |     block_config: dict[str, Any],
00848 |     rng: random.Random,
00849 | ) -> tuple[Structure, dict[str, Any]]:
00850 |     components = block_config["components"]
00851 |     positions = copy.deepcopy(reference.positions_ang)
00852 |     deltas: dict[str, Any] = {
00853 |         "atom_displacements_ang": None,
00854 |         "bond_deltas": [],
00855 |         "angle_delta": None,
00856 |         "center_of_mass_translation_removed_ang": [0.0, 0.0, 0.0],
00857 |     }
00858 |     if components["bond_displacement"]["enabled"]:
00859 |         positions, deltas["bond_deltas"] = apply_bond_displacement(
00860 |             reference,
00861 |             positions,
00862 |             rng,
00863 |             components["bond_displacement"],
00864 |         )
00865 |     if components["angle_displacement"]["enabled"]:
00866 |         positions, deltas["angle_delta"] = apply_angle_displacement(
00867 |             reference,
00868 |             positions,
00869 |             rng,
00870 |             components["angle_displacement"],
00871 |         )
00872 |     if components["atom_displacement"]["enabled"]:
00873 |         positions, deltas["atom_displacements_ang"] = apply_atom_displacement(
00874 |             reference,
00875 |             positions,
00876 |             rng,
00877 |             components["atom_displacement"],
00878 |         )
00879 |         if (
00880 |             components["atom_displacement"].get("remove_center_of_mass_translation", True)
00881 |             and (
00882 |                 components["bond_displacement"]["enabled"]
00883 |                 or components["angle_displacement"]["enabled"]
00884 |             )
00885 |         ):
00886 |             positions, deltas["center_of_mass_translation_removed_ang"] = remove_mean_translation_from_reference(
00887 |                 reference,
00888 |                 positions,
00889 |             )
00890 |     return structure_with_positions(reference, positions), deltas
```

### `validate_random_structure` — líneas 893–939

```py
00893 | def validate_random_structure(
00894 |     reference: Structure,
00895 |     candidate: Structure,
00896 |     *,
00897 |     block_config: dict[str, Any],
00898 |     base_geometry_hash: str | None = None,
00899 | ) -> tuple[bool, str, dict[str, Any]]:
00900 |     metrics = build_geometry_metrics(reference, candidate)
00901 |     if len(candidate.atom_species) != len(reference.atom_species):
00902 |         return False, "atom_count_changed", metrics
00903 |     if candidate.atom_species != reference.atom_species or candidate.symbols != reference.symbols:
00904 |         return False, "species_changed", metrics
00905 |     if candidate.lattice_vectors_ang != reference.lattice_vectors_ang:
00906 |         return False, "cell_changed", metrics
00907 |     if any(
00908 |         not math.isfinite(value)
00909 |         for position in candidate.positions_ang
00910 |         for value in position
00911 |     ):
00912 |         return False, "non_finite_coordinate", metrics
00913 |     if base_geometry_hash and json_sha256(reference.to_json_dict()) != base_geometry_hash:
00914 |         return False, "reference_mutated", metrics
00915 |     validation = block_config["validation"]
00916 |     min_distance_ang = float(validation["min_distance_ang"])
00917 |     if metrics["minimum_pair_distance_ang"] < min_distance_ang:
00918 |         return False, "min_distance_below_threshold", metrics
00919 |     max_rmsd = validation.get("max_rmsd_from_reference_ang")
00920 |     if max_rmsd is not None and metrics["rmsd_from_reference_ang"] > float(max_rmsd):
00921 |         return False, "rmsd_above_threshold", metrics
00922 |     components = block_config["components"]
00923 |     if components["bond_displacement"]["enabled"] or components["angle_displacement"]["enabled"]:
00924 |         if not {"oh_1_ang", "oh_2_ang", "hoh_angle_deg"}.issubset(metrics):
00925 |             return False, "not_h2o_topology", metrics
00926 |         bond_component = components["bond_displacement"]
00927 |         min_bond = float(bond_component["min_bond_ang"])
00928 |         max_bond = float(bond_component["max_bond_ang"])
00929 |         if metrics["oh_1_ang"] < min_bond or metrics["oh_1_ang"] > max_bond:
00930 |             return False, "oh_bond_out_of_range", metrics
00931 |         if metrics["oh_2_ang"] < min_bond or metrics["oh_2_ang"] > max_bond:
00932 |             return False, "oh_bond_out_of_range", metrics
00933 |     if components["angle_displacement"]["enabled"]:
00934 |         angle_component = components["angle_displacement"]
00935 |         min_angle = float(angle_component["min_angle_deg"])
00936 |         max_angle = float(angle_component["max_angle_deg"])
00937 |         if metrics["hoh_angle_deg"] < min_angle or metrics["hoh_angle_deg"] > max_angle:
00938 |             return False, "hoh_angle_out_of_range", metrics
00939 |     return True, "ok", metrics
```

### `random_cartesian_family_payload` — líneas 942–964

```py
00942 | def random_cartesian_family_payload(base_geometry_hash: str, config: dict[str, Any]) -> dict[str, Any]:
00943 |     amplitude = (
00944 |         config["sigma_ang"]
00945 |         if config["distribution"] == "gaussian"
00946 |         else config["uniform_range_ang"]
00947 |     )
00948 |     seed_family = int(config["seed"])
00949 |     return {
00950 |         "generation_method": "random_cartesian",
00951 |         "base_geometry_hash": base_geometry_hash,
00952 |         "distribution": config["distribution"],
00953 |         "amplitude_ang": amplitude,
00954 |         "sigma_ang": float(config["sigma_ang"]) if config["distribution"] == "gaussian" else None,
00955 |         "uniform_range_ang": float(config["uniform_range_ang"]) if config["distribution"] == "uniform" else None,
00956 |         "seed_family": seed_family,
00957 |         "species_filter": config.get("species_filter") or [],
00958 |         "move_atoms": config.get("move_atoms", "all"),
00959 |         "enabled_components": enabled_component_names(config["components"]),
00960 |         "component_configuration": config["components"],
00961 |         "validation": config["validation"],
00962 |         "recipe_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_id"),
00963 |         "block_id": config.get("block_id") or (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_id"),
00964 |     }
```

### `deterministic_split_group_id` — líneas 967–968

```py
00967 | def deterministic_split_group_id(base_geometry_hash: str, config: dict[str, Any]) -> str:
00968 |     return json_sha256(random_cartesian_family_payload(base_geometry_hash, config))
```

### `random_cartesian_split_group` — líneas 1016–1035

```py
01016 | def random_cartesian_split_group(sample: dict[str, Any]) -> tuple[str, str, str | None]:
01017 |     for key in ("split_group_id", "random_cartesian_family_id"):
01018 |         value = sample.get(key)
01019 |         if value not in (None, ""):
01020 |             return str(value), key, None
01021 |     derived = _derived_family_group_id(sample)
01022 |     if derived:
01023 |         return (
01024 |             derived,
01025 |             "derived_random_cartesian_family",
01026 |             "missing split_group_id/random_cartesian_family_id; derived split group from family metadata",
01027 |         )
01028 |     sample_id = str(sample.get("sample_id") or sample.get("sample_dir") or "")
01029 |     if not sample_id:
01030 |         sample_id = hashlib.sha256(repr(sample).encode("utf-8", errors="ignore")).hexdigest()
01031 |     return (
01032 |         f"sample_fallback:{sample_id}",
01033 |         "sample_id_fallback",
01034 |         "missing random_cartesian group metadata; each sample was treated as its own weak fallback group",
01035 |     )
```

### `grouped_split_assignment` — líneas 1038–1084

```py
01038 | def grouped_split_assignment(samples: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
01039 |     groups: dict[str, dict[str, Any]] = {}
01040 |     warnings: list[str] = []
01041 |     for order, sample in enumerate(samples):
01042 |         group_id, group_key, warning = random_cartesian_split_group(sample)
01043 |         group = groups.setdefault(
01044 |             group_id,
01045 |             {
01046 |                 "group_id": group_id,
01047 |                 "group_key": group_key,
01048 |                 "first_index": order,
01049 |                 "samples": [],
01050 |             },
01051 |         )
01052 |         group["samples"].append(sample)
01053 |         if warning:
01054 |             warnings.append(warning)
01055 | 
01056 |     ordered_groups = sorted(groups.values(), key=lambda item: (int(item["first_index"]), str(item["group_id"])))
01057 |     split_samples: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
01058 |     group_assignments: list[dict[str, Any]] = []
01059 |     for group_index, group in enumerate(ordered_groups):
01060 |         split = SPLIT_NAMES[group_index % len(SPLIT_NAMES)]
01061 |         rows = group["samples"]
01062 |         split_samples[split].extend(rows)
01063 |         group_assignments.append(
01064 |             {
01065 |                 "group_id": group["group_id"],
01066 |                 "group_key": group["group_key"],
01067 |                 "split": split,
01068 |                 "sample_count": len(rows),
01069 |                 "sample_ids": [sample.get("sample_id") for sample in rows],
01070 |             }
01071 |         )
01072 | 
01073 |     unique_warnings = sorted(dict.fromkeys(warnings))
01074 |     summary = {
01075 |         "split_strategy": RANDOM_CARTESIAN_SPLIT_STRATEGY,
01076 |         "group_aware": True,
01077 |         "split_group_keys_used": sorted({assignment["group_key"] for assignment in group_assignments}),
01078 |         "groups": group_assignments,
01079 |         "group_count": len(group_assignments),
01080 |         "counts": {split: len(rows) for split, rows in split_samples.items()},
01081 |         "warnings": unique_warnings,
01082 |         "scientific_status": "grouped_family_splits" if not unique_warnings else "grouped_family_splits_with_fallback",
01083 |     }
01084 |     return split_samples, summary
```

### `assert_group_isolation` — líneas 1087–1098

```py
01087 | def assert_group_isolation(split_samples: dict[str, list[dict[str, Any]]]) -> None:
01088 |     seen: dict[str, str] = {}
01089 |     for split, rows in split_samples.items():
01090 |         for sample in rows:
01091 |             group_id, _group_key, _warning = random_cartesian_split_group(sample)
01092 |             previous = seen.get(group_id)
01093 |             if previous is not None and previous != split:
01094 |                 raise RuntimeError(
01095 |                     "random_cartesian grouped split attempted to split a family "
01096 |                     f"across {previous} and {split}: {group_id}"
01097 |                 )
01098 |             seen[group_id] = split
```

### `write_split_manifests` — líneas 1101–1117

```py
01101 | def write_split_manifests(dataset_root: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
01102 |     split_samples, summary = grouped_split_assignment(samples)
01103 |     assert_group_isolation(split_samples)
01104 |     for split, rows in split_samples.items():
01105 |         write_json(
01106 |             dataset_root / f"split_manifest_{split}.json",
01107 |             {
01108 |                 "split": split,
01109 |                 "split_strategy": summary["split_strategy"],
01110 |                 "group_aware": True,
01111 |                 "split_group_keys_used": summary["split_group_keys_used"],
01112 |                 "warnings": summary["warnings"],
01113 |                 "samples": rows,
01114 |             },
01115 |         )
01116 |     write_json(dataset_root / "split_manifest_summary.json", summary)
01117 |     return summary
```

### `generate_generic_random_cartesian_dataset` — líneas 1267–1482

```py
01267 | def generate_generic_random_cartesian_dataset(
01268 |     config: dict[str, Any],
01269 |     rc_config: dict[str, Any],
01270 |     *,
01271 |     output_dir: str | Path | None = None,
01272 |     material_base_dir: str | Path = REPO_ROOT,
01273 | ) -> dict[str, Any]:
01274 |     dataset_root = Path(output_dir) if output_dir is not None else DATASET_DIR / RANDOM_CARTESIAN_STEPS_DIR_NAME
01275 |     if dataset_root.exists():
01276 |         shutil.rmtree(dataset_root)
01277 |     ensure_dir(dataset_root)
01278 | 
01279 |     resolved = resolve_material_bundle(config, base_dir=material_base_dir)
01280 |     validated = resolved.validated
01281 |     resources = copy_generic_material_resources(validated, dataset_root)
01282 |     reference, _fdf_structure = structure_from_material_bundle(validated)
01283 |     source_path = str(validated.bundle.fdf)
01284 |     base_geometry_hash = json_sha256(reference.to_json_dict())
01285 |     dataset_recipe = config.get("dataset_recipe") or PIPELINE_CONFIG.get("dataset_recipe") or {}
01286 |     rng = random.Random(int(rc_config["seed"]))
01287 |     validation_block = generic_validation_block(rc_config)
01288 |     selected_indices = generic_random_moving_indices(reference, rc_config)
01289 |     selected_atoms = [
01290 |         {
01291 |             "atom_index": index + 1,
01292 |             "atom_index_zero_based": index,
01293 |             "species": reference.symbols[index],
01294 |             "species_index": int(reference.atom_species[index]),
01295 |         }
01296 |         for index in selected_indices
01297 |     ]
01298 | 
01299 |     print("=== Random Cartesian dataset generation ===")
01300 |     print(f"[INFO] Recipe: {GENERIC_RANDOM_CARTESIAN_RECIPE}")
01301 |     print(f"[INFO] Material FDF: {source_path}")
01302 |     print(f"[INFO] Output root: {dataset_root}")
01303 |     print(f"[INFO] n_structures: {rc_config['n_structures']}")
01304 | 
01305 |     samples: list[dict[str, Any]] = []
01306 |     total_attempts = 0
01307 |     rejection_counts: dict[str, int] = {}
01308 |     max_component = float(rc_config["max_displacement_ang"])
01309 |     for sample_index in range(int(rc_config["n_structures"])):
01310 |         family_index = sample_index // int(rc_config["variants_per_family"])
01311 |         family_payload = generic_family_payload(base_geometry_hash, rc_config, family_index, dataset_recipe)
01312 |         split_group_id = json_sha256(family_payload)
01313 |         accepted: tuple[int, Structure, list[list[float]], dict[str, Any], str | None] | None = None
01314 |         last_reason = "not_attempted"
01315 |         last_rejected_reason: str | None = None
01316 |         for attempt in range(1, int(rc_config["max_attempts_per_structure"]) + 1):
01317 |             total_attempts += 1
01318 |             displacements = generic_random_displacement_field(reference, rc_config, rng)
01319 |             if any(abs(value) > max_component + 1e-12 for vector in displacements for value in vector):
01320 |                 reason = "max_displacement_exceeded_after_centering"
01321 |                 rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
01322 |                 last_reason = reason
01323 |                 last_rejected_reason = reason
01324 |                 continue
01325 |             positions = positions_with_displacements(reference, displacements)
01326 |             candidate = structure_with_positions(reference, positions)
01327 |             ok, reason, geometry_metrics = validate_random_structure(
01328 |                 reference,
01329 |                 candidate,
01330 |                 block_config=validation_block,
01331 |                 base_geometry_hash=base_geometry_hash,
01332 |             )
01333 |             last_reason = reason
01334 |             if ok:
01335 |                 accepted = (attempt, candidate, displacements, geometry_metrics, last_rejected_reason)
01336 |                 break
01337 |             rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
01338 |             last_rejected_reason = reason
01339 |         if accepted is None:
01340 |             raise RuntimeError(
01341 |                 "random_cartesian generic_cartesian_noise could not generate a valid structure "
01342 |                 f"for sample_index={sample_index} after {rc_config['max_attempts_per_structure']} "
01343 |                 f"attempts: {last_reason}."
01344 |             )
01345 | 
01346 |         accepted_attempt, candidate, displacements, geometry_metrics, last_rejected_reason = accepted
01347 |         sample_id = f"sample_{sample_index + 1:06d}"
01348 |         sample_dir = dataset_root / sample_id
01349 |         sample_dir.mkdir(parents=True, exist_ok=True)
01350 |         for pseudo in resources["pseudopotentials"]:
01351 |             shutil.copy2(pseudo, sample_dir / Path(pseudo).name)
01352 |         materialized = materialize_sample_fdf(
01353 |             validated.bundle.fdf,
01354 |             sample_dir / "RUN.fdf",
01355 |             positions_ang=candidate.positions_ang,
01356 |             atom_species=candidate.atom_species,
01357 |             lattice_vectors_ang=candidate.lattice_vectors_ang,
01358 |             system_label=sample_id,
01359 |             system_name=f"{validated.bundle.label} {sample_id}",
01360 |             structure_type=validated.bundle.structure_type,
01361 |         )
01362 |         metadata = {
01363 |             "id": sample_id,
01364 |             "generation_method": "random_cartesian",
01365 |             "method": "random_cartesian",
01366 |             "recipe": GENERIC_RANDOM_CARTESIAN_RECIPE,
01367 |             "recipe_id": dataset_recipe.get("recipe_id"),
01368 |             "recipe_label": dataset_recipe.get("recipe_label"),
01369 |             "material_label": validated.bundle.label,
01370 |             "selected_atoms": selected_atoms,
01371 |             "base_geometry_hash": base_geometry_hash,
01372 |             "base_geometry_source": source_path,
01373 |             "seed": int(rc_config["seed"]),
01374 |             "seed_family": int(rc_config["seed"]),
01375 |             "family_index": family_index,
01376 |             "variants_per_family": int(rc_config["variants_per_family"]),
01377 |             "sample_index": sample_index,
01378 |             "global_sample_id": sample_id,
01379 |             "distribution": rc_config["distribution"],
01380 |             "max_displacement_ang": max_component,
01381 |             "sigma_ang": float(rc_config["sigma_ang"]) if rc_config["distribution"] == "gaussian" else None,
01382 |             "selected_species": rc_config.get("selected_species"),
01383 |             "species_filter": rc_config.get("selected_species") or [],
01384 |             "remove_center_of_mass_translation": bool(rc_config.get("remove_center_of_mass_translation", True)),
01385 |             "displacements_ang": displacements,
01386 |             "atom_displacements_ang": displacements,
01387 |             "final_geometry_metrics": geometry_metrics,
01388 |             "minimum_pair_distance_ang": geometry_metrics.get("minimum_pair_distance_ang"),
01389 |             "rmsd_from_reference_ang": geometry_metrics.get("rmsd_from_reference_ang"),
01390 |             "validation_thresholds": rc_config["validation"],
01391 |             "min_distance_ang": float(rc_config["min_distance_ang"]),
01392 |             "accepted_attempt": accepted_attempt,
01393 |             "acceptance_status": "accepted",
01394 |             "last_rejection_reason": last_rejected_reason,
01395 |             "split_group_id": split_group_id,
01396 |             "random_cartesian_family": family_payload,
01397 |             "random_cartesian_family_id": split_group_id,
01398 |             "fdf_materialization": materialized.metadata,
01399 |         }
01400 |         write_json(sample_dir / "metadata.json", metadata)
01401 |         samples.append(
01402 |             {
01403 |                 "sample_id": sample_id,
01404 |                 "sample_dir": str(sample_dir),
01405 |                 "run_fdf": str(sample_dir / "RUN.fdf"),
01406 |                 "metadata": str(sample_dir / "metadata.json"),
01407 |                 "split_group_id": split_group_id,
01408 |                 "random_cartesian_family_id": split_group_id,
01409 |                 "base_geometry_hash": base_geometry_hash,
01410 |                 "distribution": metadata["distribution"],
01411 |                 "max_displacement_ang": metadata["max_displacement_ang"],
01412 |                 "sigma_ang": metadata["sigma_ang"],
01413 |                 "seed_family": metadata["seed_family"],
01414 |                 "family_index": family_index,
01415 |                 "variants_per_family": int(rc_config["variants_per_family"]),
01416 |                 "move_atoms": json.dumps("all", sort_keys=True),
01417 |                 "species_filter": json.dumps(metadata["species_filter"], sort_keys=True),
01418 |                 "accepted_attempt": accepted_attempt,
01419 |                 "enabled_components": json.dumps(["atom_displacement"], sort_keys=True),
01420 |                 "minimum_pair_distance_ang": metadata["minimum_pair_distance_ang"],
01421 |                 "rmsd_from_reference_ang": metadata["rmsd_from_reference_ang"],
01422 |                 "method": "random_cartesian",
01423 |                 "recipe": GENERIC_RANDOM_CARTESIAN_RECIPE,
01424 |                 "recipe_id": dataset_recipe.get("recipe_id"),
01425 |                 "global_sample_id": sample_id,
01426 |             }
01427 |         )
01428 | 
01429 |     split_summary = write_split_manifests(dataset_root, samples)
01430 |     manifest = {
01431 |         "method_id": "random_cartesian",
01432 |         "generation_method": "random_cartesian",
01433 |         "recipe": GENERIC_RANDOM_CARTESIAN_RECIPE,
01434 |         "dataset_root": str(dataset_root),
01435 |         "requested_structures": int(rc_config["n_structures"]),
01436 |         "generated_structures": len(samples),
01437 |         "total_attempts": total_attempts,
01438 |         "acceptance_ratio": (len(samples) / total_attempts) if total_attempts else 0.0,
01439 |         "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
01440 |         "seed": int(rc_config["seed"]),
01441 |         "base_geometry_hash": base_geometry_hash,
01442 |         "base_geometry_source": source_path,
01443 |         "material": resolved.to_manifest_dict(),
01444 |         "selected_atoms": selected_atoms,
01445 |         "validation": rc_config["validation"],
01446 |         "config_snapshot": public_random_cartesian_config(rc_config),
01447 |         "dataset_recipe": dataset_recipe,
01448 |         "samples": samples,
01449 |         "split_strategy": split_summary["split_strategy"],
01450 |         "split_group_key": ",".join(split_summary["split_group_keys_used"]),
01451 |         "split_summary": split_summary,
01452 |         "siesta_input_hashes": {
01453 |             sample["sample_id"]: file_sha256(Path(sample["run_fdf"]))
01454 |             for sample in samples
01455 |         },
01456 |         "basis_hashes": {
01457 |             path.name: file_sha256(path)
01458 |             for path in sorted((dataset_root / "basis").glob("*.ion.xml"))
01459 |         },
01460 |         "pseudo_hashes": {
01461 |             Path(path).name: file_sha256(Path(path))
01462 |             for path in resources["pseudopotentials"]
01463 |         },
01464 |         "matrix_file_hashes": {},
01465 |         "deterministic_hashes": {
01466 |             "base_geometry_hash": base_geometry_hash,
01467 |             "config_hash": json_sha256(public_random_cartesian_config(rc_config)),
01468 |             "sample_family_hashes": {
01469 |                 sample["sample_id"]: sample["random_cartesian_family_id"]
01470 |                 for sample in samples
01471 |             },
01472 |         },
01473 |         "scientific_warning": SCIENTIFIC_WARNING,
01474 |         "warnings": [SCIENTIFIC_WARNING, *split_summary["warnings"]],
01475 |         "severe_warnings": [],
01476 |     }
01477 |     manifest_path = dataset_root.parent / "samples_manifest.json" if output_dir is not None else PIPELINE_PATHS["samples_manifest_path"]
01478 |     write_json(dataset_root / "dataset_manifest.json", manifest)
01479 |     write_json(manifest_path, manifest)
01480 |     write_json(dataset_root / "artifact_hashes.json", artifact_hashes(dataset_root, resources["pseudopotentials"]))
01481 |     print(f"[OK] Random Cartesian dataset generado en {dataset_root}")
01482 |     return manifest
```

### `generate_dataset` — líneas 1485–1711

```py
01485 | def generate_dataset(
01486 |     config: dict[str, Any] | None = None,
01487 |     *,
01488 |     output_dir: str | Path | None = None,
01489 |     material_base_dir: str | Path = REPO_ROOT,
01490 | ) -> dict[str, Any]:
01491 |     rc_config = random_cartesian_config(config)
01492 |     if rc_config.get("recipe") == GENERIC_RANDOM_CARTESIAN_RECIPE:
01493 |         return generate_generic_random_cartesian_dataset(
01494 |             config or PIPELINE_CONFIG,
01495 |             rc_config,
01496 |             output_dir=output_dir,
01497 |             material_base_dir=material_base_dir,
01498 |         )
01499 | 
01500 |     dataset_root = Path(output_dir) if output_dir is not None else DATASET_DIR / RANDOM_CARTESIAN_STEPS_DIR_NAME
01501 |     if dataset_root.exists():
01502 |         shutil.rmtree(dataset_root)
01503 |     ensure_dir(dataset_root)
01504 |     resources = copy_required_resources(dataset_root)
01505 |     reference, source_path = load_reference_structure()
01506 |     base_geometry_hash = json_sha256(reference.to_json_dict())
01507 |     block_configs = random_cartesian_block_configs(rc_config)
01508 |     dataset_rng = random.Random(int(rc_config["seed"]))
01509 |     samples: list[dict[str, Any]] = []
01510 |     total_attempts = 0
01511 |     rejection_counts: dict[str, int] = {}
01512 | 
01513 |     print("=== Random Cartesian dataset generation ===")
01514 |     print(f"[INFO] Geometria base: {source_path}")
01515 |     print(f"[INFO] Output root: {dataset_root}")
01516 |     print(f"[INFO] n_structures: {rc_config['n_structures']}")
01517 |     print(f"[INFO] blocks: {len(block_configs)}")
01518 | 
01519 |     sample_index = 0
01520 |     for block_index, block_config in enumerate(block_configs):
01521 |         block_public_config = public_random_cartesian_config(block_config)
01522 |         block_enabled_components = enabled_component_names(block_config["components"])
01523 |         if (
01524 |             "bond_displacement" in block_enabled_components
01525 |             or "angle_displacement" in block_enabled_components
01526 |         ) and not is_h2o_structure(reference):
01527 |             raise RuntimeError(
01528 |                 "random_cartesian bond_displacement/angle_displacement requieren H2O "
01529 |                 "(exactamente 1 O y 2 H). Usa solo atom_displacement para otras moleculas."
01530 |             )
01531 |         family_payload = random_cartesian_family_payload(base_geometry_hash, block_config)
01532 |         split_group_id = json_sha256(family_payload)
01533 |         rng = random.Random(int(block_config["seed"])) if block_config.get("_seed_explicit") else dataset_rng
01534 |         print(
01535 |             "[INFO] block "
01536 |             f"{block_index + 1}/{len(block_configs)}: "
01537 |             f"{block_config.get('label')} · {block_config['n_structures']} structures"
01538 |         )
01539 |         for sample_index_within_block in range(int(block_config["n_structures"])):
01540 |             accepted: tuple[int, dict[str, Any], Structure, dict[str, Any], str | None] | None = None
01541 |             last_reason = "not_attempted"
01542 |             last_rejected_reason: str | None = None
01543 |             for attempt in range(1, int(block_config["max_attempts_per_structure"]) + 1):
01544 |                 total_attempts += 1
01545 |                 candidate, sampled_deltas = generate_candidate(reference, block_config, rng)
01546 |                 ok, reason, geometry_metrics = validate_random_structure(
01547 |                     reference,
01548 |                     candidate,
01549 |                     block_config=block_config,
01550 |                     base_geometry_hash=base_geometry_hash,
01551 |                 )
01552 |                 last_reason = reason
01553 |                 if ok:
01554 |                     accepted = (attempt, sampled_deltas, candidate, geometry_metrics, last_rejected_reason)
01555 |                     break
01556 |                 rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
01557 |                 last_rejected_reason = reason
01558 |             if accepted is None:
01559 |                 raise RuntimeError(
01560 |                     "random_cartesian could not generate a valid structure "
01561 |                     f"for block={block_config.get('block_id')} sample_index={sample_index_within_block} after "
01562 |                     f"{block_config['max_attempts_per_structure']} attempts: {last_reason}."
01563 |                 )
01564 |             accepted_attempt, sampled_deltas, candidate, geometry_metrics, last_rejected_reason = accepted
01565 |             sample_id = f"sample_{sample_index + 1:06d}"
01566 |             sample_dir = dataset_root / sample_id
01567 |             sample_dir.mkdir(parents=True, exist_ok=True)
01568 |             for pseudo in resources["pseudopotentials"]:
01569 |                 shutil.copy2(pseudo, sample_dir / Path(pseudo).name)
01570 |             single_point_config = copy.deepcopy(PIPELINE_CONFIG)
01571 |             single_point_config.setdefault("structure", {}).setdefault("force_constants", {})["enabled"] = False
01572 |             content = render_single_point_fdf(
01573 |                 single_point_config,
01574 |                 positions_ang=candidate.positions_ang,
01575 |                 atom_species=candidate.atom_species,
01576 |                 sample_id=sample_id,
01577 |             )
01578 |             (sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]).write_text(content, encoding="utf-8")
01579 |             metadata = {
01580 |                 "id": sample_id,
01581 |                 "generation_method": "random_cartesian",
01582 |                 "method": "random_cartesian",
01583 |                 "enabled_components": block_enabled_components,
01584 |                 "component_config": block_config["components"],
01585 |                 "recipe_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_id"),
01586 |                 "recipe_label": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_label"),
01587 |                 "block_id": block_config.get("block_id") or (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_id"),
01588 |                 "block_label": block_config.get("label") or (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_label"),
01589 |                 "generation_parameters_json": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("generation_parameters_json"),
01590 |                 "base_geometry_hash": base_geometry_hash,
01591 |                 "base_geometry_source": source_path,
01592 |                 "seed": int(block_config["seed"]),
01593 |                 "seed_family": int(block_config["seed"]),
01594 |                 "sample_index": sample_index,
01595 |                 "sample_index_within_block": sample_index_within_block,
01596 |                 "global_sample_id": sample_id,
01597 |                 "distribution": block_config["distribution"],
01598 |                 "sigma_ang": float(block_config["sigma_ang"]) if block_config["distribution"] == "gaussian" else None,
01599 |                 "uniform_range_ang": float(block_config["uniform_range_ang"]) if block_config["distribution"] == "uniform" else None,
01600 |                 "amplitude_ang": block_public_config.get("amplitude_ang"),
01601 |                 "move_atoms": block_config.get("move_atoms", "all"),
01602 |                 "species_filter": block_config.get("species_filter") or [],
01603 |                 "block_n_structures": int(block_config["n_structures"]),
01604 |                 "block_config": block_public_config,
01605 |                 "displacements_ang": sampled_deltas.get("atom_displacements_ang"),
01606 |                 "atom_displacements_ang": sampled_deltas.get("atom_displacements_ang"),
01607 |                 "bond_displacements_ang": sampled_deltas.get("bond_deltas") or [],
01608 |                 "angle_displacement_deg": sampled_deltas.get("angle_delta"),
01609 |                 "sampled_deltas": sampled_deltas,
01610 |                 "final_geometry_metrics": geometry_metrics,
01611 |                 "minimum_pair_distance_ang": geometry_metrics.get("minimum_pair_distance_ang"),
01612 |                 "rmsd_from_reference_ang": geometry_metrics.get("rmsd_from_reference_ang"),
01613 |                 "validation_thresholds": block_config["validation"],
01614 |                 "min_distance_ang": float(block_config["min_distance_ang"]),
01615 |                 "accepted_attempt": accepted_attempt,
01616 |                 "acceptance_status": "accepted",
01617 |                 "rejection_reason": None,
01618 |                 "last_rejection_reason": last_rejected_reason,
01619 |                 "split_group_id": split_group_id,
01620 |                 "random_cartesian_family": family_payload,
01621 |                 "random_cartesian_family_id": split_group_id,
01622 |             }
01623 |             write_json(sample_dir / "metadata.json", metadata)
01624 |             samples.append(
01625 |                 {
01626 |                     "sample_id": sample_id,
01627 |                     "sample_dir": str(sample_dir),
01628 |                     "run_fdf": str(sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]),
01629 |                     "metadata": str(sample_dir / "metadata.json"),
01630 |                     "split_group_id": split_group_id,
01631 |                     "random_cartesian_family_id": split_group_id,
01632 |                     "base_geometry_hash": base_geometry_hash,
01633 |                     "distribution": metadata.get("distribution"),
01634 |                     "sigma_ang": metadata.get("sigma_ang"),
01635 |                     "uniform_range_ang": metadata.get("uniform_range_ang"),
01636 |                     "seed_family": metadata.get("seed_family"),
01637 |                     "move_atoms": json.dumps(metadata.get("move_atoms", "all"), sort_keys=True),
01638 |                     "species_filter": json.dumps(metadata.get("species_filter", []), sort_keys=True),
01639 |                     "accepted_attempt": accepted_attempt,
01640 |                     "enabled_components": json.dumps(block_enabled_components, sort_keys=True),
01641 |                     "minimum_pair_distance_ang": metadata.get("minimum_pair_distance_ang"),
01642 |                     "rmsd_from_reference_ang": metadata.get("rmsd_from_reference_ang"),
01643 |                     "method": "random_cartesian",
01644 |                     "recipe_id": metadata.get("recipe_id"),
01645 |                     "recipe_label": metadata.get("recipe_label"),
01646 |                     "block_id": metadata.get("block_id"),
01647 |                     "block_label": metadata.get("block_label"),
01648 |                     "generation_parameters_json": metadata.get("generation_parameters_json"),
01649 |                     "sample_index_within_block": sample_index_within_block,
01650 |                     "global_sample_id": sample_id,
01651 |                 }
01652 |             )
01653 |             sample_index += 1
01654 | 
01655 |     split_summary = write_split_manifests(dataset_root, samples)
01656 |     manifest = {
01657 |         "method_id": "random_cartesian",
01658 |         "generation_method": "random_cartesian",
01659 |         "dataset_root": str(dataset_root),
01660 |         "requested_structures": int(rc_config["n_structures"]),
01661 |         "generated_structures": len(samples),
01662 |         "total_attempts": total_attempts,
01663 |         "acceptance_ratio": (len(samples) / total_attempts) if total_attempts else 0.0,
01664 |         "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
01665 |         "seed": int(rc_config["seed"]),
01666 |         "base_geometry_hash": base_geometry_hash,
01667 |         "base_geometry_source": source_path,
01668 |         "component_config": rc_config["components"],
01669 |         "validation": rc_config["validation"],
01670 |         "config_snapshot": public_random_cartesian_config(rc_config),
01671 |         "blocks": [public_random_cartesian_config(block) for block in block_configs],
01672 |         "dataset_recipe": PIPELINE_CONFIG.get("dataset_recipe") or {},
01673 |         "samples": samples,
01674 |         "split_strategy": split_summary["split_strategy"],
01675 |         "split_group_key": ",".join(split_summary["split_group_keys_used"]),
01676 |         "split_summary": split_summary,
01677 |         "siesta_input_hashes": {
01678 |             sample["sample_id"]: file_sha256(Path(sample["run_fdf"]))
01679 |             for sample in samples
01680 |         },
01681 |         "basis_hashes": {
01682 |             path.name: file_sha256(path)
01683 |             for path in sorted((dataset_root / "basis").glob("*.ion.xml"))
01684 |         },
01685 |         "pseudo_hashes": {
01686 |             Path(path).name: file_sha256(Path(path))
01687 |             for path in resources["pseudopotentials"]
01688 |         },
01689 |         "matrix_file_hashes": {},
01690 |         "deterministic_hashes": {
01691 |             "base_geometry_hash": base_geometry_hash,
01692 |             "config_hash": json_sha256(public_random_cartesian_config(rc_config)),
01693 |             "block_config_hashes": {
01694 |                 str(block.get("block_id") or index): json_sha256(public_random_cartesian_config(block))
01695 |                 for index, block in enumerate(block_configs, start=1)
01696 |             },
01697 |             "sample_family_hashes": {
01698 |                 sample["sample_id"]: sample["random_cartesian_family_id"]
01699 |                 for sample in samples
01700 |             },
01701 |         },
01702 |         "scientific_warning": SCIENTIFIC_WARNING,
01703 |         "warnings": [SCIENTIFIC_WARNING, *split_summary["warnings"]],
01704 |         "severe_warnings": [],
01705 |     }
01706 |     manifest_path = dataset_root.parent / "samples_manifest.json" if output_dir is not None else PIPELINE_PATHS["samples_manifest_path"]
01707 |     write_json(dataset_root / "dataset_manifest.json", manifest)
01708 |     write_json(manifest_path, manifest)
01709 |     write_json(dataset_root / "artifact_hashes.json", artifact_hashes(dataset_root, resources["pseudopotentials"]))
01710 |     print(f"[OK] Random Cartesian dataset generado en {dataset_root}")
01711 |     return manifest
```
