# Dossier 2C — Equivalencia de base Graph2Mat–DeepH

## Objeto de revisión

Auditar mapeo y orden orbital, bloques R, construcción H(k), overlap, shift energético, hermiticidad y evidencia necesaria para declarar equivalencia.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `Comparison/scripts/reference_selection.py`

SHA-256: `aec1cd1071cafb6ea79ba05749846ce1ac681eadce4f11e58fe22e7f8560e015`

```py
00001 | #!/usr/bin/env python3
00002 | """Strict SIESTA reference matrix selection shared by comparison scripts."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import hashlib
00007 | from dataclasses import dataclass
00008 | from pathlib import Path
00009 | 
00010 | 
00011 | MATRIX_SUFFIXES = (".TSHS", ".HSX")
00012 | REFERENCE_SELECTION_POLICY = (
00013 |     "strict_single_reference_v1: prefer exactly one non-predicted .TSHS; "
00014 |     "if no .TSHS exists, allow exactly one non-predicted .HSX; reject ambiguity."
00015 | )
00016 | 
00017 | 
00018 | @dataclass(frozen=True)
00019 | class ReferenceSelection:
00020 |     path: Path | None
00021 |     reason: str
00022 |     ambiguous: bool
00023 |     candidate_count: int
00024 |     candidates: tuple[str, ...]
00025 | 
00026 |     @property
00027 |     def ok(self) -> bool:
00028 |         return self.path is not None and self.reason == "ok"
00029 | 
00030 |     @property
00031 |     def kind(self) -> str | None:
00032 |         return self.path.suffix if self.path is not None else None
00033 | 
00034 | 
00035 | def matrix_sort_key(path: Path) -> tuple[int, str]:
00036 |     numbers: list[int] = []
00037 |     for chunk in path.stem.replace("-", ".").replace("_", ".").split("."):
00038 |         if chunk.isdigit():
00039 |             numbers.append(int(chunk))
00040 |     return (numbers[-1] if numbers else 10**9, path.name)
00041 | 
00042 | 
00043 | def is_reference_candidate(path: Path) -> bool:
00044 |     return (
00045 |         path.is_file()
00046 |         and path.suffix in MATRIX_SUFFIXES
00047 |         and "ML_prediction" not in path.name
00048 |     )
00049 | 
00050 | 
00051 | def reference_candidates(sample_dir: Path) -> list[Path]:
00052 |     if not sample_dir.exists():
00053 |         return []
00054 |     return sorted(
00055 |         [
00056 |             path
00057 |             for suffix in MATRIX_SUFFIXES
00058 |             for path in sample_dir.glob(f"*{suffix}")
00059 |             if is_reference_candidate(path)
00060 |         ],
00061 |         key=matrix_sort_key,
00062 |     )
00063 | 
00064 | 
00065 | def choose_reference_matrix(sample_dir: Path) -> ReferenceSelection:
00066 |     candidates = reference_candidates(sample_dir)
00067 |     candidate_names = tuple(path.name for path in candidates)
00068 |     if not candidates:
00069 |         return ReferenceSelection(None, "missing_reference_matrix", False, 0, candidate_names)
00070 | 
00071 |     tshs = [path for path in candidates if path.suffix == ".TSHS"]
00072 |     hsx = [path for path in candidates if path.suffix == ".HSX"]
00073 | 
00074 |     if len(tshs) == 1:
00075 |         return ReferenceSelection(tshs[0], "ok", False, len(candidates), candidate_names)
00076 |     if len(tshs) > 1:
00077 |         return ReferenceSelection(
00078 |             None,
00079 |             "ambiguous_reference_matrix_multiple_tshs",
00080 |             True,
00081 |             len(candidates),
00082 |             candidate_names,
00083 |         )
00084 |     if len(hsx) == 1:
00085 |         return ReferenceSelection(hsx[0], "ok", False, len(candidates), candidate_names)
00086 |     return ReferenceSelection(
00087 |         None,
00088 |         "ambiguous_reference_matrix_multiple_hsx",
00089 |         True,
00090 |         len(candidates),
00091 |         candidate_names,
00092 |     )
00093 | 
00094 | 
00095 | def file_sha256(path: Path | None) -> str | None:
00096 |     if path is None or not path.exists() or not path.is_file():
00097 |         return None
00098 |     digest = hashlib.sha256()
00099 |     with path.open("rb") as handle:
00100 |         for chunk in iter(lambda: handle.read(1024 * 1024), b""):
00101 |             digest.update(chunk)
00102 |     return digest.hexdigest()
```

## `tests/test_deeph_raw_global_equivalence_preflight.py`

SHA-256: `4471bb3e9a08ee271a24aa531bebc7f41ff2a7e6f4e10bc3903ef75eca4de696`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import json
00004 | import sys
00005 | import tempfile
00006 | import unittest
00007 | from pathlib import Path
00008 | 
00009 | 
00010 | REPO_ROOT = Path(__file__).resolve().parents[1]
00011 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00012 | if str(SCRIPTS_DIR) not in sys.path:
00013 |     sys.path.insert(0, str(SCRIPTS_DIR))
00014 | 
00015 | try:
00016 |     import h5py  # type: ignore[import-not-found]
00017 |     import numpy as np  # type: ignore[import-not-found]
00018 | 
00019 |     H5PY_AVAILABLE = True
00020 | except ImportError:
00021 |     h5py = None
00022 |     np = None
00023 |     H5PY_AVAILABLE = False
00024 | 
00025 | import deeph_raw_global_equivalence_preflight as preflight  # noqa: E402
00026 | from deeph_prediction_adapter import adapt_deeph_prediction_sample  # noqa: E402
00027 | 
00028 | 
00029 | def write_json(path: Path, payload: dict) -> None:
00030 |     path.parent.mkdir(parents=True, exist_ok=True)
00031 |     path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
00032 | 
00033 | 
00034 | def write_h5(path: Path, blocks: dict[str, object]) -> None:
00035 |     assert h5py is not None
00036 |     assert np is not None
00037 |     path.parent.mkdir(parents=True, exist_ok=True)
00038 |     with h5py.File(path, "w") as handle:
00039 |         for key, value in blocks.items():
00040 |             handle[key] = np.asarray(value, dtype=float)
00041 | 
00042 | 
00043 | class DeepHRawGlobalEquivalencePreflightMissingMappingTests(unittest.TestCase):
00044 |     def test_missing_sample_mapping_writes_failed_evidence_without_numeric_dependencies(self) -> None:
00045 |         with tempfile.TemporaryDirectory() as tmp:
00046 |             root = Path(tmp)
00047 |             sample = root / "sample0"
00048 |             sample.mkdir()
00049 |             (sample / "reference.HSX").write_text("reference-placeholder\n", encoding="utf-8")
00050 |             frozen = root / "frozen_split_manifest.json"
00051 |             write_json(
00052 |                 frozen,
00053 |                 {
00054 |                     "rows": [
00055 |                         {
00056 |                             "sample_id": "sample0",
00057 |                             "split": "test",
00058 |                             "sample_dir": str(sample),
00059 |                             "artifact_paths": {"reference_hsx": str(sample / "reference.HSX")},
00060 |                         }
00061 |                     ]
00062 |                 },
00063 |             )
00064 | 
00065 |             manifest = preflight.build_preflight_manifest(
00066 |                 frozen_split_manifest=frozen,
00067 |                 graph2mat_result_dir=root / "g2m",
00068 |                 deeph_processed_dir=root / "processed",
00069 |                 deeph_predictions_dir=root / "predictions",
00070 |                 output_dir=root / "equivalence",
00071 |                 sample_limit=5,
00072 |                 command=["unit"],
00073 |             )
00074 | 
00075 |             self.assertEqual(manifest["status"], "failed")
00076 |             evidence = json.loads(
00077 |                 (root / "equivalence" / "sample0" / "raw_global_equivalence_evidence.json").read_text(
00078 |                     encoding="utf-8"
00079 |                 )
00080 |             )
00081 |             self.assertEqual(evidence["equivalence_status"], "failed")
00082 |             self.assertIn("missing DeepH processed sample mapping", evidence["failure_reason"])
00083 | 
00084 | 
00085 | @unittest.skipUnless(H5PY_AVAILABLE, "h5py/numpy are required for numeric preflight tests")
00086 | class DeepHRawGlobalEquivalencePreflightNumericTests(unittest.TestCase):
00087 |     def setUp(self) -> None:
00088 |         self.tmp = tempfile.TemporaryDirectory()
00089 |         self.root = Path(self.tmp.name)
00090 |         self.sample = self.root / "sample0"
00091 |         self.processed = self.root / "processed" / "sample0"
00092 |         self.predictions = self.root / "predictions" / "sample0"
00093 |         self.output = self.root / "equivalence"
00094 |         self.sample.mkdir(parents=True)
00095 |         self.processed.mkdir(parents=True)
00096 |         self.predictions.mkdir(parents=True)
00097 |         self.reference = self.sample / "reference.HSX"
00098 |         self.reference.write_text("reference-placeholder\n", encoding="utf-8")
00099 |         self.run_fdf = self.sample / "RUN.fdf"
00100 |         self.run_fdf.write_text("SystemLabel graphene\n", encoding="utf-8")
00101 |         self.frozen = self.root / "frozen_split_manifest.json"
00102 |         write_json(
00103 |             self.frozen,
00104 |             {
00105 |                 "rows": [
00106 |                     {
00107 |                         "sample_id": "sample0",
00108 |                         "split": "test",
00109 |                         "sample_dir": str(self.sample),
00110 |                         "artifact_paths": {
00111 |                             "reference_hsx": str(self.reference),
00112 |                             "run_fdf": str(self.run_fdf),
00113 |                         },
00114 |                     }
00115 |                 ]
00116 |             },
00117 |         )
00118 |         self._old_raw_reference_matrices = preflight.raw_reference_matrices
00119 |         self._old_runtime_helpers = preflight._runtime_helpers
00120 |         self._old_kpoints_from_fdf = preflight.kpoints_from_fdf
00121 |         preflight.raw_reference_matrices = self.fake_raw_reference_matrices  # type: ignore[method-assign]
00122 |         preflight._runtime_helpers = self.fake_runtime_helpers  # type: ignore[method-assign]
00123 |         preflight.kpoints_from_fdf = lambda _path: ([(0.0, 0.0, 0.0)], [])  # type: ignore[assignment]
00124 | 
00125 |     def tearDown(self) -> None:
00126 |         preflight.raw_reference_matrices = self._old_raw_reference_matrices  # type: ignore[assignment]
00127 |         preflight._runtime_helpers = self._old_runtime_helpers  # type: ignore[assignment]
00128 |         preflight.kpoints_from_fdf = self._old_kpoints_from_fdf  # type: ignore[assignment]
00129 |         self.tmp.cleanup()
00130 | 
00131 |     def fake_raw_reference_matrices(self, _reference_path, _kpoint):
00132 |         assert np is not None
00133 |         return {
00134 |             "hamiltonian": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
00135 |             "overlap": np.eye(2, dtype=np.complex128),
00136 |             "spin": "",
00137 |             "orthogonal": False,
00138 |         }
00139 | 
00140 |     def fake_runtime_helpers(self):
00141 |         assert np is not None
00142 | 
00143 |         def eigenvalues(hamiltonian, _overlap):
00144 |             return np.linalg.eigvalsh(np.asarray(hamiltonian, dtype=np.complex128))
00145 | 
00146 |         return {"np": np, "complex_generalized_eigenvalues": eigenvalues}
00147 | 
00148 |     def write_processed(self, *, scale: float = 1.0, shape_mismatch: bool = False, include_overlap: bool = True) -> None:
00149 |         (self.processed / "orbital_types.dat").write_text("0\n0\n", encoding="utf-8")
00150 |         write_json(self.processed / "info.json", {"isspinful": False, "isorthogonal": False})
00151 |         h_blocks = {
00152 |             "[0, 0, 0, 1, 1]": [[0.0]],
00153 |             "[0, 0, 0, 1, 2]": [[scale, scale]] if shape_mismatch else [[scale]],
00154 |             "[0, 0, 0, 2, 1]": [[scale]],
00155 |             "[0, 0, 0, 2, 2]": [[0.0]],
00156 |         }
00157 |         write_h5(self.processed / "hamiltonians.h5", h_blocks)
00158 |         if include_overlap:
00159 |             write_h5(
00160 |                 self.processed / "overlaps.h5",
00161 |                 {
00162 |                     "[0, 0, 0, 1, 1]": [[1.0]],
00163 |                     "[0, 0, 0, 1, 2]": [[0.0]],
00164 |                     "[0, 0, 0, 2, 1]": [[0.0]],
00165 |                     "[0, 0, 0, 2, 2]": [[1.0]],
00166 |                 },
00167 |             )
00168 |         pred_blocks = {
00169 |             "[0, 0, 0, 1, 1]": [[0.0]],
00170 |             "[0, 0, 0, 1, 2]": [[scale]],
00171 |             "[0, 0, 0, 2, 1]": [[scale]],
00172 |             "[0, 0, 0, 2, 2]": [[0.0]],
00173 |         }
00174 |         write_h5(self.predictions / "hamiltonians_pred.h5", pred_blocks)
00175 | 
00176 |     def run_preflight(self) -> dict:
00177 |         return preflight.build_preflight_manifest(
00178 |             frozen_split_manifest=self.frozen,
00179 |             graph2mat_result_dir=self.root / "g2m",
00180 |             deeph_processed_dir=self.root / "processed",
00181 |             deeph_predictions_dir=self.root / "predictions",
00182 |             output_dir=self.output,
00183 |             sample_limit=5,
00184 |             command=["unit"],
00185 |         )
00186 | 
00187 |     def test_exact_synthetic_equivalence_passes_and_adapter_accepts_evidence(self) -> None:
00188 |         self.write_processed()
00189 | 
00190 |         manifest = self.run_preflight()
00191 |         result = adapt_deeph_prediction_sample(
00192 |             work_dir=self.predictions,
00193 |             processed_sample_dir=self.processed,
00194 |             sample_id="sample0",
00195 |         )
00196 | 
00197 |         self.assertEqual(manifest["status"], "proven")
00198 |         self.assertFalse(result.diagnostic_only)
00199 |         self.assertTrue(result.metric_fields()["deeph_raw_global_equivalence_proven"])
00200 | 
00201 |     def test_siesta_orbital_mapping_signs_and_energy_shift_pass(self) -> None:
00202 |         assert np is not None
00203 |         shift_eV = 2.5
00204 |         raw_h = np.asarray(
00205 |             [
00206 |                 [1.2, 0.4, -0.3, 0.2],
00207 |                 [0.4, -0.7, 0.5, -0.6],
00208 |                 [-0.3, 0.5, 0.9, 0.1],
00209 |                 [0.2, -0.6, 0.1, -1.1],
00210 |             ],
00211 |             dtype=float,
00212 |         )
00213 |         raw_s = np.asarray(
00214 |             [
00215 |                 [1.0, 0.03, -0.02, 0.04],
00216 |                 [0.03, 1.1, 0.05, -0.01],
00217 |                 [-0.02, 0.05, 0.9, 0.02],
00218 |                 [0.04, -0.01, 0.02, 1.2],
00219 |             ],
00220 |             dtype=float,
00221 |         )
00222 | 
00223 |         def raw_reference(_reference_path, _kpoint):
00224 |             return {
00225 |                 "hamiltonian": raw_h,
00226 |                 "overlap": raw_s,
00227 |                 "spin": "",
00228 |                 "orthogonal": False,
00229 |             }
00230 | 
00231 |         preflight.raw_reference_matrices = raw_reference  # type: ignore[assignment]
00232 |         orb_indx = self.sample / "graphene.ORB_INDX"
00233 |         orb_indx.write_text(
00234 |             "\n".join(
00235 |                 [
00236 |                     "1 1 1 C 1 0 0 0 0 0 s 0.0 0 0 0 1",
00237 |                     "2 1 1 C 1 0 1 -1 0 0 py 0.0 0 0 0 2",
00238 |                     "3 1 1 C 1 0 1 0 0 0 pz 0.0 0 0 0 3",
00239 |                     "4 1 1 C 1 0 1 1 0 0 px 0.0 0 0 0 4",
00240 |                 ]
00241 |             )
00242 |             + "\n",
00243 |             encoding="utf-8",
00244 |         )
00245 |         write_json(
00246 |             self.frozen,
00247 |             {
00248 |                 "rows": [
00249 |                     {
00250 |                         "sample_id": "sample0",
00251 |                         "split": "test",
00252 |                         "sample_dir": str(self.sample),
00253 |                         "artifact_paths": {
00254 |                             "reference_hsx": str(self.reference),
00255 |                             "run_fdf": str(self.run_fdf),
00256 |                             "orb_indx": str(orb_indx),
00257 |                         },
00258 |                     }
00259 |                 ]
00260 |             },
00261 |         )
00262 |         (self.processed / "orbital_types.dat").write_text("0 1\n", encoding="utf-8")
00263 |         write_json(self.processed / "info.json", {"isspinful": False, "isorthogonal": False})
00264 | 
00265 |         # SIESTA order is s,py,pz,px. DeepH processed blocks are s,px,py,pz.
00266 |         # The processed Hamiltonian also uses an energy zero shifted by c*S.
00267 |         permutation = [0, 2, 3, 1]
00268 |         signs = np.asarray([1.0, -1.0, 1.0, -1.0])
00269 |         converted_h = raw_h - shift_eV * raw_s
00270 |         converted_s = raw_s
00271 |         deeph_h = np.zeros_like(converted_h)
00272 |         deeph_s = np.zeros_like(converted_s)
00273 |         for siesta_i, deeph_i in enumerate(permutation):
00274 |             for siesta_j, deeph_j in enumerate(permutation):
00275 |                 deeph_h[deeph_i, deeph_j] = signs[siesta_i] * converted_h[siesta_i, siesta_j] * signs[siesta_j]
00276 |                 deeph_s[deeph_i, deeph_j] = signs[siesta_i] * converted_s[siesta_i, siesta_j] * signs[siesta_j]
00277 |         blocks = {"[0, 0, 0, 1, 1]": deeph_h}
00278 |         write_h5(self.processed / "hamiltonians.h5", blocks)
00279 |         write_h5(self.processed / "overlaps.h5", {"[0, 0, 0, 1, 1]": deeph_s})
00280 |         write_h5(self.predictions / "hamiltonians_pred.h5", blocks)
00281 | 
00282 |         manifest = self.run_preflight()
00283 |         evidence = json.loads((self.predictions / "raw_global_equivalence_evidence.json").read_text(encoding="utf-8"))
00284 | 
00285 |         self.assertEqual(manifest["status"], "proven")
00286 |         self.assertEqual(evidence["equivalence_status"], "proven")
00287 |         self.assertEqual(evidence["basis_transform"]["status"], "applied")
00288 |         self.assertEqual(evidence["basis_transform"]["permutation"], permutation)
00289 |         self.assertEqual(evidence["basis_transform"]["signs"], signs.tolist())
00290 |         self.assertAlmostEqual(evidence["energy_reference_alignment"]["shift_eV"], shift_eV, places=12)
00291 | 
00292 |     def test_shape_mismatch_fails(self) -> None:
00293 |         self.write_processed(shape_mismatch=True)
00294 | 
00295 |         manifest = self.run_preflight()
00296 | 
00297 |         self.assertEqual(manifest["status"], "failed")
00298 |         evidence = json.loads((self.predictions / "raw_global_equivalence_evidence.json").read_text(encoding="utf-8"))
00299 |         self.assertEqual(evidence["equivalence_status"], "failed")
00300 | 
00301 |     def test_unit_scale_mismatch_fails_and_adapter_remains_diagnostic(self) -> None:
00302 |         self.write_processed(scale=2.0)
00303 | 
00304 |         manifest = self.run_preflight()
00305 |         result = adapt_deeph_prediction_sample(
00306 |             work_dir=self.predictions,
00307 |             processed_sample_dir=self.processed,
00308 |             sample_id="sample0",
00309 |         )
00310 | 
00311 |         self.assertEqual(manifest["status"], "failed")
00312 |         self.assertTrue(result.diagnostic_only)
00313 |         self.assertFalse(result.metric_fields()["deeph_raw_global_equivalence_proven"])
00314 | 
00315 |     def test_missing_overlap_fails(self) -> None:
00316 |         self.write_processed(include_overlap=False)
00317 | 
00318 |         manifest = self.run_preflight()
00319 | 
00320 |         self.assertEqual(manifest["status"], "failed")
00321 |         evidence = json.loads((self.predictions / "raw_global_equivalence_evidence.json").read_text(encoding="utf-8"))
00322 |         self.assertIn("overlaps.h5", evidence["failure_reason"])
00323 | 
00324 | 
00325 | if __name__ == "__main__":
00326 |     unittest.main()
```

## `tests/test_method_provenance_fairness.py`

SHA-256: `186780e2e8389ed769bca41b0efc150e3280151ece424c8af91cb4f0ebf9cba9`

```py
00001 | import copy
00002 | import importlib.util
00003 | import json
00004 | import sys
00005 | import tempfile
00006 | import unittest
00007 | from pathlib import Path
00008 | 
00009 | 
00010 | REPO_ROOT = Path(__file__).resolve().parents[1]
00011 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00012 | 
00013 | 
00014 | def load_script_module(name: str, relative_path: str):
00015 |     sys.path.insert(0, str(SCRIPTS_DIR))
00016 |     spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / relative_path)
00017 |     assert spec and spec.loader
00018 |     module = importlib.util.module_from_spec(spec)
00019 |     sys.modules[spec.name] = module
00020 |     spec.loader.exec_module(module)
00021 |     return module
00022 | 
00023 | 
00024 | def load_repo_script_module(name: str, relative_path: str, *extra_paths: Path):
00025 |     for path in (SCRIPTS_DIR, *extra_paths):
00026 |         path_text = str(path)
00027 |         if path_text not in sys.path:
00028 |             sys.path.insert(0, path_text)
00029 |     spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
00030 |     assert spec and spec.loader
00031 |     module = importlib.util.module_from_spec(spec)
00032 |     sys.modules[spec.name] = module
00033 |     spec.loader.exec_module(module)
00034 |     return module
00035 | 
00036 | 
00037 | def siesta_fixtures() -> tuple[dict, dict, dict, dict]:
00038 |     shared = {"MeshCutoff": "200 Ry"}
00039 |     md = {
00040 |         "md": {
00041 |             "lattice_constant": 15,
00042 |             "lattice_vectors": [[15, 0, 0], [0, 15, 0], [0, 0, 15]],
00043 |             "basis_type": "split",
00044 |             "basis_size": "DZP",
00045 |             "energy_shift": "0.03 eV",
00046 |             "mesh_cutoff": "200 Ry",
00047 |             "xc_functional": "GGA",
00048 |             "xc_authors": "PBE",
00049 |             "max_scf_iterations": 200,
00050 |             "solution_method": "diagon",
00051 |             "dm_mixing_weight": 0.02,
00052 |             "dm_number_pulay": 3,
00053 |             "dm_tolerance": "1.d-5",
00054 |             "dm_require_energy_convergence": "T",
00055 |             "dm_energy_tolerance": "1.e-5 eV",
00056 |             "spin_polarized": "F",
00057 |             "fix_spin": "F",
00058 |             "non_collinear_spin": "F",
00059 |             "force_aux_cell": False,
00060 |             "save_hs_file": True,
00061 |             "save_hs": True,
00062 |             "save_de": True,
00063 |             "xml_write": True,
00064 |         }
00065 |     }
00066 |     fc = {
00067 |         "structure": {
00068 |             "lattice_constant": 15,
00069 |             "lattice_vectors": [[15, 0, 0], [0, 15, 0], [0, 0, 15]],
00070 |             "force_constants": {"save_tshs": True, "save_tsde": True},
00071 |             "siesta": {
00072 |                 "ForceAuxCell": "F",
00073 |                 "Save.HS": "T",
00074 |                 "MeshCutoff": "200 Ry",
00075 |                 "PAO.BasisType": "split",
00076 |                 "PAO.BasisSize": "DZP",
00077 |                 "PAO.EnergyShift": "0.03 eV",
00078 |                 "XC.functional": "GGA",
00079 |                 "XC.authors": "PBE",
00080 |                 "MaxSCFIterations": 200,
00081 |                 "SolutionMethod": "diagon",
00082 |                 "DM.MixingWeight": 0.02,
00083 |                 "DM.NumberPulay": 3,
00084 |                 "DM.Tolerance": "1.d-5",
00085 |                 "DM.Require.Energy.Convergence": "T",
00086 |                 "DM.Energy.Tolerance": "1.e-5 eV",
00087 |                 "SpinPolarized": "F",
00088 |                 "FixSpin": "F",
00089 |                 "NonCollinearSpin": "F",
00090 |                 "XML.Write": "T",
00091 |             },
00092 |         }
00093 |     }
00094 |     rc = copy.deepcopy(fc)
00095 |     rc["structure"]["random_cartesian"] = {"siesta": {}}
00096 |     return shared, md, fc, rc
00097 | 
00098 | 
00099 | def model_fixtures() -> tuple[dict, dict, dict]:
00100 |     training = {
00101 |         "torch_float32_matmul_precision": "high",
00102 |         "data": {
00103 |             "out_matrix": "hamiltonian",
00104 |             "symmetric_matrix": True,
00105 |             "sub_point_matrix": False,
00106 |             "matrix_component_policy": "h_only",
00107 |             "n_matrix_components": 1,
00108 |             "basis_files": "../dataset/basis/*.ion.xml",
00109 |             "train_runs": "../dataset/train/*/RUN.fdf",
00110 |             "batch_size": 8,
00111 |             "store_in_memory": True,
00112 |         },
00113 |         "model": {
00114 |             "num_interactions": 1,
00115 |             "correlation": 1,
00116 |             "max_ell": 2,
00117 |             "hidden_irreps": "10x0e + 10x1o + 10x2e",
00118 |             "loss": "graph2mat.metrics.block_type_mae",
00119 |             "optim_lr": 0.005,
00120 |         },
00121 |         "trainer": {
00122 |             "accelerator": "cpu",
00123 |             "logger": {
00124 |                 "class_path": "TensorBoardLogger",
00125 |                 "init_args": {"name": "method_model", "save_dir": "lightning_logs"},
00126 |             },
00127 |             "max_epochs": 100,
00128 |         },
00129 |     }
00130 |     md = {"training": copy.deepcopy(training)}
00131 |     fc = {"training": copy.deepcopy(training)}
00132 |     rc = {"training": copy.deepcopy(training)}
00133 |     return md, fc, rc
00134 | 
00135 | 
00136 | def provenance_run(root: Path, method: str, label: str, size: int) -> dict:
00137 |     result_dir = root / method / label
00138 |     split_dir = result_dir / "splits"
00139 |     split_dir.mkdir(parents=True)
00140 |     for split in ("train", "validation", "test"):
00141 |         (split_dir / f"{split}_manifest.csv").write_text("sample_id\nsample_1\n", encoding="utf-8")
00142 |     checkpoint_path = result_dir / "training" / "best.ckpt"
00143 |     checkpoint_path.parent.mkdir(parents=True)
00144 |     checkpoint_path.write_bytes(b"checkpoint")
00145 |     return {
00146 |         "method_id": method,
00147 |         "pipeline": "atom_displacement" if method == "siesta_fc_cartesian" else method,
00148 |         "dataset_label": label,
00149 |         "dataset_size": size,
00150 |         "effective_dataset_size": size,
00151 |         "result_dir": str(result_dir),
00152 |         "model_checkpoint": str(checkpoint_path),
00153 |         "model_checkpoint_sha256": f"{method}_checkpoint_hash",
00154 |         "checkpoint_manifest": str(result_dir / "training" / "checkpoint_manifest.json"),
00155 |         "checkpoint_selection_warning": "",
00156 |         "artifact_hashes": {
00157 |             "basis": f"{method}_basis_hash",
00158 |             "pseudopotentials": f"{method}_pseudo_hash",
00159 |         },
00160 |         "recipe_id": f"{method}_recipe",
00161 |         "recipe_label": f"{method} recipe",
00162 |         "recipe_set_hash": f"{method}_recipe_hash",
00163 |         "returncode": 0,
00164 |         "seed": 42,
00165 |     }
00166 | 
00167 | 
00168 | class MethodProvenanceFairnessTests(unittest.TestCase):
00169 |     def compare_siesta(self, configs: dict[str, dict], *, artifacts: dict[str, dict[str, str]] | None = None):
00170 |         module = load_script_module("siesta_settings_method_provenance_tests", "siesta_settings.py")
00171 |         shared, _, _, _ = siesta_fixtures()
00172 |         return module.compare_method_settings(
00173 |             configs,
00174 |             shared,
00175 |             artifact_hashes_by_method=artifacts,
00176 |             selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
00177 |         )
00178 | 
00179 |     def compare_models(self, configs: dict[str, dict]):
00180 |         module = load_script_module("model_settings_method_provenance_tests", "model_settings.py")
00181 |         return module.compare_method_model_settings(
00182 |             configs,
00183 |             selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
00184 |         )
00185 | 
00186 |     def test_equivalent_siesta_settings_have_hashes_and_no_severe_warning(self) -> None:
00187 |         _, md, fc, rc = siesta_fixtures()
00188 |         artifacts = {
00189 |             method: {"basis_hash": "basis_same", "pseudopotential_hash": "pseudo_same"}
00190 |             for method in ("md", "siesta_fc_cartesian", "random_cartesian")
00191 |         }
00192 |         report = self.compare_siesta(
00193 |             {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
00194 |             artifacts=artifacts,
00195 |         )
00196 |         self.assertTrue(report["ok"])
00197 |         self.assertEqual(report["pairwise_mismatch_report"], [])
00198 |         self.assertEqual(report["severe_mismatches"], [])
00199 |         self.assertEqual(
00200 |             set(report["siesta_settings_hash_by_method"]),
00201 |             {"md", "siesta_fc_cartesian", "random_cartesian"},
00202 |         )
00203 |         self.assertTrue(all(report["siesta_settings_hash_by_method"].values()))
00204 |         self.assertEqual(report["basis_hash_by_method"]["random_cartesian"], "basis_same")
00205 | 
00206 |     def test_random_cartesian_meshcutoff_mismatch_is_severe(self) -> None:
00207 |         _, md, fc, rc = siesta_fixtures()
00208 |         rc["structure"]["random_cartesian"]["siesta"]["MeshCutoff"] = "300 Ry"
00209 |         report = self.compare_siesta({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
00210 |         self.assertFalse(report["ok"])
00211 |         self.assertTrue(report["severe_warning"])
00212 |         self.assertTrue(
00213 |             any(
00214 |                 mismatch["key"] == "MeshCutoff" and "random_cartesian" in mismatch["methods"]
00215 |                 for mismatch in report["severe_mismatches"]
00216 |             )
00217 |         )
00218 | 
00219 |     def test_hamiltonian_output_flag_mismatch_is_method_provenance_severe(self) -> None:
00220 |         _, md, fc, rc = siesta_fixtures()
00221 |         fc["structure"]["siesta"]["Save.HS"] = "F"
00222 |         report = self.compare_siesta({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
00223 |         self.assertFalse(report["ok"])
00224 |         severe = [
00225 |             mismatch
00226 |             for mismatch in report["severe_mismatches"]
00227 |             if mismatch["key"] == "Save.HS"
00228 |         ]
00229 |         self.assertTrue(severe)
00230 |         self.assertTrue(all(mismatch["severity"] == "severe" for mismatch in severe))
00231 | 
00232 |         module = load_script_module("pipeline_ui_output_flag_provenance_tests", "pipeline_ui.py")
00233 |         with tempfile.TemporaryDirectory() as tmp:
00234 |             root = Path(tmp)
00235 |             manifest = {
00236 |                 "experiment_id": "output_flag_mismatch_case",
00237 |                 "selected_methods": ["md", "siesta_fc_cartesian"],
00238 |                 "runs": [
00239 |                     provenance_run(root, "md", "md_190", 190),
00240 |                     provenance_run(root, "siesta_fc_cartesian", "fc_190", 190),
00241 |                 ],
00242 |                 "siesta_settings_hash_by_method": {
00243 |                     "md": "siesta_md",
00244 |                     "siesta_fc_cartesian": "siesta_fc",
00245 |                 },
00246 |                 "model_config_hash_by_method": {
00247 |                     "md": "model_md",
00248 |                     "siesta_fc_cartesian": "model_fc",
00249 |                 },
00250 |                 "basis_hash_by_method": {
00251 |                     "md": "basis_md",
00252 |                     "siesta_fc_cartesian": "basis_fc",
00253 |                 },
00254 |                 "pseudopotential_hash_by_method": {
00255 |                     "md": "pseudo_md",
00256 |                     "siesta_fc_cartesian": "pseudo_fc",
00257 |                 },
00258 |                 "siesta_settings_severe_mismatches": severe,
00259 |             }
00260 |             module.refresh_method_provenance(manifest)
00261 | 
00262 |         self.assertTrue(
00263 |             any("Save.HS" in warning for warning in manifest["method_provenance_severe_warnings"])
00264 |         )
00265 | 
00266 |     def test_fc_basis_hash_mismatch_is_severe(self) -> None:
00267 |         _, md, fc, rc = siesta_fixtures()
00268 |         artifacts = {
00269 |             "md": {"basis_hash": "basis_same", "pseudopotential_hash": "pseudo_same"},
00270 |             "siesta_fc_cartesian": {"basis_hash": "basis_fc_different", "pseudopotential_hash": "pseudo_same"},
00271 |             "random_cartesian": {"basis_hash": "basis_same", "pseudopotential_hash": "pseudo_same"},
00272 |         }
00273 |         report = self.compare_siesta(
00274 |             {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
00275 |             artifacts=artifacts,
00276 |         )
00277 |         severe = [
00278 |             mismatch
00279 |             for mismatch in report["severe_mismatches"]
00280 |             if mismatch["type"] == "basis_pseudopotential" and mismatch["key"] == "basis_hash"
00281 |         ]
00282 |         self.assertFalse(report["ok"])
00283 |         self.assertTrue(severe)
00284 |         self.assertTrue(any("siesta_fc_cartesian" in mismatch["methods"] for mismatch in severe))
00285 |         self.assertTrue(report["basis_pseudopotential_warning"])
00286 | 
00287 |     def test_random_cartesian_graph2mat_architecture_mismatch_is_severe(self) -> None:
00288 |         md, fc, rc = model_fixtures()
00289 |         rc["training"]["model"]["num_interactions"] = 2
00290 |         report = self.compare_models({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
00291 |         severe = [
00292 |             mismatch
00293 |             for mismatch in report["severe_mismatches"]
00294 |             if mismatch["section"] == "model" and mismatch["key"] == "num_interactions"
00295 |         ]
00296 |         self.assertFalse(report["ok"])
00297 |         self.assertTrue(severe)
00298 |         self.assertTrue(any("random_cartesian" in mismatch["methods"] for mismatch in severe))
00299 | 
00300 |     def test_fc_graph2mat_loss_mismatch_is_severe(self) -> None:
00301 |         md, fc, rc = model_fixtures()
00302 |         fc["training"]["model"]["loss"] = "graph2mat.metrics.mse"
00303 |         report = self.compare_models({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
00304 |         severe = [
00305 |             mismatch
00306 |             for mismatch in report["severe_mismatches"]
00307 |             if mismatch["section"] == "model" and mismatch["key"] == "loss"
00308 |         ]
00309 |         self.assertFalse(report["ok"])
00310 |         self.assertTrue(severe)
00311 |         self.assertTrue(any("siesta_fc_cartesian" in mismatch["methods"] for mismatch in severe))
00312 | 
00313 |     def test_dataset_path_only_model_differences_have_no_severe_warning(self) -> None:
00314 |         md, fc, rc = model_fixtures()
00315 |         md["training"]["data"]["dataset_path"] = "/tmp/md/dataset"
00316 |         fc["training"]["data"]["dataset_path"] = "/tmp/fc/dataset"
00317 |         rc["training"]["data"]["dataset_path"] = "/tmp/rc/dataset"
00318 |         md["training"]["trainer"]["default_root_dir"] = "/tmp/md/out"
00319 |         fc["training"]["trainer"]["default_root_dir"] = "/tmp/fc/out"
00320 |         rc["training"]["trainer"]["default_root_dir"] = "/tmp/rc/out"
00321 |         report = self.compare_models({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
00322 |         self.assertTrue(report["ok"])
00323 |         self.assertEqual(report["pairwise_mismatch_report"], [])
00324 |         self.assertEqual(report["severe_mismatches"], [])
00325 |         self.assertEqual(
00326 |             set(report["model_config_hash_by_method"]),
00327 |             {"md", "siesta_fc_cartesian", "random_cartesian"},
00328 |         )
00329 |         self.assertEqual(len(set(report["model_config_hash_by_method"].values())), 1)
00330 | 
00331 |     def test_final_manifest_method_provenance_includes_three_explicit_methods(self) -> None:
00332 |         module = load_script_module("pipeline_ui_method_provenance_tests", "pipeline_ui.py")
00333 |         with tempfile.TemporaryDirectory() as tmp:
00334 |             root = Path(tmp)
00335 |             runs = [
00336 |                 provenance_run(root, "md", "md_190", 190),
00337 |                 provenance_run(root, "siesta_fc_cartesian", "fc_190", 190),
00338 |                 provenance_run(root, "random_cartesian", "rc_3500", 3500),
00339 |             ]
00340 |             manifest = {
00341 |                 "experiment_id": "method_provenance_case",
00342 |                 "selected_methods": ["md", "siesta_fc_cartesian", "random_cartesian"],
00343 |                 "runs": runs,
00344 |                 "siesta_settings_hash_by_method": {
00345 |                     "md": "siesta_md",
00346 |                     "siesta_fc_cartesian": "siesta_fc",
00347 |                     "random_cartesian": "siesta_rc",
00348 |                 },
00349 |                 "model_config_hash_by_method": {
00350 |                     "md": "model_md",
00351 |                     "siesta_fc_cartesian": "model_fc",
00352 |                     "random_cartesian": "model_rc",
00353 |                 },
00354 |                 "basis_hash_by_method": {
00355 |                     "md": "basis_md",
00356 |                     "siesta_fc_cartesian": "basis_fc",
00357 |                     "random_cartesian": "basis_rc",
00358 |                 },
00359 |                 "pseudopotential_hash_by_method": {
00360 |                     "md": "pseudo_md",
00361 |                     "siesta_fc_cartesian": "pseudo_fc",
00362 |                     "random_cartesian": "pseudo_rc",
00363 |                 },
00364 |             }
00365 |             module.refresh_method_provenance(manifest)
00366 | 
00367 |         provenance = manifest["method_provenance"]
00368 |         self.assertEqual(set(provenance), {"md", "siesta_fc_cartesian", "random_cartesian"})
00369 |         self.assertNotIn("atom_displacement", provenance)
00370 |         self.assertEqual(provenance["random_cartesian"]["dataset_label"], "rc_3500")
00371 |         self.assertEqual(provenance["random_cartesian"]["siesta_settings_hash"], "siesta_rc")
00372 |         self.assertEqual(provenance["random_cartesian"]["model_settings_hash"], "model_rc")
00373 |         self.assertEqual(provenance["random_cartesian"]["basis_hash"], "basis_rc")
00374 |         self.assertEqual(provenance["random_cartesian"]["pseudopotential_hash"], "pseudo_rc")
00375 |         self.assertFalse(manifest["method_provenance_severe_warnings"])
00376 | 
00377 |     def test_method_provenance_warnings_are_aggregated_as_blocking_warnings(self) -> None:
00378 |         module = load_script_module("aggregate_cross_method_provenance_warning_tests", "aggregate_cross_metrics.py")
00379 |         with tempfile.TemporaryDirectory() as tmp:
00380 |             result_dir = Path(tmp) / "md__on__test_md"
00381 |             metrics_dir = result_dir / "metrics"
00382 |             metrics_dir.mkdir(parents=True)
00383 |             (metrics_dir / "sparse_metrics.csv").write_text(
00384 |                 "sample,relative_frobenius_union\nsample_1,0.2\n",
00385 |                 encoding="utf-8",
00386 |             )
00387 |             (metrics_dir / "spectral_metrics.csv").write_text(
00388 |                 "sample,low_energy_rmse_eV\nsample_1,0.1\n",
00389 |                 encoding="utf-8",
00390 |             )
00391 |             (result_dir / "cross_evaluation_manifest.json").write_text(
00392 |                 json.dumps(
00393 |                     {
00394 |                         "train_method": "md",
00395 |                         "test_set": "test_md",
00396 |                         "method_provenance_warnings": ["md: Missing SIESTA settings hash."],
00397 |                         "method_provenance_severe_warnings": [
00398 |                             "md: Severe SIESTA settings mismatch: Save.HS"
00399 |                         ],
00400 |                     }
00401 |                 ),
00402 |                 encoding="utf-8",
00403 |             )
00404 | 
00405 |             rows = module.aggregate_one(result_dir, "warning_case")
00406 | 
00407 |         self.assertEqual(len(rows), 1)
00408 |         self.assertIn("Missing SIESTA settings hash", rows[0]["severe_warnings"])
00409 |         self.assertIn("Save.HS", rows[0]["severe_warnings"])
00410 |         self.assertIn("Missing SIESTA settings hash", rows[0]["method_provenance_warnings"])
00411 |         self.assertIn("Save.HS", rows[0]["method_provenance_severe_warnings"])
00412 | 
00413 |     def test_winner_analysis_treats_method_provenance_warning_as_severe(self) -> None:
00414 |         module = load_script_module("analyze_winners_method_provenance_warning_tests", "analyze_winners.py")
00415 |         rows = [
00416 |             {
00417 |                 "train_method": "md",
00418 |                 "test_set": "test_md",
00419 |                 "low_energy_rmse_eV": 0.2,
00420 |                 "method_provenance_warnings": "md: Missing SIESTA settings hash.",
00421 |                 "seed": 1,
00422 |             },
00423 |             {
00424 |                 "train_method": "siesta_fc_cartesian",
00425 |                 "test_set": "test_md",
00426 |                 "low_energy_rmse_eV": 0.3,
00427 |                 "seed": 1,
00428 |             },
00429 |         ]
00430 |         recommendation = module.build_recommendation(
00431 |             rows,
00432 |             summary_rows=[],
00433 |             pair_rows=[],
00434 |             primary_metric="low_energy_rmse_eV",
00435 |         )
00436 | 
00437 |         self.assertIn("Missing SIESTA settings hash", " | ".join(recommendation["severe_warnings"]))
00438 |         self.assertNotEqual(recommendation["scientific_status"], "robust_comparison")
00439 |         self.assertIsNone(recommendation["winner"])
00440 | 
00441 |     def test_md_blocked_split_has_temporal_gap_between_partitions(self) -> None:
00442 |         module = load_repo_script_module(
00443 |             "md_blocked_split_fairness_tests",
00444 |             "MD/scripts/generate_md_dataset.py",
00445 |             REPO_ROOT / "MD" / "scripts",
00446 |         )
00447 |         items = [Path(str(index)) for index in range(10)]
00448 |         split_ranges, excluded = module._split_blocked_with_gap(
00449 |             items,
00450 |             {"train": 4, "validation": 2, "test": 2},
00451 |             temporal_gap=1,
00452 |             block_order=["train", "validation", "test"],
00453 |         )
00454 | 
00455 |         split_by_frame = {
00456 |             int(path.name): split
00457 |             for split, paths in split_ranges.items()
00458 |             for path in paths
00459 |         }
00460 |         for frame, split in split_by_frame.items():
00461 |             for neighbor in (frame - 1, frame + 1):
00462 |                 if neighbor in split_by_frame:
00463 |                     self.assertEqual(split, split_by_frame[neighbor])
00464 |         self.assertEqual([int(path.name) for path, _reason in excluded], [4, 7])
00465 | 
00466 |     def test_md_spread_split_summary_is_marked_exploratory(self) -> None:
00467 |         module = load_repo_script_module(
00468 |             "md_spread_split_fairness_tests",
00469 |             "MD/scripts/generate_md_dataset.py",
00470 |             REPO_ROOT / "MD" / "scripts",
00471 |         )
00472 |         with tempfile.TemporaryDirectory() as tmp:
00473 |             split_root = Path(tmp)
00474 |             module.write_split_summary(
00475 |                 split_root,
00476 |                 {"train": [Path("0")], "validation": [Path("2")], "test": [Path("4")]},
00477 |                 [],
00478 |                 strategy="spread",
00479 |                 temporal_gap=0,
00480 |                 warnings=[module.SPREAD_SPLIT_WARNING],
00481 |             )
00482 |             summary = json.loads((split_root / "split_summary.json").read_text(encoding="utf-8"))
00483 | 
00484 |         self.assertEqual(summary["scientific_status"], "exploratory_temporal_leakage_risk")
00485 |         self.assertTrue(any("interleaves trajectory frames" in warning for warning in summary["warnings"]))
00486 | 
00487 |     def test_random_cartesian_grouped_split_keeps_family_together(self) -> None:
00488 |         module = load_repo_script_module(
00489 |             "random_cartesian_group_split_fairness_tests",
00490 |             "AtomDisplacement/scripts/generate_random_cartesian_dataset.py",
00491 |             REPO_ROOT / "AtomDisplacement" / "scripts",
00492 |         )
00493 |         samples = [
00494 |             {"sample_id": "a1", "split_group_id": "family_a"},
00495 |             {"sample_id": "a2", "split_group_id": "family_a"},
00496 |             {"sample_id": "b1", "split_group_id": "family_b"},
00497 |             {"sample_id": "c1", "split_group_id": "family_c"},
00498 |         ]
00499 |         split_samples, summary = module.grouped_split_assignment(samples)
00500 |         module.assert_group_isolation(split_samples)
00501 | 
00502 |         family_a_splits = [
00503 |             split
00504 |             for split, rows in split_samples.items()
00505 |             if any(row["sample_id"].startswith("a") for row in rows)
00506 |         ]
00507 |         self.assertEqual(family_a_splits, ["train"])
00508 |         self.assertTrue(summary["group_aware"])
00509 |         self.assertEqual(summary["scientific_status"], "grouped_family_splits")
00510 | 
00511 |     def test_random_cartesian_group_isolation_rejects_split_family(self) -> None:
00512 |         module = load_repo_script_module(
00513 |             "random_cartesian_group_isolation_fairness_tests",
00514 |             "AtomDisplacement/scripts/generate_random_cartesian_dataset.py",
00515 |             REPO_ROOT / "AtomDisplacement" / "scripts",
00516 |         )
00517 |         with self.assertRaisesRegex(RuntimeError, "split a family"):
00518 |             module.assert_group_isolation(
00519 |                 {
00520 |                     "train": [{"sample_id": "a1", "split_group_id": "family_a"}],
00521 |                     "validation": [],
00522 |                     "test": [{"sample_id": "a2", "split_group_id": "family_a"}],
00523 |                 }
00524 |             )
00525 | 
00526 |     def test_random_cartesian_provenance_is_not_hidden_under_atom_displacement(self) -> None:
00527 |         module = load_script_module("pipeline_ui_method_provenance_alias_tests", "pipeline_ui.py")
00528 |         with tempfile.TemporaryDirectory() as tmp:
00529 |             root = Path(tmp)
00530 |             manifest = {
00531 |                 "experiment_id": "rc_provenance_case",
00532 |                 "selected_methods": ["random_cartesian"],
00533 |                 "runs": [provenance_run(root, "random_cartesian", "rc_570", 570)],
00534 |                 "siesta_settings_hash_by_method": {"random_cartesian": "siesta_rc"},
00535 |                 "model_config_hash_by_method": {"random_cartesian": "model_rc"},
00536 |                 "basis_hash_by_method": {"random_cartesian": "basis_rc"},
00537 |                 "pseudopotential_hash_by_method": {"random_cartesian": "pseudo_rc"},
00538 |             }
00539 |             module.refresh_method_provenance(manifest)
00540 | 
00541 |         self.assertEqual(set(manifest["method_provenance"]), {"random_cartesian"})
00542 |         rc_entry = manifest["method_provenance"]["random_cartesian"]
00543 |         self.assertEqual(rc_entry["method_id"], "random_cartesian")
00544 |         self.assertEqual(rc_entry["dataset_size"], 570)
00545 |         self.assertEqual(rc_entry["runs"][0]["pipeline"], "random_cartesian")
00546 |         self.assertNotEqual(rc_entry["runs"][0]["pipeline"], "atom_displacement")
00547 | 
00548 | 
00549 | if __name__ == "__main__":
00550 |     unittest.main()
```

## `Comparison/scripts/deeph_prediction_adapter.py` — extractos seleccionados

SHA-256 del archivo completo: `548f860d2cda979ee4b7f2761f20d552ec6186448aea91cf242281a56927ba37`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Validate DeepH HDF5 predictions before common metric evaluation.
00003 | 
00004 | This adapter is intentionally narrow. It does not fabricate SIESTA HSX files,
00005 | and it does not claim that DeepH's processed HDF5 block convention is identical
00006 | to Graph2Mat's raw HSX convention. It only verifies the DeepH HDF5 prediction
00007 | against the DeepH processed SIESTA reference produced from the same snapshot.
00008 | """
00009 | 
00010 | from __future__ import annotations
00011 | 
00012 | import hashlib
00013 | import json
00014 | import math
00015 | from dataclasses import asdict, dataclass, field
00016 | from pathlib import Path
00017 | from typing import Any
00018 | 
00019 | 
00020 | ADAPTER_NAME = "deeph_hdf5_prediction_adapter"
00021 | ADAPTER_VERSION = "deeph_hdf5_prediction_adapter_v1"
00022 | GLOBAL_PREDICTION_FILENAME = "hamiltonians_pred.h5"
00023 | LOCAL_FRAME_PREDICTION_FILENAME = "rh_pred.h5"
00024 | EQUIVALENCE_PROVEN_RAW_GLOBAL = "proven_raw_global_hamiltonian_equivalent"
00025 | EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME = "diagnostic_local_frame_only"
00026 | EQUIVALENCE_INVALID_SHAPE = "invalid_shape_mismatch"
00027 | EQUIVALENCE_INVALID_ORBITAL_ORDER = "invalid_orbital_order_unknown"
00028 | EQUIVALENCE_INVALID_UNITS = "invalid_units_unknown"
00029 | EQUIVALENCE_INVALID_R_VECTOR = "invalid_r_vector_convention_unknown"
00030 | EQUIVALENCE_INVALID_MISSING_REFERENCE = "invalid_missing_reference_mapping"
00031 | EQUIVALENCE_INVALID_EVIDENCE = "invalid_raw_global_equivalence_evidence"
00032 | PROVEN_ADAPTER_EQUIVALENCE_STATUSES = {EQUIVALENCE_PROVEN_RAW_GLOBAL}
00033 | EQUIVALENCE_STATUS_PROVEN = "proven"
00034 | EQUIVALENCE_STATUS_FAILED = "failed"
00035 | EQUIVALENCE_STATUS_UNPROVEN = "unproven"
00036 | EQUIVALENCE_STATUS_NOT_APPLICABLE = "not_applicable"
00037 | EQUIVALENCE_SCOPE_RAW_GLOBAL = "raw_global"
00038 | EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE = "deeph_processed_blockwise_global_hdf5"
00039 | EQUIVALENCE_SCOPE_LOCAL_FRAME = "local_frame_hprime"
00040 | EQUIVALENCE_SCOPE_UNKNOWN = "unknown"
00041 | RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME = "raw_global_equivalence_evidence.json"
00042 | RAW_GLOBAL_EQUIVALENCE_EVIDENCE_SCHEMA = "deeph_raw_global_equivalence_evidence_v1"
00043 | RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS = (
00044 |     "shape",
00045 |     "units",
00046 |     "orbital_order",
00047 |     "atom_order",
00048 |     "r_vectors",
00049 |     "spin",
00050 |     "sparse_support",
00051 |     "hk",
00052 |     "s_ref",
00053 |     "eigenvalues",
00054 | )
00055 | 
```

### `equivalence_status_from_adapter_status` — líneas 61–69

```py
00061 | def equivalence_status_from_adapter_status(adapter_status: str) -> str:
00062 |     status = str(adapter_status or "").strip()
00063 |     if status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES:
00064 |         return EQUIVALENCE_STATUS_PROVEN
00065 |     if status == EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME:
00066 |         return EQUIVALENCE_STATUS_NOT_APPLICABLE
00067 |     if status in {EQUIVALENCE_INVALID_SHAPE, EQUIVALENCE_INVALID_MISSING_REFERENCE, EQUIVALENCE_INVALID_EVIDENCE}:
00068 |         return EQUIVALENCE_STATUS_FAILED
00069 |     return EQUIVALENCE_STATUS_UNPROVEN
```

### `equivalence_scope_from_adapter_status` — líneas 72–81

```py
00072 | def equivalence_scope_from_adapter_status(adapter_status: str, target_space: str = "") -> str:
00073 |     status = str(adapter_status or "").strip()
00074 |     target = str(target_space or "").strip()
00075 |     if status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES:
00076 |         return EQUIVALENCE_SCOPE_RAW_GLOBAL
00077 |     if status == EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME or "local_coordinate" in target:
00078 |         return EQUIVALENCE_SCOPE_LOCAL_FRAME
00079 |     if "global_hamiltonian_h5_blocks" in target:
00080 |         return EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE
00081 |     return EQUIVALENCE_SCOPE_UNKNOWN
```

### `DeepHPredictionAdapterResult` — líneas 84–173

```py
00084 | @dataclass
00085 | class DeepHPredictionAdapterResult:
00086 |     sample_id: str
00087 |     status: str
00088 |     metrics_ready: bool
00089 |     diagnostic_only: bool
00090 |     diagnostic_reason: str
00091 |     prediction_path: str | None
00092 |     processed_sample_dir: str
00093 |     reference_hamiltonian_path: str | None
00094 |     reference_overlap_path: str | None
00095 |     orbital_types_path: str | None
00096 |     n_orbitals: int | None
00097 |     block_count: int
00098 |     prediction_key_count: int
00099 |     reference_key_count: int
00100 |     missing_reference_keys: list[str] = field(default_factory=list)
00101 |     extra_prediction_keys: list[str] = field(default_factory=list)
00102 |     gamma_hermiticity_defect: float | None = None
00103 |     reference_gamma_hermiticity_defect: float | None = None
00104 |     comparability_status: str = "unknown"
00105 |     adapter_equivalence_status: str = EQUIVALENCE_INVALID_MISSING_REFERENCE
00106 |     target_space: str = "unknown"
00107 |     units_status: str = "unknown"
00108 |     orbital_order_status: str = "unknown"
00109 |     r_vector_convention_status: str = "unknown"
00110 |     support_semantics_status: str = "unknown"
00111 |     equivalence_status: str = ""
00112 |     equivalence_scope: str = ""
00113 |     equivalence_evidence_paths: list[str] = field(default_factory=list)
00114 |     equivalence_reason: str = ""
00115 |     adapter_name: str = ADAPTER_NAME
00116 |     adapter_version: str = ADAPTER_VERSION
00117 |     provenance: dict[str, Any] = field(default_factory=dict)
00118 |     warnings: list[str] = field(default_factory=list)
00119 | 
00120 |     def __post_init__(self) -> None:
00121 |         if not self.equivalence_status:
00122 |             self.equivalence_status = equivalence_status_from_adapter_status(self.adapter_equivalence_status)
00123 |         if not self.equivalence_scope:
00124 |             self.equivalence_scope = equivalence_scope_from_adapter_status(
00125 |                 self.adapter_equivalence_status,
00126 |                 self.target_space,
00127 |             )
00128 |         if not self.equivalence_reason:
00129 |             if self.equivalence_status == EQUIVALENCE_STATUS_PROVEN:
00130 |                 self.equivalence_reason = "raw/global Hamiltonian equivalence evidence is recorded."
00131 |             elif self.diagnostic_reason:
00132 |                 self.equivalence_reason = self.diagnostic_reason
00133 |             else:
00134 |                 self.equivalence_reason = "DeepH raw/global equivalence evidence is unavailable."
00135 |         if not self.equivalence_evidence_paths:
00136 |             self.equivalence_evidence_paths = [
00137 |                 str(path)
00138 |                 for path in (
00139 |                     self.prediction_path,
00140 |                     self.reference_hamiltonian_path,
00141 |                     self.reference_overlap_path,
00142 |                     self.orbital_types_path,
00143 |                 )
00144 |                 if path
00145 |             ]
00146 |         if self.equivalence_status != EQUIVALENCE_STATUS_PROVEN:
00147 |             self.diagnostic_only = True
00148 | 
00149 |     def to_dict(self) -> dict[str, Any]:
00150 |         return asdict(self)
00151 | 
00152 |     def metric_fields(self) -> dict[str, Any]:
00153 |         return {
00154 |             "prediction_adapter": self.adapter_name,
00155 |             "prediction_adapter_version": self.adapter_version,
00156 |             "deeph_comparability_status": self.comparability_status,
00157 |             "deeph_adapter_equivalence_status": self.adapter_equivalence_status,
00158 |             "deeph_equivalence_status": self.equivalence_status,
00159 |             "deeph_equivalence_scope": self.equivalence_scope,
00160 |             "deeph_equivalence_evidence_paths": list(self.equivalence_evidence_paths),
00161 |             "deeph_equivalence_reason": self.equivalence_reason,
00162 |             "deeph_raw_global_equivalence_proven": self.adapter_equivalence_status
00163 |             in PROVEN_ADAPTER_EQUIVALENCE_STATUSES
00164 |             and self.equivalence_status == EQUIVALENCE_STATUS_PROVEN,
00165 |             "deeph_diagnostic_only": self.diagnostic_only,
00166 |             "deeph_diagnostic_reason": self.diagnostic_reason,
00167 |             "deeph_prediction_target_space": self.target_space,
00168 |             "deeph_units_status": self.units_status,
00169 |             "deeph_orbital_order_status": self.orbital_order_status,
00170 |             "deeph_r_vector_convention_status": self.r_vector_convention_status,
00171 |             "deeph_support_semantics_status": self.support_semantics_status,
00172 |             "deeph_prediction_metrics_ready": self.metrics_ready,
00173 |         }
```

### `find_raw_global_equivalence_evidence` — líneas 207–222

```py
00207 | def find_raw_global_equivalence_evidence(
00208 |     *,
00209 |     work_dir: Path,
00210 |     processed_sample_dir: Path,
00211 |     sample_id: str,
00212 | ) -> Path | None:
00213 |     candidates = [
00214 |         work_dir / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME,
00215 |         processed_sample_dir / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME,
00216 |         work_dir / f"{sample_id}_{RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME}",
00217 |         processed_sample_dir / f"{sample_id}_{RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME}",
00218 |     ]
00219 |     for path in candidates:
00220 |         if path.exists():
00221 |             return path
00222 |     return None
```

### `validate_raw_global_equivalence_evidence` — líneas 232–302

```py
00232 | def validate_raw_global_equivalence_evidence(
00233 |     path: Path,
00234 |     *,
00235 |     sample_id: str,
00236 | ) -> dict[str, Any]:
00237 |     payload = read_json(path)
00238 |     status = str(payload.get("equivalence_status") or payload.get("status") or "").strip().lower()
00239 |     scope = str(payload.get("equivalence_scope") or payload.get("scope") or "").strip().lower()
00240 |     if payload.get("sample_id") not in (None, "", sample_id):
00241 |         return {
00242 |             "status": EQUIVALENCE_STATUS_FAILED,
00243 |             "reason": (
00244 |                 f"{EQUIVALENCE_INVALID_EVIDENCE}: sample_id mismatch in {path}: "
00245 |                 f"{payload.get('sample_id')!r} != {sample_id!r}"
00246 |             ),
00247 |             "payload": payload,
00248 |         }
00249 |     if status != EQUIVALENCE_STATUS_PROVEN or scope != EQUIVALENCE_SCOPE_RAW_GLOBAL:
00250 |         return {
00251 |             "status": EQUIVALENCE_STATUS_FAILED,
00252 |             "reason": (
00253 |                 f"{EQUIVALENCE_INVALID_EVIDENCE}: evidence must declare "
00254 |                 f"status=proven and scope=raw_global."
00255 |             ),
00256 |             "payload": payload,
00257 |         }
00258 |     checks = payload.get("checks")
00259 |     if not isinstance(checks, dict):
00260 |         return {
00261 |             "status": EQUIVALENCE_STATUS_FAILED,
00262 |             "reason": f"{EQUIVALENCE_INVALID_EVIDENCE}: missing checks object.",
00263 |             "payload": payload,
00264 |         }
00265 |     missing_or_failed = [key for key in RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS if not _bool_check(checks, key)]
00266 |     if missing_or_failed:
00267 |         return {
00268 |             "status": EQUIVALENCE_STATUS_FAILED,
00269 |             "reason": (
00270 |                 f"{EQUIVALENCE_INVALID_EVIDENCE}: required checks failed or missing: "
00271 |                 + ", ".join(missing_or_failed)
00272 |             ),
00273 |             "payload": payload,
00274 |         }
00275 |     errors = payload.get("errors") or {}
00276 |     tolerances = payload.get("tolerances") or {}
00277 |     if isinstance(errors, dict) and isinstance(tolerances, dict):
00278 |         for key, raw_error in errors.items():
00279 |             if key not in tolerances:
00280 |                 continue
00281 |             try:
00282 |                 error = abs(float(raw_error))
00283 |                 tolerance = abs(float(tolerances[key]))
00284 |             except (TypeError, ValueError):
00285 |                 return {
00286 |                     "status": EQUIVALENCE_STATUS_FAILED,
00287 |                     "reason": f"{EQUIVALENCE_INVALID_EVIDENCE}: non-numeric error/tolerance for {key}.",
00288 |                     "payload": payload,
00289 |                 }
00290 |             if not math.isfinite(error) or not math.isfinite(tolerance) or error > tolerance:
00291 |                 return {
00292 |                     "status": EQUIVALENCE_STATUS_FAILED,
00293 |                     "reason": (
00294 |                         f"{EQUIVALENCE_INVALID_EVIDENCE}: {key}={error} exceeds tolerance {tolerance}."
00295 |                     ),
00296 |                     "payload": payload,
00297 |                 }
00298 |     return {
00299 |         "status": EQUIVALENCE_STATUS_PROVEN,
00300 |         "reason": "raw/global Hamiltonian equivalence evidence passed all required checks.",
00301 |         "payload": payload,
00302 |     }
```

### `parse_block_key` — líneas 318–325

```py
00318 | def parse_block_key(key: str) -> tuple[tuple[int, int, int], int, int]:
00319 |     try:
00320 |         values = json.loads(key)
00321 |     except json.JSONDecodeError as exc:
00322 |         raise DeepHPredictionAdapterError(f"Invalid DeepH block key JSON: {key}") from exc
00323 |     if not isinstance(values, list) or len(values) != 5:
00324 |         raise DeepHPredictionAdapterError(f"Invalid DeepH block key: {key}")
00325 |     return (int(values[0]), int(values[1]), int(values[2])), int(values[3]) - 1, int(values[4]) - 1
```

### `expected_block_shape` — líneas 328–332

```py
00328 | def expected_block_shape(key: str, orbital_counts: list[int]) -> tuple[int, int]:
00329 |     _, atom_i, atom_j = parse_block_key(key)
00330 |     if atom_i < 0 or atom_j < 0 or atom_i >= len(orbital_counts) or atom_j >= len(orbital_counts):
00331 |         raise DeepHPredictionAdapterError(f"DeepH block key atom index out of range: {key}")
00332 |     return int(orbital_counts[atom_i]), int(orbital_counts[atom_j])
```

### `h5_block_shapes` — líneas 335–352

```py
00335 | def h5_block_shapes(path: Path, orbital_counts: list[int], *, label: str) -> dict[str, tuple[int, int]]:
00336 |     h5py, _ = _require_h5py_numpy()
00337 |     if not path.exists():
00338 |         raise DeepHPredictionAdapterError(f"Missing {label} HDF5 file: {path}")
00339 |     shapes: dict[str, tuple[int, int]] = {}
00340 |     with h5py.File(path, "r") as handle:
00341 |         for key in sorted(handle.keys()):
00342 |             expected = expected_block_shape(key, orbital_counts)
00343 |             shape = tuple(int(value) for value in handle[key].shape)
00344 |             if shape != expected:
00345 |                 raise DeepHPredictionAdapterError(
00346 |                     f"{EQUIVALENCE_INVALID_SHAPE}: block shape mismatch in {path}: "
00347 |                     f"{key} has {shape}, expected {expected}"
00348 |                 )
00349 |             shapes[key] = expected
00350 |     if not shapes:
00351 |         raise DeepHPredictionAdapterError(f"{label} HDF5 file contains no blocks: {path}")
00352 |     return shapes
```

### `assemble_hk` — líneas 355–377

```py
00355 | def assemble_hk(block_h5: Path, orbital_types: Path, kpoint: tuple[float, float, float]) -> Any:
00356 |     h5py, np = _require_h5py_numpy()
00357 |     orbital_counts = count_orbitals_from_orbital_types(orbital_types)
00358 |     offsets = np.cumsum([0, *orbital_counts])
00359 |     matrix = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=np.complex128)
00360 |     with h5py.File(block_h5, "r") as handle:
00361 |         for key in handle.keys():
00362 |             lattice_r, atom_i, atom_j = parse_block_key(key)
00363 |             r0, r1 = int(offsets[atom_i]), int(offsets[atom_i + 1])
00364 |             c0, c1 = int(offsets[atom_j]), int(offsets[atom_j + 1])
00365 |             block = np.asarray(handle[key][()])
00366 |             if block.shape != (r1 - r0, c1 - c0):
00367 |                 raise DeepHPredictionAdapterError(
00368 |                     f"{EQUIVALENCE_INVALID_SHAPE}: block shape mismatch in {block_h5}: "
00369 |                     f"{key} has {block.shape}, expected {(r1-r0, c1-c0)}"
00370 |                 )
00371 |             phase = np.exp(
00372 |                 2j
00373 |                 * np.pi
00374 |                 * float(np.dot(np.asarray(kpoint, dtype=float), np.asarray(lattice_r, dtype=float)))
00375 |             )
00376 |             matrix[r0:r1, c0:c1] += block * phase
00377 |     return matrix
```

### `hermiticity_defect` — líneas 380–387

```py
00380 | def hermiticity_defect(matrix: Any) -> float:
00381 |     _, np = _require_h5py_numpy()
00382 |     if matrix.size == 0:
00383 |         return math.nan
00384 |     denominator = float(np.linalg.norm(matrix))
00385 |     if denominator == 0.0:
00386 |         return 0.0
00387 |     return float(np.linalg.norm(matrix - matrix.conj().T) / denominator)
```

### `adapt_deeph_prediction_sample` — líneas 432–618

```py
00432 | def adapt_deeph_prediction_sample(
00433 |     *,
00434 |     work_dir: Path,
00435 |     processed_sample_dir: Path,
00436 |     sample_id: str | None = None,
00437 |     prediction_filename: str = GLOBAL_PREDICTION_FILENAME,
00438 | ) -> DeepHPredictionAdapterResult:
00439 |     work_dir = Path(work_dir)
00440 |     processed_sample_dir = Path(processed_sample_dir)
00441 |     sample_id = str(sample_id or work_dir.name)
00442 |     prediction_path, prediction_kind = find_prediction_file(work_dir, prediction_filename)
00443 |     orbital_types = processed_sample_dir / "orbital_types.dat"
00444 |     reference_hamiltonian = processed_sample_dir / "hamiltonians.h5"
00445 |     reference_overlap = processed_sample_dir / "overlaps.h5"
00446 | 
00447 |     if prediction_kind == "local_frame":
00448 |         return DeepHPredictionAdapterResult(
00449 |             sample_id=sample_id,
00450 |             status="local_frame_prediction_only",
00451 |             metrics_ready=False,
00452 |             diagnostic_only=True,
00453 |             diagnostic_reason=(
00454 |                 "DeepH produced rh_pred.h5 only. This is the local-coordinate H' "
00455 |                 "representation and is not a raw/global Hamiltonian for common metrics."
00456 |             ),
00457 |             prediction_path=str(prediction_path),
00458 |             processed_sample_dir=str(processed_sample_dir),
00459 |             reference_hamiltonian_path=str(reference_hamiltonian) if reference_hamiltonian.exists() else None,
00460 |             reference_overlap_path=str(reference_overlap) if reference_overlap.exists() else None,
00461 |             orbital_types_path=str(orbital_types) if orbital_types.exists() else None,
00462 |             n_orbitals=None,
00463 |             block_count=0,
00464 |             prediction_key_count=0,
00465 |             reference_key_count=0,
00466 |             comparability_status="diagnostic_only_local_frame_hprime",
00467 |             adapter_equivalence_status=EQUIVALENCE_DIAGNOSTIC_LOCAL_FRAME,
00468 |             target_space="deeph_local_coordinate_hprime",
00469 |             units_status="not_applicable_to_common_raw_global_metrics",
00470 |             orbital_order_status="not_validated",
00471 |             r_vector_convention_status="not_validated",
00472 |             support_semantics_status="not_validated",
00473 |             equivalence_status=EQUIVALENCE_STATUS_NOT_APPLICABLE,
00474 |             equivalence_scope=EQUIVALENCE_SCOPE_LOCAL_FRAME,
00475 |             equivalence_reason="DeepH local-coordinate H' output is not a raw/global Hamiltonian.",
00476 |             provenance=_provenance(
00477 |                 prediction_path=prediction_path,
00478 |                 reference_hamiltonian=reference_hamiltonian if reference_hamiltonian.exists() else None,
00479 |                 reference_overlap=reference_overlap if reference_overlap.exists() else None,
00480 |                 orbital_types=orbital_types if orbital_types.exists() else None,
00481 |             ),
00482 |         )
00483 | 
00484 |     orbital_counts = count_orbitals_from_orbital_types(orbital_types)
00485 |     n_orbitals = int(sum(orbital_counts))
00486 |     pred_shapes = h5_block_shapes(prediction_path, orbital_counts, label="DeepH prediction")
00487 |     ref_shapes = h5_block_shapes(reference_hamiltonian, orbital_counts, label="DeepH processed reference")
00488 |     h5_block_shapes(reference_overlap, orbital_counts, label="DeepH processed overlap")
00489 |     pred_keys = set(pred_shapes)
00490 |     ref_keys = set(ref_shapes)
00491 |     missing_reference_keys = sorted(ref_keys - pred_keys)
00492 |     extra_prediction_keys = sorted(pred_keys - ref_keys)
00493 |     if missing_reference_keys or extra_prediction_keys:
00494 |         raise DeepHPredictionAdapterError(
00495 |             f"{EQUIVALENCE_INVALID_MISSING_REFERENCE}: DeepH prediction/reference block support mismatch: "
00496 |             f"missing_prediction_keys={missing_reference_keys[:10]} "
00497 |             f"extra_prediction_keys={extra_prediction_keys[:10]}"
00498 |         )
00499 |     for key in sorted(pred_keys):
00500 |         if pred_shapes[key] != ref_shapes[key]:
00501 |             raise DeepHPredictionAdapterError(
00502 |                 f"{EQUIVALENCE_INVALID_SHAPE}: DeepH prediction/reference block shape mismatch for {key}: "
00503 |                 f"{pred_shapes[key]} vs {ref_shapes[key]}"
00504 |             )
00505 | 
00506 |     pred_gamma = assemble_hk(prediction_path, orbital_types, (0.0, 0.0, 0.0))
00507 |     ref_gamma = assemble_hk(reference_hamiltonian, orbital_types, (0.0, 0.0, 0.0))
00508 |     evidence_path = find_raw_global_equivalence_evidence(
00509 |         work_dir=work_dir,
00510 |         processed_sample_dir=processed_sample_dir,
00511 |         sample_id=sample_id,
00512 |     )
00513 |     evidence: dict[str, Any] | None = None
00514 |     evidence_reason = ""
00515 |     evidence_status = ""
00516 |     if evidence_path is not None:
00517 |         evidence = validate_raw_global_equivalence_evidence(evidence_path, sample_id=sample_id)
00518 |         evidence_status = str(evidence.get("status") or "")
00519 |         evidence_reason = str(evidence.get("reason") or "")
00520 |         if evidence_status == EQUIVALENCE_STATUS_PROVEN:
00521 |             provenance = _provenance(
00522 |                 prediction_path=prediction_path,
00523 |                 reference_hamiltonian=reference_hamiltonian,
00524 |                 reference_overlap=reference_overlap,
00525 |                 orbital_types=orbital_types,
00526 |             )
00527 |             provenance["raw_global_equivalence_evidence"] = {
00528 |                 "path": str(evidence_path),
00529 |                 "sha256": file_sha256(evidence_path),
00530 |                 "schema": (evidence.get("payload") or {}).get("schema"),
00531 |             }
00532 |             return DeepHPredictionAdapterResult(
00533 |                 sample_id=sample_id,
00534 |                 status="ok",
00535 |                 metrics_ready=True,
00536 |                 diagnostic_only=False,
00537 |                 diagnostic_reason="",
00538 |                 prediction_path=str(prediction_path),
00539 |                 processed_sample_dir=str(processed_sample_dir),
00540 |                 reference_hamiltonian_path=str(reference_hamiltonian),
00541 |                 reference_overlap_path=str(reference_overlap),
00542 |                 orbital_types_path=str(orbital_types),
00543 |                 n_orbitals=n_orbitals,
00544 |                 block_count=len(pred_shapes),
00545 |                 prediction_key_count=len(pred_shapes),
00546 |                 reference_key_count=len(ref_shapes),
00547 |                 missing_reference_keys=missing_reference_keys,
00548 |                 extra_prediction_keys=extra_prediction_keys,
00549 |                 gamma_hermiticity_defect=hermiticity_defect(pred_gamma),
00550 |                 reference_gamma_hermiticity_defect=hermiticity_defect(ref_gamma),
00551 |                 comparability_status="raw_global_equivalence_proven",
00552 |                 adapter_equivalence_status=EQUIVALENCE_PROVEN_RAW_GLOBAL,
00553 |                 target_space="deeph_rotate_back_global_hamiltonian_h5_blocks_verified_raw_global",
00554 |                 units_status="verified_by_raw_global_equivalence_evidence",
00555 |                 orbital_order_status="verified_by_raw_global_equivalence_evidence",
00556 |                 r_vector_convention_status="verified_by_raw_global_equivalence_evidence",
00557 |                 support_semantics_status="verified_by_raw_global_equivalence_evidence",
00558 |                 equivalence_status=EQUIVALENCE_STATUS_PROVEN,
00559 |                 equivalence_scope=EQUIVALENCE_SCOPE_RAW_GLOBAL,
00560 |                 equivalence_evidence_paths=[str(evidence_path)],
00561 |                 equivalence_reason=evidence_reason,
00562 |                 provenance=provenance,
00563 |             )
00564 |     warnings = [
00565 |         "DeepH HDF5 blocks were validated against DeepH processed SIESTA HDF5 artifacts only.",
00566 |         "Equivalence to Graph2Mat raw HSX orbital order/sign convention is not independently proven.",
00567 |         f"{EQUIVALENCE_INVALID_UNITS}: DeepH processed energy units were not independently checked against Graph2Mat HSX.",
00568 |         f"{EQUIVALENCE_INVALID_R_VECTOR}: DeepH HDF5 R-vector convention was not independently checked against Graph2Mat HSX.",
00569 |     ]
00570 |     provenance = _provenance(
00571 |         prediction_path=prediction_path,
00572 |         reference_hamiltonian=reference_hamiltonian,
00573 |         reference_overlap=reference_overlap,
00574 |         orbital_types=orbital_types,
00575 |     )
00576 |     if evidence_path is not None:
00577 |         provenance["raw_global_equivalence_evidence"] = {
00578 |             "path": str(evidence_path),
00579 |             "sha256": file_sha256(evidence_path),
00580 |             "schema": (evidence or {}).get("payload", {}).get("schema") if isinstance(evidence, dict) else None,
00581 |         }
00582 |     return DeepHPredictionAdapterResult(
00583 |         sample_id=sample_id,
00584 |         status="ok",
00585 |         metrics_ready=True,
00586 |         diagnostic_only=True,
00587 |         diagnostic_reason="basis_equivalence_to_graph2mat_raw_hsx_not_proven",
00588 |         prediction_path=str(prediction_path),
00589 |         processed_sample_dir=str(processed_sample_dir),
00590 |         reference_hamiltonian_path=str(reference_hamiltonian),
00591 |         reference_overlap_path=str(reference_overlap),
00592 |         orbital_types_path=str(orbital_types),
00593 |         n_orbitals=n_orbitals,
00594 |         block_count=len(pred_shapes),
00595 |         prediction_key_count=len(pred_shapes),
00596 |         reference_key_count=len(ref_shapes),
00597 |         missing_reference_keys=missing_reference_keys,
00598 |         extra_prediction_keys=extra_prediction_keys,
00599 |         gamma_hermiticity_defect=hermiticity_defect(pred_gamma),
00600 |         reference_gamma_hermiticity_defect=hermiticity_defect(ref_gamma),
00601 |         comparability_status="diagnostic_deeph_processed_global_hdf5_blocks_shape_validated",
00602 |         adapter_equivalence_status=EQUIVALENCE_INVALID_EVIDENCE if evidence_path is not None else EQUIVALENCE_INVALID_ORBITAL_ORDER,
00603 |         target_space="deeph_rotate_back_global_hamiltonian_h5_blocks",
00604 |         units_status="deeph_siesta_preprocess_internal_energy_units_unverified_against_graph2mat_hsx",
00605 |         orbital_order_status="validated_against_deeph_processed_reference_only",
00606 |         r_vector_convention_status="validated_against_deeph_processed_reference_only",
00607 |         support_semantics_status="prediction_and_processed_reference_key_sets_match",
00608 |         equivalence_status=EQUIVALENCE_STATUS_FAILED if evidence_path is not None else EQUIVALENCE_STATUS_UNPROVEN,
00609 |         equivalence_scope=EQUIVALENCE_SCOPE_DEEPH_PROCESSED_BLOCKWISE,
00610 |         equivalence_evidence_paths=[str(evidence_path)] if evidence_path is not None else [],
00611 |         equivalence_reason=evidence_reason
00612 |         or (
00613 |             "DeepH prediction was validated only against DeepH processed SIESTA HDF5 blocks; "
00614 |             "raw/global HSX units, orbital order, and R-vector convention are not proven."
00615 |         ),
00616 |         provenance=provenance,
00617 |         warnings=warnings,
00618 |     )
```

### `write_adapter_manifest` — líneas 621–668

```py
00621 | def write_adapter_manifest(path: Path, results: list[DeepHPredictionAdapterResult]) -> dict[str, Any]:
00622 |     adapter_equivalence_statuses = sorted({result.adapter_equivalence_status for result in results})
00623 |     equivalence_statuses = sorted({result.equivalence_status for result in results})
00624 |     equivalence_scopes = sorted({result.equivalence_scope for result in results})
00625 |     equivalence_evidence_paths = sorted(
00626 |         {path for result in results for path in result.equivalence_evidence_paths if path}
00627 |     )
00628 |     proven_count = sum(
00629 |         1
00630 |         for result in results
00631 |         if result.adapter_equivalence_status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES
00632 |         and result.equivalence_status == EQUIVALENCE_STATUS_PROVEN
00633 |     )
00634 |     robust_allowed = bool(results) and proven_count == len(results)
00635 |     blocked_reasons = sorted(
00636 |         {
00637 |             result.equivalence_reason
00638 |             for result in results
00639 |             if result.equivalence_status != EQUIVALENCE_STATUS_PROVEN and result.equivalence_reason
00640 |         }
00641 |     )
00642 |     payload = {
00643 |         "schema": ADAPTER_VERSION,
00644 |         "adapter_name": ADAPTER_NAME,
00645 |         "adapter_version": ADAPTER_VERSION,
00646 |         "sample_count": len(results),
00647 |         "metrics_ready_count": sum(1 for result in results if result.metrics_ready),
00648 |         "diagnostic_only_count": sum(1 for result in results if result.diagnostic_only),
00649 |         "adapter_equivalence_statuses": adapter_equivalence_statuses,
00650 |         "equivalence_statuses": equivalence_statuses,
00651 |         "equivalence_scopes": equivalence_scopes,
00652 |         "equivalence_evidence_paths": equivalence_evidence_paths,
00653 |         "raw_global_equivalence_proven_count": proven_count,
00654 |         "robust_matrix_metrics_allowed": robust_allowed,
00655 |         "equivalence_gate": {
00656 |             "robust_claim_allowed": robust_allowed,
00657 |             "diagnostic_only": not robust_allowed,
00658 |             "required_status": EQUIVALENCE_STATUS_PROVEN,
00659 |             "required_scope": EQUIVALENCE_SCOPE_RAW_GLOBAL,
00660 |             "diagnostic_only_reason": "; ".join(blocked_reasons)
00661 |             if blocked_reasons
00662 |             else ("" if robust_allowed else "DeepH equivalence evidence is missing."),
00663 |         },
00664 |         "samples": [result.to_dict() for result in results],
00665 |     }
00666 |     path.parent.mkdir(parents=True, exist_ok=True)
00667 |     path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
00668 |     return payload
```

## `Comparison/scripts/deeph_raw_global_equivalence_preflight.py` — extractos seleccionados

SHA-256 del archivo completo: `51d995d9a1c343d4a6f16e9873f97d95cd6784bbad9e55a0995d7f772b4d83fe`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Generate numeric DeepH raw/global equivalence evidence.
00003 | 
00004 | The preflight compares DeepH processed global HDF5 reference blocks against the
00005 | raw SIESTA/Graph2Mat reference matrix for selected frozen samples. It writes
00006 | ``raw_global_equivalence_evidence.json`` files in the DeepH prediction work
00007 | directories so the existing DeepH adapter can consume them fail-closed.
00008 | """
00009 | 
00010 | from __future__ import annotations
00011 | 
00012 | import argparse
00013 | import json
00014 | import math
00015 | import re
00016 | import sys
00017 | import time
00018 | from pathlib import Path
00019 | from typing import Any
00020 | 
00021 | from deeph_prediction_adapter import (
00022 |     RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME,
00023 |     RAW_GLOBAL_EQUIVALENCE_EVIDENCE_SCHEMA,
00024 |     RAW_GLOBAL_EQUIVALENCE_REQUIRED_CHECKS,
00025 |     assemble_hk,
00026 |     count_orbitals_from_orbital_types,
00027 |     file_sha256,
00028 |     h5_block_shapes,
00029 |     validate_raw_global_equivalence_evidence,
00030 | )
00031 | 
00032 | 
00033 | PREFLIGHT_SCHEMA = "deeph_raw_global_equivalence_preflight_v1"
00034 | DEFAULT_MATRIX_TOLERANCE = 1e-6
00035 | DEFAULT_EIGENVALUE_TOLERANCE = 3e-6
00036 | SUPPORT_THRESHOLD = 1e-12
00037 | FORBIDDEN_REFERENCE_NAME = "ML_prediction.HSX"
00038 | SUPPORTED_ORBITAL_LABELS = {"s", "px", "py", "pz"}
00039 | SIESTA_TO_DEEPH_ORBITAL_SIGNS = {
00040 |     "s": 1.0,
00041 |     "px": -1.0,
00042 |     "py": -1.0,
00043 |     "pz": 1.0,
00044 | }
00045 | 
```

### `normalize_orbital_label` — líneas 100–110

```py
00100 | def normalize_orbital_label(*, sym: str, angular_l: int, angular_m: int) -> str:
00101 |     label = str(sym or "").strip().lower()
00102 |     if label in SUPPORTED_ORBITAL_LABELS:
00103 |         return label
00104 |     if angular_l == 0:
00105 |         return "s"
00106 |     if angular_l == 1:
00107 |         # SIESTA ORB_INDX commonly stores the real p orbitals as m=-1,0,+1
00108 |         # with labels py,pz,px. Prefer the explicit label above when present.
00109 |         return {-1: "py", 0: "pz", 1: "px"}.get(int(angular_m), f"p{angular_m}")
00110 |     return label or f"l{angular_l}_m{angular_m}"
```

### `siesta_orbital_labels_from_orb_indx` — líneas 113–147

```py
00113 | def siesta_orbital_labels_from_orb_indx(path: Path, *, expected_count: int) -> list[str]:
00114 |     if not path.exists():
00115 |         raise DeepHEquivalencePreflightError(f"missing SIESTA ORB_INDX for orbital mapping: {path}")
00116 |     orbitals: list[tuple[int, str]] = []
00117 |     for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
00118 |         tokens = [token for token in line.split() if token.strip()]
00119 |         if len(tokens) < 16:
00120 |             continue
00121 |         try:
00122 |             int(tokens[0])
00123 |             angular_l = int(tokens[6])
00124 |             angular_m = int(tokens[7])
00125 |             isc = tuple(int(value) for value in tokens[-4:-1])
00126 |             unit_cell_index = int(tokens[-1])
00127 |         except ValueError:
00128 |             continue
00129 |         if isc != (0, 0, 0):
00130 |             continue
00131 |         if unit_cell_index < 1 or unit_cell_index > expected_count:
00132 |             continue
00133 |         label = normalize_orbital_label(sym=tokens[10], angular_l=angular_l, angular_m=angular_m)
00134 |         orbitals.append((unit_cell_index - 1, label))
00135 |     orbitals = sorted(dict(orbitals).items())
00136 |     labels = [label for _index, label in orbitals]
00137 |     if len(labels) != expected_count:
00138 |         raise DeepHEquivalencePreflightError(
00139 |             f"SIESTA ORB_INDX unit-cell orbital count mismatch in {path}: "
00140 |             f"{len(labels)} != {expected_count}"
00141 |         )
00142 |     unsupported = sorted({label for label in labels if label not in SUPPORTED_ORBITAL_LABELS})
00143 |     if unsupported:
00144 |         raise DeepHEquivalencePreflightError(
00145 |             "unsupported SIESTA orbital label(s) for DeepH raw/global mapping: " + ", ".join(unsupported)
00146 |         )
00147 |     return labels
```

### `deeph_orbital_labels_from_orbital_types` — líneas 150–170

```py
00150 | def deeph_orbital_labels_from_orbital_types(path: Path) -> list[str]:
00151 |     if not path.exists():
00152 |         raise DeepHEquivalencePreflightError(f"missing DeepH orbital_types.dat for orbital mapping: {path}")
00153 |     labels: list[str] = []
00154 |     for line in path.read_text(encoding="utf-8").splitlines():
00155 |         tokens = [token for token in line.split() if token.strip()]
00156 |         for token in tokens:
00157 |             angular_l = int(token)
00158 |             if angular_l == 0:
00159 |                 labels.append("s")
00160 |             elif angular_l == 1:
00161 |                 # DeepH rotate_back writes p blocks in OpenMX xyz order.
00162 |                 labels.extend(["px", "py", "pz"])
00163 |             else:
00164 |                 raise DeepHEquivalencePreflightError(
00165 |                     f"unsupported DeepH angular momentum l={angular_l} in {path}; "
00166 |                     "raw/global mapping is currently verified only for s/p orbitals"
00167 |                 )
00168 |     if not labels:
00169 |         raise DeepHEquivalencePreflightError(f"no DeepH orbital labels found in {path}")
00170 |     return labels
```

### `derive_deeph_to_siesta_basis_transform` — líneas 173–221

```py
00173 | def derive_deeph_to_siesta_basis_transform(
00174 |     *,
00175 |     row: dict[str, Any],
00176 |     manifest_dir: Path,
00177 |     orbital_types: Path,
00178 |     n_orbitals: int,
00179 | ) -> dict[str, Any]:
00180 |     orb_indx = artifact_path(row, "orb_indx", manifest_dir=manifest_dir)
00181 |     if orb_indx is None or not orb_indx.exists():
00182 |         return {
00183 |             "status": "identity_missing_orb_indx",
00184 |             "orb_indx_path": str(orb_indx or ""),
00185 |             "permutation": list(range(n_orbitals)),
00186 |             "signs": [1.0] * n_orbitals,
00187 |             "siesta_orbital_labels": [],
00188 |             "deeph_orbital_labels": [],
00189 |             "warnings": ["missing ORB_INDX; DeepH/SIESTA orbital mapping was left as identity"],
00190 |         }
00191 |     siesta_labels = siesta_orbital_labels_from_orb_indx(orb_indx, expected_count=n_orbitals)
00192 |     deeph_labels = deeph_orbital_labels_from_orbital_types(orbital_types)
00193 |     if len(deeph_labels) != n_orbitals:
00194 |         raise DeepHEquivalencePreflightError(
00195 |             f"DeepH orbital_types count mismatch in {orbital_types}: {len(deeph_labels)} != {n_orbitals}"
00196 |         )
00197 | 
00198 |     used: set[int] = set()
00199 |     permutation: list[int] = []
00200 |     for label in siesta_labels:
00201 |         try:
00202 |             source_index = next(
00203 |                 index for index, source_label in enumerate(deeph_labels) if index not in used and source_label == label
00204 |             )
00205 |         except StopIteration as exc:
00206 |             raise DeepHEquivalencePreflightError(
00207 |                 "cannot map DeepH orbital order to SIESTA ORB_INDX order; missing label " + label
00208 |             ) from exc
00209 |         used.add(source_index)
00210 |         permutation.append(source_index)
00211 |     signs = [float(SIESTA_TO_DEEPH_ORBITAL_SIGNS[label]) for label in siesta_labels]
00212 |     return {
00213 |         "status": "applied",
00214 |         "orb_indx_path": str(orb_indx),
00215 |         "permutation": permutation,
00216 |         "signs": signs,
00217 |         "siesta_orbital_labels": siesta_labels,
00218 |         "deeph_orbital_labels": deeph_labels,
00219 |         "sign_policy": "SIESTA real-orbital signs matched to DeepH/OpenMX rotate_back convention",
00220 |         "warnings": [],
00221 |     }
```

### `apply_basis_transform` — líneas 224–236

```py
00224 | def apply_basis_transform(matrix: Any, transform: dict[str, Any]) -> Any:
00225 |     try:
00226 |         import numpy as np  # type: ignore[import-not-found]
00227 |     except ImportError as exc:  # pragma: no cover - guarded by runtime helper in numeric path.
00228 |         raise DeepHEquivalencePreflightError("numpy is required for basis transforms") from exc
00229 |     permutation = [int(index) for index in transform.get("permutation") or []]
00230 |     signs = np.asarray([float(value) for value in transform.get("signs") or []], dtype=float)
00231 |     if not permutation:
00232 |         return matrix
00233 |     transformed = np.asarray(matrix)[np.ix_(permutation, permutation)]
00234 |     if signs.size != transformed.shape[0]:
00235 |         raise DeepHEquivalencePreflightError("basis transform sign count does not match matrix shape")
00236 |     return (signs[:, None] * transformed) * signs[None, :]
```

### `fit_energy_reference_shift` — líneas 239–254

```py
00239 | def fit_energy_reference_shift(*, deeph_h: Any, raw_h: Any, raw_s: Any) -> float:
00240 |     try:
00241 |         import numpy as np  # type: ignore[import-not-found]
00242 |     except ImportError as exc:  # pragma: no cover - guarded by runtime helper in numeric path.
00243 |         raise DeepHEquivalencePreflightError("numpy is required for energy reference fitting") from exc
00244 |     residual = (np.asarray(raw_h) - np.asarray(deeph_h)).reshape(-1)
00245 |     overlap = np.asarray(raw_s).reshape(-1)
00246 |     denominator = np.vdot(overlap, overlap)
00247 |     if abs(denominator) == 0.0:
00248 |         raise DeepHEquivalencePreflightError("cannot fit DeepH energy reference shift: zero overlap norm")
00249 |     shift = np.vdot(overlap, residual) / denominator
00250 |     if abs(float(np.imag(shift))) > 1e-10:
00251 |         raise DeepHEquivalencePreflightError(
00252 |             f"DeepH energy reference shift has non-negligible imaginary component: {shift}"
00253 |         )
00254 |     return float(np.real(shift))
```

### `select_frozen_rows` — líneas 262–293

```py
00262 | def select_frozen_rows(
00263 |     rows: list[dict[str, Any]],
00264 |     *,
00265 |     sample_ids: list[str],
00266 |     sample_limit: int,
00267 | ) -> list[dict[str, Any]]:
00268 |     if sample_limit <= 0:
00269 |         raise DeepHEquivalencePreflightError("--sample-limit must be positive")
00270 |     by_id = {sample_id_from_row(row): row for row in rows}
00271 |     if sample_ids:
00272 |         missing = [sample for sample in sample_ids if sample not in by_id]
00273 |         if missing:
00274 |             raise DeepHEquivalencePreflightError("Frozen split manifest is missing requested samples: " + ", ".join(missing))
00275 |         return [by_id[sample] for sample in sample_ids]
00276 | 
00277 |     selected: list[dict[str, Any]] = []
00278 |     seen: set[str] = set()
00279 |     for split in ("train", "validation", "test"):
00280 |         for row in rows:
00281 |             if row.get("split") == split:
00282 |                 sample_id = sample_id_from_row(row)
00283 |                 selected.append(row)
00284 |                 seen.add(sample_id)
00285 |                 break
00286 |     for row in rows:
00287 |         sample_id = sample_id_from_row(row)
00288 |         if sample_id not in seen:
00289 |             selected.append(row)
00290 |             seen.add(sample_id)
00291 |         if len(selected) >= sample_limit:
00292 |             break
00293 |     return selected[:sample_limit]
```

### `raw_reference_matrices` — líneas 390–404

```py
00390 | def raw_reference_matrices(reference_path: Path, kpoint: tuple[float, float, float]) -> dict[str, Any]:
00391 |     helpers = _runtime_helpers()
00392 |     np = helpers["np"]
00393 |     sile = helpers["sisl"].get_sile(str(reference_path))
00394 |     hamiltonian = sile.read_hamiltonian()
00395 |     h_k = helpers["kpoint_hamiltonian_matrix"](hamiltonian, kpoint)
00396 |     s_k = helpers["kpoint_overlap_matrix"](hamiltonian, kpoint)
00397 |     if s_k is None:
00398 |         raise DeepHEquivalencePreflightError(f"Reference overlap S(k) is unavailable for {reference_path}")
00399 |     return {
00400 |         "hamiltonian": np.asarray(h_k, dtype=np.complex128),
00401 |         "overlap": np.asarray(s_k, dtype=np.complex128),
00402 |         "spin": str(getattr(hamiltonian, "spin", "")) or "",
00403 |         "orthogonal": bool(getattr(hamiltonian, "orthogonal", False)),
00404 |     }
```

### `kpoints_from_fdf` — líneas 407–425

```py
00407 | def kpoints_from_fdf(path: Path | None) -> tuple[list[tuple[float, float, float]], list[str]]:
00408 |     warnings: list[str] = []
00409 |     if path is None or not path.exists():
00410 |         warnings.append("missing RUN.fdf; only Gamma was checked.")
00411 |         return [(0.0, 0.0, 0.0)], warnings
00412 |     helpers = _runtime_helpers()
00413 |     kgrid = helpers["parse_monkhorst_pack_kgrid"](path)
00414 |     kpoints: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
00415 |     if kgrid is None:
00416 |         warnings.append("RUN.fdf has no Monkhorst-Pack grid; Gamma was checked.")
00417 |         return kpoints, warnings
00418 |     if not kgrid.ok:
00419 |         warnings.append(f"RUN.fdf k-grid could not be parsed ({kgrid.error}); Gamma was checked.")
00420 |         return kpoints, warnings
00421 |     for kpoint in kgrid.fractional_kpoints:
00422 |         item = tuple(float(value) for value in kpoint)
00423 |         if item not in kpoints:
00424 |             kpoints.append(item)
00425 |     return kpoints, warnings
```

### `numeric_evidence_for_sample` — líneas 468–698

```py
00468 | def numeric_evidence_for_sample(
00469 |     *,
00470 |     row: dict[str, Any],
00471 |     manifest_dir: Path,
00472 |     graph2mat_result_dir: Path,
00473 |     processed_dir: Path,
00474 |     predictions_dir: Path,
00475 |     matrix_tolerance: float,
00476 |     eigenvalue_tolerance: float,
00477 |     command: list[str] | None,
00478 | ) -> tuple[dict[str, Any], Path | None, Path]:
00479 |     frozen_sample_id = sample_id_from_row(row)
00480 |     processed_sample = existing_child_for_sample(processed_dir, row)
00481 |     prediction_sample = existing_child_for_sample(predictions_dir, row)
00482 |     adapter_sample_id = prediction_sample.name if prediction_sample is not None else frozen_sample_id
00483 |     output_name = safe_sample_id(adapter_sample_id)
00484 |     source_files: dict[str, Any] = {}
00485 |     warnings: list[str] = []
00486 | 
00487 |     reference_path = graph2mat_reference_path(
00488 |         row=row,
00489 |         manifest_dir=manifest_dir,
00490 |         graph2mat_result_dir=graph2mat_result_dir,
00491 |     )
00492 |     fdf_path = run_fdf_path(row=row, manifest_dir=manifest_dir, graph2mat_result_dir=graph2mat_result_dir)
00493 |     source_files["raw_reference"] = file_entry(reference_path)
00494 |     source_files["run_fdf"] = file_entry(fdf_path)
00495 |     if reference_path is None:
00496 |         return (
00497 |             failed_evidence(
00498 |                 sample_id=adapter_sample_id,
00499 |                 frozen_sample_id=frozen_sample_id,
00500 |                 reason="missing raw SIESTA/Graph2Mat reference HSX/TSHS",
00501 |                 source_files=source_files,
00502 |                 command=command,
00503 |             ),
00504 |             prediction_sample,
00505 |             Path(output_name),
00506 |         )
00507 |     if processed_sample is None:
00508 |         return (
00509 |             failed_evidence(
00510 |                 sample_id=adapter_sample_id,
00511 |                 frozen_sample_id=frozen_sample_id,
00512 |                 reason="missing DeepH processed sample mapping",
00513 |                 source_files=source_files,
00514 |                 command=command,
00515 |             ),
00516 |             prediction_sample,
00517 |             Path(output_name),
00518 |         )
00519 |     if prediction_sample is None:
00520 |         return (
00521 |             failed_evidence(
00522 |                 sample_id=adapter_sample_id,
00523 |                 frozen_sample_id=frozen_sample_id,
00524 |                 reason="missing DeepH prediction sample mapping",
00525 |                 source_files=source_files,
00526 |                 command=command,
00527 |             ),
00528 |             prediction_sample,
00529 |             Path(output_name),
00530 |         )
00531 | 
00532 |     ref_h5 = processed_sample / "hamiltonians.h5"
00533 |     overlap_h5 = processed_sample / "overlaps.h5"
00534 |     orbital_types = processed_sample / "orbital_types.dat"
00535 |     prediction_h5 = prediction_sample / "hamiltonians_pred.h5"
00536 |     info_json = processed_sample / "info.json"
00537 |     orb_indx_path = artifact_path(row, "orb_indx", manifest_dir=manifest_dir)
00538 |     source_files.update(
00539 |         {
00540 |             "deeph_processed_hamiltonian": file_entry(ref_h5),
00541 |             "deeph_processed_overlap": file_entry(overlap_h5),
00542 |             "deeph_orbital_types": file_entry(orbital_types),
00543 |             "deeph_prediction": file_entry(prediction_h5),
00544 |             "deeph_info": file_entry(info_json),
00545 |             "siesta_orb_indx": file_entry(orb_indx_path),
00546 |         }
00547 |     )
00548 |     missing = [
00549 |         str(path)
00550 |         for path in (ref_h5, overlap_h5, orbital_types, prediction_h5, info_json)
00551 |         if not path.exists()
00552 |     ]
00553 |     if missing:
00554 |         return (
00555 |             failed_evidence(
00556 |                 sample_id=adapter_sample_id,
00557 |                 frozen_sample_id=frozen_sample_id,
00558 |                 reason="missing required DeepH artifact(s): " + ", ".join(missing),
00559 |                 source_files=source_files,
00560 |                 command=command,
00561 |             ),
00562 |             prediction_sample,
00563 |             Path(output_name),
00564 |         )
00565 | 
00566 |     try:
00567 |         helpers = _runtime_helpers()
00568 |         np = helpers["np"]
00569 |         eigenvalues = helpers["complex_generalized_eigenvalues"]
00570 |         orbital_counts = count_orbitals_from_orbital_types(orbital_types)
00571 |         n_orbitals = int(sum(orbital_counts))
00572 |         basis_transform = derive_deeph_to_siesta_basis_transform(
00573 |             row=row,
00574 |             manifest_dir=manifest_dir,
00575 |             orbital_types=orbital_types,
00576 |             n_orbitals=n_orbitals,
00577 |         )
00578 |         warnings.extend(str(item) for item in basis_transform.get("warnings") or [])
00579 |         ref_shapes = h5_block_shapes(ref_h5, orbital_counts, label="DeepH processed reference")
00580 |         pred_shapes = h5_block_shapes(prediction_h5, orbital_counts, label="DeepH prediction")
00581 |         overlap_shapes = h5_block_shapes(overlap_h5, orbital_counts, label="DeepH processed overlap")
00582 |         support_keys_match = set(ref_shapes) == set(pred_shapes) == set(overlap_shapes)
00583 |         kpoints, k_warnings = kpoints_from_fdf(fdf_path)
00584 |         warnings.extend(k_warnings)
00585 |         gamma_kpoint = kpoints[0]
00586 |         gamma_raw = raw_reference_matrices(reference_path, gamma_kpoint)
00587 |         gamma_deeph_h = apply_basis_transform(assemble_hk(ref_h5, orbital_types, gamma_kpoint), basis_transform)
00588 |         energy_reference_shift_eV = fit_energy_reference_shift(
00589 |             deeph_h=gamma_deeph_h,
00590 |             raw_h=gamma_raw["hamiltonian"],
00591 |             raw_s=gamma_raw["overlap"],
00592 |         )
00593 |         max_hk_error = 0.0
00594 |         max_s_error = 0.0
00595 |         max_eigen_error = 0.0
00596 |         support_match = True
00597 |         shape_match = True
00598 |         kpoint_rows: list[dict[str, float]] = []
00599 |         for kpoint in kpoints:
00600 |             raw = raw_reference_matrices(reference_path, kpoint)
00601 |             raw_h = raw["hamiltonian"]
00602 |             raw_s = raw["overlap"]
00603 |             deeph_h = apply_basis_transform(assemble_hk(ref_h5, orbital_types, kpoint), basis_transform)
00604 |             deeph_s = apply_basis_transform(assemble_hk(overlap_h5, orbital_types, kpoint), basis_transform)
00605 |             shape_match = shape_match and raw_h.shape == deeph_h.shape and raw_s.shape == deeph_s.shape
00606 |             if not shape_match:
00607 |                 continue
00608 |             deeph_h_aligned = np.asarray(deeph_h + energy_reference_shift_eV * raw_s)
00609 |             h_delta = np.asarray(deeph_h_aligned - raw_h)
00610 |             s_delta = np.asarray(deeph_s - raw_s)
00611 |             max_hk_error = max(max_hk_error, float(np.max(np.abs(h_delta))) if h_delta.size else 0.0)
00612 |             max_s_error = max(max_s_error, float(np.max(np.abs(s_delta))) if s_delta.size else 0.0)
00613 |             raw_eig = eigenvalues(raw_h, raw_s)
00614 |             deeph_eig = eigenvalues(deeph_h_aligned, raw_s)
00615 |             if raw_eig.shape != deeph_eig.shape:
00616 |                 shape_match = False
00617 |                 continue
00618 |             eig_delta = np.asarray(deeph_eig - raw_eig)
00619 |             max_eigen_error = max(max_eigen_error, float(np.max(np.abs(eig_delta))) if eig_delta.size else 0.0)
00620 |             support_match = support_match and bool(
00621 |                 np.array_equal(np.abs(raw_h) > SUPPORT_THRESHOLD, np.abs(deeph_h_aligned) > SUPPORT_THRESHOLD)
00622 |             )
00623 |             kpoint_rows.append({"kx": float(kpoint[0]), "ky": float(kpoint[1]), "kz": float(kpoint[2])})
00624 |         hk_pass = shape_match and max_hk_error <= matrix_tolerance
00625 |         s_pass = shape_match and max_s_error <= matrix_tolerance
00626 |         eig_pass = shape_match and max_eigen_error <= eigenvalue_tolerance
00627 |         support_pass = bool(shape_match and support_keys_match and support_match)
00628 |         info_payload = read_json(info_json)
00629 |         spin_pass = "isspinful" in info_payload
00630 |         if not spin_pass:
00631 |             warnings.append(f"DeepH info.json does not expose isspinful: {info_json}")
00632 |         checks = {
00633 |             "shape": _pass_check(shape_match, "raw and DeepH H(k)/S(k) shapes match" if shape_match else "matrix shape mismatch"),
00634 |             "units": _pass_check(hk_pass, "DeepH processed Hamiltonian numerically matches raw reference units"),
00635 |             "orbital_order": _pass_check(hk_pass and s_pass, "numeric H(k)/S(k) equality proves orbital ordering for checked k-points"),
00636 |             "atom_order": _pass_check(hk_pass and s_pass, "block assembly numerically matches raw reference for checked k-points"),
00637 |             "r_vectors": _pass_check(hk_pass and s_pass, "DeepH R-vector phases reproduce raw reference H(k)/S(k)"),
00638 |             "spin": _pass_check(spin_pass, "DeepH spin metadata is present"),
00639 |             "sparse_support": _pass_check(support_pass, "DeepH block and dense support match raw reference"),
00640 |             "hk": _pass_check(hk_pass, "DeepH processed H(k) matches raw reference"),
00641 |             "s_ref": _pass_check(s_pass, "DeepH processed S(k) matches raw reference overlap"),
00642 |             "eigenvalues": _pass_check(eig_pass, "generalized eigenvalues match with S_ref(k)"),
00643 |         }
00644 |         errors = {
00645 |             "max_abs_hk_error_eV": max_hk_error,
00646 |             "max_abs_s_ref_error": max_s_error,
00647 |             "max_abs_eigenvalue_error_eV": max_eigen_error,
00648 |             "energy_reference_shift_eV": energy_reference_shift_eV,
00649 |         }
00650 |         tolerances = {
00651 |             "max_abs_hk_error_eV": matrix_tolerance,
00652 |             "max_abs_s_ref_error": matrix_tolerance,
00653 |             "max_abs_eigenvalue_error_eV": eigenvalue_tolerance,
00654 |         }
00655 |         proven = all(str(check["status"]) == "pass" for check in checks.values())
00656 |     except Exception as exc:
00657 |         return (
00658 |             failed_evidence(
00659 |                 sample_id=adapter_sample_id,
00660 |                 frozen_sample_id=frozen_sample_id,
00661 |                 reason=str(exc),
00662 |                 warnings=warnings,
00663 |                 source_files=source_files,
00664 |                 command=command,
00665 |             ),
00666 |             prediction_sample,
00667 |             Path(output_name),
00668 |         )
00669 | 
00670 |     payload = {
00671 |         "schema": RAW_GLOBAL_EQUIVALENCE_EVIDENCE_SCHEMA,
00672 |         "sample_id": adapter_sample_id,
00673 |         "frozen_sample_id": frozen_sample_id,
00674 |         "equivalence_status": "proven" if proven else "failed",
00675 |         "equivalence_scope": "raw_global",
00676 |         "checks": checks,
00677 |         "errors": errors,
00678 |         "tolerances": tolerances,
00679 |         "source_files": source_files,
00680 |         "kpoints_checked": kpoint_rows,
00681 |         "basis_transform": basis_transform,
00682 |         "energy_reference_alignment": {
00683 |             "policy": "least_squares_shift_from_first_kpoint: H_raw ~= H_deeph_converted + c*S_ref",
00684 |             "shift_eV": energy_reference_shift_eV,
00685 |             "fit_kpoint": {
00686 |                 "kx": float(gamma_kpoint[0]),
00687 |                 "ky": float(gamma_kpoint[1]),
00688 |                 "kz": float(gamma_kpoint[2]),
00689 |             },
00690 |         },
00691 |         "generator": {
00692 |             "script": Path(__file__).name,
00693 |             "command": command or sys.argv,
00694 |             "comparison_policy": "DeepH processed hamiltonians.h5/overlaps.h5 versus raw SIESTA HSX/TSHS reference",
00695 |         },
00696 |         "warnings": warnings,
00697 |     }
00698 |     return payload, prediction_sample, Path(output_name)
```

### `build_preflight_manifest` — líneas 712–781

```py
00712 | def build_preflight_manifest(
00713 |     *,
00714 |     frozen_split_manifest: Path,
00715 |     graph2mat_result_dir: Path,
00716 |     deeph_processed_dir: Path,
00717 |     deeph_predictions_dir: Path,
00718 |     output_dir: Path,
00719 |     sample_limit: int,
00720 |     sample_ids: list[str] | None = None,
00721 |     matrix_tolerance: float = DEFAULT_MATRIX_TOLERANCE,
00722 |     eigenvalue_tolerance: float = DEFAULT_EIGENVALUE_TOLERANCE,
00723 |     command: list[str] | None = None,
00724 | ) -> dict[str, Any]:
00725 |     output_dir.mkdir(parents=True, exist_ok=True)
00726 |     frozen = read_json(frozen_split_manifest)
00727 |     rows = [row for row in frozen.get("rows") or [] if isinstance(row, dict)]
00728 |     selected = select_frozen_rows(rows, sample_ids=list(sample_ids or []), sample_limit=sample_limit)
00729 |     sample_results: list[dict[str, Any]] = []
00730 |     for row in selected:
00731 |         evidence, prediction_sample, output_name = numeric_evidence_for_sample(
00732 |             row=row,
00733 |             manifest_dir=frozen_split_manifest.parent,
00734 |             graph2mat_result_dir=graph2mat_result_dir,
00735 |             processed_dir=deeph_processed_dir,
00736 |             predictions_dir=deeph_predictions_dir,
00737 |             matrix_tolerance=matrix_tolerance,
00738 |             eigenvalue_tolerance=eigenvalue_tolerance,
00739 |             command=command,
00740 |         )
00741 |         evidence_dir = output_dir / output_name.name
00742 |         evidence_path = evidence_dir / RAW_GLOBAL_EQUIVALENCE_EVIDENCE_FILENAME
00743 |         paths = install_evidence(evidence, evidence_path, prediction_sample)
00744 |         adapter_validation = validate_raw_global_equivalence_evidence(
00745 |             Path(paths["adapter_discoverable"] if "adapter_discoverable" in paths else paths["output"]),
00746 |             sample_id=str(evidence.get("sample_id") or ""),
00747 |         )
00748 |         sample_results.append(
00749 |             {
00750 |                 "sample_id": evidence.get("sample_id"),
00751 |                 "frozen_sample_id": evidence.get("frozen_sample_id"),
00752 |                 "status": evidence.get("equivalence_status"),
00753 |                 "equivalence_scope": evidence.get("equivalence_scope"),
00754 |                 "evidence_paths": paths,
00755 |                 "adapter_validation_status": adapter_validation.get("status"),
00756 |                 "adapter_validation_reason": adapter_validation.get("reason"),
00757 |                 "warnings": evidence.get("warnings") or [],
00758 |             }
00759 |         )
00760 |     proven_count = sum(1 for row in sample_results if row.get("status") == "proven")
00761 |     failed_count = len(sample_results) - proven_count
00762 |     aggregate = {
00763 |         "schema": PREFLIGHT_SCHEMA,
00764 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
00765 |         "frozen_split_manifest": str(frozen_split_manifest),
00766 |         "graph2mat_result_dir": str(graph2mat_result_dir),
00767 |         "deeph_processed_dir": str(deeph_processed_dir),
00768 |         "deeph_predictions_dir": str(deeph_predictions_dir),
00769 |         "output_dir": str(output_dir),
00770 |         "sample_limit": sample_limit,
00771 |         "requested_sample_ids": list(sample_ids or []),
00772 |         "matrix_tolerance": matrix_tolerance,
00773 |         "eigenvalue_tolerance": eigenvalue_tolerance,
00774 |         "samples_seen": len(selected),
00775 |         "samples_proven": proven_count,
00776 |         "samples_failed": failed_count,
00777 |         "status": "proven" if sample_results and failed_count == 0 else "failed",
00778 |         "samples": sample_results,
00779 |     }
00780 |     write_json(output_dir / "deeph_raw_global_equivalence_preflight.json", aggregate)
00781 |     return aggregate
```

## `Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/artifact_validation.json` — vista compacta

SHA-256 del JSON completo: `9eb3a7729af37143695e7bbf2a39c74eb9f70b2725ce54db4eccff4a7a78f115`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "artifact_validation_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/artifact_validation.json",
00003 |   "basis_present": true,
00004 |   "benchmark_ready": true,
00005 |   "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00006 |   "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps",
00007 |   "errors": [],
00008 |   "generation_contract_diagnostics": {
00009 |     "required_fdf_output_flags": [
00010 |       "SaveHS",
00011 |       "Save.HS",
00012 |       "TS.HS.Save",
00013 |       "TS.DE.Save",
00014 |       "XML.Write",
00015 |       "Write.OrbitalIndex"
00016 |     ],
00017 |     "store_file_patterns": [
00018 |       "*fdf",
00019 |       "*TSHS",
00020 |       "*TSDE",
00021 |       "*XV",
00022 |       "*HSX",
00023 |       "*STRUCT_OUT",
00024 |       "*ORB_INDX",
00025 |       "*out"
00026 |     ],
00027 |     "store_files": "*fdf *TSHS *TSDE *XV *HSX *STRUCT_OUT *ORB_INDX *out"
00028 |   },
00029 |   "invalid_snapshots": 0,
00030 |   "material_identity_present": true,
00031 |   "pseudopotential_provenance_present": true,
00032 |   "repair_required_snapshots": 0,
00033 |   "scientific_status": "benchmark_ready",
00034 |   "siesta_input_provenance_present": true,
00035 |   "snapshots": {
00036 |     "_list_length": 1000,
00037 |     "_first_two": [
00038 |       {
00039 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00040 |         "errors": [],
00041 |         "missing_required": [],
00042 |         "present_artifacts": {
00043 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.HSX",
00044 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/metadata.json",
00045 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.ORB_INDX",
00046 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/RUN.fdf",
00047 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/RUN.out",
00048 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.STRUCT_OUT",
00049 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.TSDE",
00050 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.TSHS",
00051 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.XV"
00052 |         },
00053 |         "repair_required": false,
00054 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0",
00055 |         "system_label": "graphene",
00056 |         "valid": true,
00057 |         "warnings": []
00058 |       },
00059 |       {
00060 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00061 |         "errors": [],
00062 |         "missing_required": [],
00063 |         "present_artifacts": {
00064 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.HSX",
00065 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/metadata.json",
00066 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.ORB_INDX",
00067 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/RUN.fdf",
00068 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/RUN.out",
00069 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.STRUCT_OUT",
00070 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.TSDE",
00071 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.TSHS",
00072 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.XV"
00073 |         },
00074 |         "repair_required": false,
00075 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1",
00076 |         "system_label": "graphene",
00077 |         "valid": true,
00078 |         "warnings": []
00079 |       }
00080 |     ],
00081 |     "_last_two": [
00082 |       {
00083 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00084 |         "errors": [],
00085 |         "missing_required": [],
00086 |         "present_artifacts": {
00087 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.HSX",
00088 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/metadata.json",
00089 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.ORB_INDX",
00090 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/RUN.fdf",
00091 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/RUN.out",
00092 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.STRUCT_OUT",
00093 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.TSDE",
00094 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.TSHS",
00095 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.XV"
00096 |         },
00097 |         "repair_required": false,
00098 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998",
00099 |         "system_label": "graphene",
00100 |         "valid": true,
00101 |         "warnings": []
00102 |       },
00103 |       {
00104 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00105 |         "errors": [],
00106 |         "missing_required": [],
00107 |         "present_artifacts": {
00108 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.HSX",
00109 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/metadata.json",
00110 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.ORB_INDX",
00111 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/RUN.fdf",
00112 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/RUN.out",
00113 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.STRUCT_OUT",
00114 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.TSDE",
00115 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.TSHS",
00116 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.XV"
00117 |         },
00118 |         "repair_required": false,
00119 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999",
00120 |         "system_label": "graphene",
00121 |         "valid": true,
00122 |         "warnings": []
00123 |       }
00124 |     ]
00125 |   },
00126 |   "steps_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps",
00127 |   "total_snapshots": 1000,
00128 |   "valid": true,
00129 |   "valid_snapshots": 1000,
00130 |   "warnings": []
00131 | }
```

## `Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/benchmark_dataset_manifest.json` — vista compacta

SHA-256 del JSON completo: `edf631868c439086a8c03b51b469890ff5b502692692fda43d1c717202f8a1de`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
00003 |   "artifact_validation": {
00004 |     "artifact_validation_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/artifact_validation.json",
00005 |     "basis_present": true,
00006 |     "benchmark_ready": true,
00007 |     "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00008 |     "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps",
00009 |     "errors": [],
00010 |     "generation_contract_diagnostics": {
00011 |       "required_fdf_output_flags": [
00012 |         "SaveHS",
00013 |         "Save.HS",
00014 |         "TS.HS.Save",
00015 |         "TS.DE.Save",
00016 |         "XML.Write",
00017 |         "Write.OrbitalIndex"
00018 |       ],
00019 |       "store_file_patterns": [
00020 |         "*fdf",
00021 |         "*TSHS",
00022 |         "*TSDE",
00023 |         "*XV",
00024 |         "*HSX",
00025 |         "*STRUCT_OUT",
00026 |         "*ORB_INDX",
00027 |         "*out"
00028 |       ],
00029 |       "store_files": "*fdf *TSHS *TSDE *XV *HSX *STRUCT_OUT *ORB_INDX *out"
00030 |     },
00031 |     "invalid_snapshots": 0,
00032 |     "material_identity_present": true,
00033 |     "pseudopotential_provenance_present": true,
00034 |     "repair_required_snapshots": 0,
00035 |     "scientific_status": "benchmark_ready",
00036 |     "siesta_input_provenance_present": true,
00037 |     "snapshots": {
00038 |       "_list_length": 1000,
00039 |       "_first_two": [
00040 |         {
00041 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00042 |           "errors": [],
00043 |           "missing_required": [],
00044 |           "present_artifacts": {
00045 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.HSX",
00046 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/metadata.json",
00047 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.ORB_INDX",
00048 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/RUN.fdf",
00049 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/RUN.out",
00050 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.STRUCT_OUT",
00051 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.TSDE",
00052 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.TSHS",
00053 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.XV"
00054 |           },
00055 |           "repair_required": false,
00056 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0",
00057 |           "system_label": "graphene",
00058 |           "valid": true,
00059 |           "warnings": []
00060 |         },
00061 |         {
00062 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00063 |           "errors": [],
00064 |           "missing_required": [],
00065 |           "present_artifacts": {
00066 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.HSX",
00067 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/metadata.json",
00068 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.ORB_INDX",
00069 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/RUN.fdf",
00070 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/RUN.out",
00071 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.STRUCT_OUT",
00072 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.TSDE",
00073 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.TSHS",
00074 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.XV"
00075 |           },
00076 |           "repair_required": false,
00077 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1",
00078 |           "system_label": "graphene",
00079 |           "valid": true,
00080 |           "warnings": []
00081 |         }
00082 |       ],
00083 |       "_last_two": [
00084 |         {
00085 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00086 |           "errors": [],
00087 |           "missing_required": [],
00088 |           "present_artifacts": {
00089 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.HSX",
00090 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/metadata.json",
00091 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.ORB_INDX",
00092 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/RUN.fdf",
00093 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/RUN.out",
00094 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.STRUCT_OUT",
00095 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.TSDE",
00096 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.TSHS",
00097 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.XV"
00098 |           },
00099 |           "repair_required": false,
00100 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998",
00101 |           "system_label": "graphene",
00102 |           "valid": true,
00103 |           "warnings": []
00104 |         },
00105 |         {
00106 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00107 |           "errors": [],
00108 |           "missing_required": [],
00109 |           "present_artifacts": {
00110 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.HSX",
00111 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/metadata.json",
00112 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.ORB_INDX",
00113 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/RUN.fdf",
00114 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/RUN.out",
00115 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.STRUCT_OUT",
00116 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.TSDE",
00117 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.TSHS",
00118 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.XV"
00119 |           },
00120 |           "repair_required": false,
00121 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999",
00122 |           "system_label": "graphene",
00123 |           "valid": true,
00124 |           "warnings": []
00125 |         }
00126 |       ]
00127 |     },
00128 |     "steps_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps",
00129 |     "total_snapshots": 1000,
00130 |     "valid": true,
00131 |     "valid_snapshots": 1000,
00132 |     "warnings": []
00133 |   },
00134 |   "basis_hashes": {
00135 |     "C.ion.xml": "6740d9f56df9f2d42ff27e6a7abd9b7b0224a49cbda58b52b7f38136bcfc8b6f",
00136 |     "Ghost-H.ion.xml": "bd23b909b0cb2de1f9c9cfa421ca2448e6e2934f4e0102a55ceb0357bcf45ca5"
00137 |   },
00138 |   "benchmark_dataset_id": "joint_graph2mat_deeph_368c03fe7191949d",
00139 |   "benchmark_ready": true,
00140 |   "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000",
00141 |   "deeph_pack_commit": "",
00142 |   "environment": {
00143 |     "platform": "Linux-6.17.0-23-generic-x86_64-with-glibc2.39",
00144 |     "python_version": "3.12.3"
00145 |   },
00146 |   "frozen_split_manifest": {
00147 |     "path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/frozen_split_manifest.json",
00148 |     "split_counts": {
00149 |       "test": 100,
00150 |       "train": 798,
00151 |       "validation": 100
00152 |     },
00153 |     "split_hash": "acc00551ed4a5ce2be843e7c0280c22fa76e0434f9d5ea6cd83314651b1e8aa6",
00154 |     "valid": true
00155 |   },
00156 |   "generation_mode": "clean_one_pass",
00157 |   "graph2mat_commit": "76101aff7113581f9808434c93992fd08c9fba8d",
00158 |   "kpoint_summary": {
00159 |     "kgrid_monkhorst_pack": [
00160 |       "20   0   0  0.0",
00161 |       "0  20   0  0.0",
00162 |       "0   0   1  0.0"
00163 |     ],
00164 |     "present": true
00165 |   },
00166 |   "material_label": "graphene",
00167 |   "material_source": {
00168 |     "absolute_paths_used": false,
00169 |     "basis_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/basis",
00170 |     "basis_file_sha256": {
00171 |       "C.ion.xml": "6740d9f56df9f2d42ff27e6a7abd9b7b0224a49cbda58b52b7f38136bcfc8b6f",
00172 |       "Ghost-H.ion.xml": "bd23b909b0cb2de1f9c9cfa421ca2448e6e2934f4e0102a55ceb0357bcf45ca5"
00173 |     },
00174 |     "deeph_pack_commit": "",
00175 |     "environment": {
00176 |       "platform": "Linux-6.17.0-23-generic-x86_64-with-glibc2.39",
00177 |       "python_version": "3.12.3",
00178 |       "reconstructed_after_generation": true,
00179 |       "reconstruction_note": "Original generation environment lockfile was not present; this records current host/runtime only."
00180 |     },
00181 |     "fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/RUN.fdf",
00182 |     "fdf_sha256": "faf623f1bf7f3d9cf73b191d291c042f2aff43b828a8800119e2eaa8b5379494",
00183 |     "graph2mat_commit": "76101aff7113581f9808434c93992fd08c9fba8d",
00184 |     "label": "graphene",
00185 |     "material_source": "explicit_preset",
00186 |     "preset": "graphene",
00187 |     "provenance_reconstruction": {
00188 |       "backup_original": "material_provenance.pre_reconstructed_20260529_133043.json",
00189 |       "created_at": "2026-05-29T11:30:43.702128+00:00",
00190 |       "limitations": [
00191 |         "Original exact shell command was not recorded in the existing dataset manifest.",
00192 |         "Environment is reconstructed from the current host, not an original lockfile."
00193 |       ],
00194 |       "notes": [
00195 |         "siesta_stdout_path stored as an absolute path because runner validates MD_steps as snapshot_root while dataset verifier validates dataset_root."
00196 |       ],
00197 |       "source_files": [
00198 |         "RUN.out",
00199 |         "RUN.fdf"
00200 |       ]
00201 |     },
00202 |     "pseudopotential_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/pseudos",
00203 |     "pseudopotential_sha256": {
00204 |       "C": "835287a894d851d30c2c613edcc4124c493a08dad74a2aa1a982266c15d1a0e6",
00205 |       "Ghost-H": "50878f06f8171e9a243eeaaedd49461a1e72d9d015f66f66a8158f14116aefdb"
00206 |     },
00207 |     "pseudopotentials": {
00208 |       "C": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/pseudos/C.psf",
00209 |       "Ghost-H": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/pseudos/Ghost-H.psf"
00210 |     },
00211 |     "pseudopotentials_copied_to_dataset": {
00212 |       "C": "C.psf",
00213 |       "Ghost-H": "Ghost-H.psf"
00214 |     },
00215 |     "pseudopotentials_verified_in_dataset": {},
00216 |     "run_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/RUN.out",
00217 |     "siesta_build_info": {
00218 |       "architecture": "x86_64",
00219 |       "compiler_flags": "-fallow-argument-mismatch -O3 -march=native",
00220 |       "compiler_version": "GNU-13.3.0",
00221 |       "executable": "siesta",
00222 |       "parallelisations": "MPI",
00223 |       "source_file": "RUN.out",
00224 |       "version": "5.4.2-11-g4e9a46060"
00225 |     },
00226 |     "siesta_command_line": "siesta < RUN.fdf > RUN.out (reconstructed from dataset layout and SIESTA stdout; original launch shell command was not recorded)",
00227 |     "siesta_executable": "siesta",
00228 |     "siesta_stdout_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/RUN.out",
00229 |     "siesta_version": "5.4.2-11-g4e9a46060",
00230 |     "siesta_version_source_file": "RUN.out",
00231 |     "species": [
00232 |       {
00233 |         "atomic_number": 6,
00234 |         "index": 1,
00235 |         "label": "C"
00236 |       },
00237 |       {
00238 |         "atomic_number": -1,
00239 |         "index": 2,
00240 |         "label": "Ghost-H"
00241 |       }
00242 |     ],
00243 |     "structure_type": "crystal",
00244 |     "warning": null
00245 |   },
00246 |   "provenance_status": {
00247 |     "basis_provenance": true,
00248 |     "material_identity": true,
00249 |     "missing": [],
00250 |     "pseudopotential_provenance": true,
00251 |     "siesta_command_line_provenance": true,
00252 |     "siesta_environment_provenance": true,
00253 |     "siesta_execution_log_provenance": true,
00254 |     "siesta_input_provenance": true,
00255 |     "siesta_version_provenance": true,
00256 |     "strict_paper_ready": true,
00257 |     "valid": true
00258 |   },
00259 |   "pseudopotential_hashes": {
00260 |     "C": "835287a894d851d30c2c613edcc4124c493a08dad74a2aa1a982266c15d1a0e6",
00261 |     "Ghost-H": "50878f06f8171e9a243eeaaedd49461a1e72d9d015f66f66a8158f14116aefdb"
00262 |   },
00263 |   "samples": {
00264 |     "_list_length": 1000,
00265 |     "_first_two": [
00266 |       {
00267 |         "artifact_paths": {
00268 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.HSX",
00269 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/metadata.json",
00270 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.ORB_INDX",
00271 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/RUN.fdf",
00272 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/RUN.out",
00273 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.STRUCT_OUT",
00274 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.TSDE",
00275 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.TSHS",
00276 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0/graphene.XV"
00277 |         },
00278 |         "artifact_sha256": {
00279 |           "hsx": "ccdaf5768d19b3e25bfc4e5922a8df218d4ef32b33d23f13aebd9a80ec1e332f",
00280 |           "metadata": "95c15080120ad2dfad0bad890a25944ca491395f93efd11050eb63f5e9c3262d",
00281 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00282 |           "run_fdf": "3e921183a07f6e77f79d52b4130317f38284a1aec0079c6246ae450929c75f5c",
00283 |           "run_output": "cbdcec8f07c286966771f901ae7549f641607738f24196b92e3981230ceca6f1",
00284 |           "struct_out": "3c473d6090f51945e532e184fa1309b3b1535236fccb3ca0d3ce25db39810d89",
00285 |           "tsde": "039883e2a7e89c59efcbd8e3d0645b03756d3dd4793209831b8da64d3b941676",
00286 |           "tshs": "b64b166be3bf7d223ba41a91e4cbeaa1e6b589ae0b9869d52ff322adaf5991d9",
00287 |           "xv": "35583661150f4f75cdc23e36ba6fca1d18078e5e64d120eac58e4c4d140d7182"
00288 |         },
00289 |         "errors": [],
00290 |         "missing_required": [],
00291 |         "repair_required": false,
00292 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/0",
00293 |         "system_label": "graphene",
00294 |         "valid": true,
00295 |         "warnings": []
00296 |       },
00297 |       {
00298 |         "artifact_paths": {
00299 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.HSX",
00300 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/metadata.json",
00301 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.ORB_INDX",
00302 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/RUN.fdf",
00303 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/RUN.out",
00304 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.STRUCT_OUT",
00305 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.TSDE",
00306 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.TSHS",
00307 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1/graphene.XV"
00308 |         },
00309 |         "artifact_sha256": {
00310 |           "hsx": "0093b7a77eed41689181fcff11e89695dff7c2ea7e5e0bfff4aed084d9167047",
00311 |           "metadata": "3a257afca0aa9bdb7c49b786a09503d519e2b5f049282b854c05143b327ebdb2",
00312 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00313 |           "run_fdf": "ffae5bb676271701eaafc719efa7b009e1c71869e0673542fd0279b6991be59c",
00314 |           "run_output": "9d5db4fbaec41ce53c006b021a805636b28a8e2f6d3bac7a19d0ffeb49a9acd3",
00315 |           "struct_out": "0a0802c0b27e2e979a2f4083fb54db681cf79870d9c711452e976e086e26d708",
00316 |           "tsde": "8a3a1f6ecfa9c9332e66aec1f64d437af31984c146f5b169758c190d3a130d30",
00317 |           "tshs": "aba6738da92ac66d80d3334d71dbca7b71aa19b9d6e36bb3716ed3826b21af20",
00318 |           "xv": "dd4cfdb0c170a546d1eeff6f6f9ac4438a6f12cd23ec6384750e66574585cc43"
00319 |         },
00320 |         "errors": [],
00321 |         "missing_required": [],
00322 |         "repair_required": false,
00323 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/1",
00324 |         "system_label": "graphene",
00325 |         "valid": true,
00326 |         "warnings": []
00327 |       }
00328 |     ],
00329 |     "_last_two": [
00330 |       {
00331 |         "artifact_paths": {
00332 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.HSX",
00333 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/metadata.json",
00334 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.ORB_INDX",
00335 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/RUN.fdf",
00336 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/RUN.out",
00337 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.STRUCT_OUT",
00338 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.TSDE",
00339 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.TSHS",
00340 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998/graphene.XV"
00341 |         },
00342 |         "artifact_sha256": {
00343 |           "hsx": "aa8bc3ad8c3ad788189027d5c4c5eea44308c72b89f976f3c8188cb098d671b8",
00344 |           "metadata": "e3b01b8394838770c812044c610b348764483918d0e3b5c27643f58014773a5c",
00345 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00346 |           "run_fdf": "5c4a7542b42437806da7560afc1e06e02f884ae3f5dbc43a281938eb0836a99a",
00347 |           "run_output": "0a884b8f07e47dada3be3325f7acb28d21168f289596e4d4adecf18feb5115b8",
00348 |           "struct_out": "229b72b44f72489a33f159432c7f4c913112368008a67ae844de50bd1e9a49de",
00349 |           "tsde": "7eaddda772e4a765badf3975813b86d4dcad0465cde041ffe546c5dac1c9aa07",
00350 |           "tshs": "876cb7005e1d0c3d62e3fe261679b9e375580e78b02b58864b396b5a1543510e",
00351 |           "xv": "31229cfe055db2e4264a45b10494c50f3db54a6731b88c20526aad33a0eab334"
00352 |         },
00353 |         "errors": [],
00354 |         "missing_required": [],
00355 |         "repair_required": false,
00356 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/998",
00357 |         "system_label": "graphene",
00358 |         "valid": true,
00359 |         "warnings": []
00360 |       },
00361 |       {
00362 |         "artifact_paths": {
00363 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.HSX",
00364 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/metadata.json",
00365 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.ORB_INDX",
00366 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/RUN.fdf",
00367 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/RUN.out",
00368 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.STRUCT_OUT",
00369 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.TSDE",
00370 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.TSHS",
00371 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999/graphene.XV"
00372 |         },
00373 |         "artifact_sha256": {
00374 |           "hsx": "d9d64957733926c96ae4b820e067b9892c5fea3b065aa0c318714c075f3f94f3",
00375 |           "metadata": "d8f8ae74738976c57dbdf2151572c8d868de9828a7f04c7fefa3f2082d09f80b",
00376 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00377 |           "run_fdf": "549772617404d776f2103dc81f2f41402696a6a4cf59a5f2744d68d80654b414",
00378 |           "run_output": "ba00e89b317d0aacb33ce04815c4edd31e001cb8d6b4b219b7bc895523daed31",
00379 |           "struct_out": "2ba10ead272a6f580178b6efb3331b9634fe6e6c3d5b886fd1b57f6f300882d2",
00380 |           "tsde": "a1cc7acb74773917e9eabf5c50d89be1bc3afad100e424a10c8fd1b73b24470e",
00381 |           "tshs": "a86e7942fe389cad7ab549ba8d80774f5098a8ff298167c78d31d8cb5debc54a",
00382 |           "xv": "f40efa58359c198c90eb6350f42a178b5c4173e5a3eacf6c53840fb47e332b08"
00383 |         },
00384 |         "errors": [],
00385 |         "missing_required": [],
00386 |         "repair_required": false,
00387 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/MD_steps/999",
00388 |         "system_label": "graphene",
00389 |         "valid": true,
00390 |         "warnings": []
00391 |       }
00392 |     ]
00393 |   },
00394 |   "schema": "joint_graph2mat_deeph_benchmark_manifest_v1",
00395 |   "siesta_build_info": {
00396 |     "architecture": "x86_64",
00397 |     "compiler_flags": "-fallow-argument-mismatch -O3 -march=native",
00398 |     "compiler_version": "GNU-13.3.0",
00399 |     "executable": "siesta",
00400 |     "parallelisations": "MPI",
00401 |     "source_file": "RUN.out",
00402 |     "version": "5.4.2-11-g4e9a46060"
00403 |   },
00404 |   "siesta_command_line": "siesta < RUN.fdf > RUN.out (reconstructed from dataset layout and SIESTA stdout; original launch shell command was not recorded)",
00405 |   "siesta_executable": "siesta",
00406 |   "siesta_flags": {
00407 |     "Save.HS": "T",
00408 |     "SaveHS": "true",
00409 |     "TS.DE.Save": "T",
00410 |     "TS.HS.Save": "T",
00411 |     "Write.OrbitalIndex": "T",
00412 |     "XML.Write": "T"
00413 |   },
00414 |   "siesta_input_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/RUN.fdf",
00415 |   "siesta_input_sha256": "fdabde9796800349afcad807c1c3892afbd67f9fd0de3a3dd4d8d04e9d749749",
00416 |   "siesta_returncode": null,
00417 |   "siesta_stdout_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/RUN.out",
00418 |   "siesta_version": "5.4.2-11-g4e9a46060",
00419 |   "siesta_version_source_file": "RUN.out",
00420 |   "spin_summary": {},
00421 |   "system_label": "graphene",
00422 |   "validation_status": "valid",
00423 |   "warnings": []
00424 | }
```

## `Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/frozen_split_manifest.json` — vista compacta

SHA-256 del JSON completo: `fccfbdac90fb06b7976dfd11836383421d230c2493a8536f8a095cd1123fcf64`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
00003 |   "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000",
00004 |   "rows": {
00005 |     "_list_length": 998,
00006 |     "_first_two": [
00007 |       {
00008 |         "artifact_paths": {
00009 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/metadata.json",
00010 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.ORB_INDX",
00011 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.HSX",
00012 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.TSDE",
00013 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.TSHS",
00014 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/RUN.fdf",
00015 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/RUN.out",
00016 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.STRUCT_OUT",
00017 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.XV"
00018 |         },
00019 |         "artifact_sha256": {
00020 |           "metadata": "95c15080120ad2dfad0bad890a25944ca491395f93efd11050eb63f5e9c3262d",
00021 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00022 |           "reference_hsx": "ccdaf5768d19b3e25bfc4e5922a8df218d4ef32b33d23f13aebd9a80ec1e332f",
00023 |           "reference_tsde": "039883e2a7e89c59efcbd8e3d0645b03756d3dd4793209831b8da64d3b941676",
00024 |           "reference_tshs": "b64b166be3bf7d223ba41a91e4cbeaa1e6b589ae0b9869d52ff322adaf5991d9",
00025 |           "run_fdf": "3e921183a07f6e77f79d52b4130317f38284a1aec0079c6246ae450929c75f5c",
00026 |           "run_output": "cbdcec8f07c286966771f901ae7549f641607738f24196b92e3981230ceca6f1",
00027 |           "struct_out": "3c473d6090f51945e532e184fa1309b3b1535236fccb3ca0d3ce25db39810d89",
00028 |           "xv": "35583661150f4f75cdc23e36ba6fca1d18078e5e64d120eac58e4c4d140d7182"
00029 |         },
00030 |         "deeph_sample_id": "md_0",
00031 |         "graph2mat_sample_id": "md_0",
00032 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/metadata.json",
00033 |         "metadata_sha256": "95c15080120ad2dfad0bad890a25944ca491395f93efd11050eb63f5e9c3262d",
00034 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.ORB_INDX",
00035 |         "orb_indx_sha256": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00036 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.HSX",
00037 |         "reference_hsx_sha256": "ccdaf5768d19b3e25bfc4e5922a8df218d4ef32b33d23f13aebd9a80ec1e332f",
00038 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.TSDE",
00039 |         "reference_tsde_sha256": "039883e2a7e89c59efcbd8e3d0645b03756d3dd4793209831b8da64d3b941676",
00040 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.TSHS",
00041 |         "reference_tshs_sha256": "b64b166be3bf7d223ba41a91e4cbeaa1e6b589ae0b9869d52ff322adaf5991d9",
00042 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/RUN.fdf",
00043 |         "run_fdf_sha256": "3e921183a07f6e77f79d52b4130317f38284a1aec0079c6246ae450929c75f5c",
00044 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/RUN.out",
00045 |         "run_output_sha256": "cbdcec8f07c286966771f901ae7549f641607738f24196b92e3981230ceca6f1",
00046 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0",
00047 |         "sample_id": "md_0",
00048 |         "split": "train",
00049 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.STRUCT_OUT",
00050 |         "struct_out_sha256": "3c473d6090f51945e532e184fa1309b3b1535236fccb3ca0d3ce25db39810d89",
00051 |         "system_label": "graphene",
00052 |         "valid": true,
00053 |         "validation_problems": [],
00054 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/0/graphene.XV",
00055 |         "xv_sha256": "35583661150f4f75cdc23e36ba6fca1d18078e5e64d120eac58e4c4d140d7182"
00056 |       },
00057 |       {
00058 |         "artifact_paths": {
00059 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/metadata.json",
00060 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.ORB_INDX",
00061 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.HSX",
00062 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.TSDE",
00063 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.TSHS",
00064 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/RUN.fdf",
00065 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/RUN.out",
00066 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.STRUCT_OUT",
00067 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.XV"
00068 |         },
00069 |         "artifact_sha256": {
00070 |           "metadata": "3a257afca0aa9bdb7c49b786a09503d519e2b5f049282b854c05143b327ebdb2",
00071 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00072 |           "reference_hsx": "0093b7a77eed41689181fcff11e89695dff7c2ea7e5e0bfff4aed084d9167047",
00073 |           "reference_tsde": "8a3a1f6ecfa9c9332e66aec1f64d437af31984c146f5b169758c190d3a130d30",
00074 |           "reference_tshs": "aba6738da92ac66d80d3334d71dbca7b71aa19b9d6e36bb3716ed3826b21af20",
00075 |           "run_fdf": "ffae5bb676271701eaafc719efa7b009e1c71869e0673542fd0279b6991be59c",
00076 |           "run_output": "9d5db4fbaec41ce53c006b021a805636b28a8e2f6d3bac7a19d0ffeb49a9acd3",
00077 |           "struct_out": "0a0802c0b27e2e979a2f4083fb54db681cf79870d9c711452e976e086e26d708",
00078 |           "xv": "dd4cfdb0c170a546d1eeff6f6f9ac4438a6f12cd23ec6384750e66574585cc43"
00079 |         },
00080 |         "deeph_sample_id": "md_1",
00081 |         "graph2mat_sample_id": "md_1",
00082 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/metadata.json",
00083 |         "metadata_sha256": "3a257afca0aa9bdb7c49b786a09503d519e2b5f049282b854c05143b327ebdb2",
00084 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.ORB_INDX",
00085 |         "orb_indx_sha256": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00086 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.HSX",
00087 |         "reference_hsx_sha256": "0093b7a77eed41689181fcff11e89695dff7c2ea7e5e0bfff4aed084d9167047",
00088 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.TSDE",
00089 |         "reference_tsde_sha256": "8a3a1f6ecfa9c9332e66aec1f64d437af31984c146f5b169758c190d3a130d30",
00090 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.TSHS",
00091 |         "reference_tshs_sha256": "aba6738da92ac66d80d3334d71dbca7b71aa19b9d6e36bb3716ed3826b21af20",
00092 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/RUN.fdf",
00093 |         "run_fdf_sha256": "ffae5bb676271701eaafc719efa7b009e1c71869e0673542fd0279b6991be59c",
00094 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/RUN.out",
00095 |         "run_output_sha256": "9d5db4fbaec41ce53c006b021a805636b28a8e2f6d3bac7a19d0ffeb49a9acd3",
00096 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1",
00097 |         "sample_id": "md_1",
00098 |         "split": "train",
00099 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.STRUCT_OUT",
00100 |         "struct_out_sha256": "0a0802c0b27e2e979a2f4083fb54db681cf79870d9c711452e976e086e26d708",
00101 |         "system_label": "graphene",
00102 |         "valid": true,
00103 |         "validation_problems": [],
00104 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/train/1/graphene.XV",
00105 |         "xv_sha256": "dd4cfdb0c170a546d1eeff6f6f9ac4438a6f12cd23ec6384750e66574585cc43"
00106 |       }
00107 |     ],
00108 |     "_last_two": [
00109 |       {
00110 |         "artifact_paths": {
00111 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/metadata.json",
00112 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.ORB_INDX",
00113 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.HSX",
00114 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.TSDE",
00115 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.TSHS",
00116 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/RUN.fdf",
00117 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/RUN.out",
00118 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.STRUCT_OUT",
00119 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.XV"
00120 |         },
00121 |         "artifact_sha256": {
00122 |           "metadata": "e3b01b8394838770c812044c610b348764483918d0e3b5c27643f58014773a5c",
00123 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00124 |           "reference_hsx": "aa8bc3ad8c3ad788189027d5c4c5eea44308c72b89f976f3c8188cb098d671b8",
00125 |           "reference_tsde": "7eaddda772e4a765badf3975813b86d4dcad0465cde041ffe546c5dac1c9aa07",
00126 |           "reference_tshs": "876cb7005e1d0c3d62e3fe261679b9e375580e78b02b58864b396b5a1543510e",
00127 |           "run_fdf": "5c4a7542b42437806da7560afc1e06e02f884ae3f5dbc43a281938eb0836a99a",
00128 |           "run_output": "0a884b8f07e47dada3be3325f7acb28d21168f289596e4d4adecf18feb5115b8",
00129 |           "struct_out": "229b72b44f72489a33f159432c7f4c913112368008a67ae844de50bd1e9a49de",
00130 |           "xv": "31229cfe055db2e4264a45b10494c50f3db54a6731b88c20526aad33a0eab334"
00131 |         },
00132 |         "deeph_sample_id": "md_998",
00133 |         "graph2mat_sample_id": "md_998",
00134 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/metadata.json",
00135 |         "metadata_sha256": "e3b01b8394838770c812044c610b348764483918d0e3b5c27643f58014773a5c",
00136 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.ORB_INDX",
00137 |         "orb_indx_sha256": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00138 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.HSX",
00139 |         "reference_hsx_sha256": "aa8bc3ad8c3ad788189027d5c4c5eea44308c72b89f976f3c8188cb098d671b8",
00140 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.TSDE",
00141 |         "reference_tsde_sha256": "7eaddda772e4a765badf3975813b86d4dcad0465cde041ffe546c5dac1c9aa07",
00142 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.TSHS",
00143 |         "reference_tshs_sha256": "876cb7005e1d0c3d62e3fe261679b9e375580e78b02b58864b396b5a1543510e",
00144 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/RUN.fdf",
00145 |         "run_fdf_sha256": "5c4a7542b42437806da7560afc1e06e02f884ae3f5dbc43a281938eb0836a99a",
00146 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/RUN.out",
00147 |         "run_output_sha256": "0a884b8f07e47dada3be3325f7acb28d21168f289596e4d4adecf18feb5115b8",
00148 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998",
00149 |         "sample_id": "md_998",
00150 |         "split": "test",
00151 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.STRUCT_OUT",
00152 |         "struct_out_sha256": "229b72b44f72489a33f159432c7f4c913112368008a67ae844de50bd1e9a49de",
00153 |         "system_label": "graphene",
00154 |         "valid": true,
00155 |         "validation_problems": [],
00156 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/998/graphene.XV",
00157 |         "xv_sha256": "31229cfe055db2e4264a45b10494c50f3db54a6731b88c20526aad33a0eab334"
00158 |       },
00159 |       {
00160 |         "artifact_paths": {
00161 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/metadata.json",
00162 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.ORB_INDX",
00163 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.HSX",
00164 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.TSDE",
00165 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.TSHS",
00166 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/RUN.fdf",
00167 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/RUN.out",
00168 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.STRUCT_OUT",
00169 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.XV"
00170 |         },
00171 |         "artifact_sha256": {
00172 |           "metadata": "d8f8ae74738976c57dbdf2151572c8d868de9828a7f04c7fefa3f2082d09f80b",
00173 |           "orb_indx": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00174 |           "reference_hsx": "d9d64957733926c96ae4b820e067b9892c5fea3b065aa0c318714c075f3f94f3",
00175 |           "reference_tsde": "a1cc7acb74773917e9eabf5c50d89be1bc3afad100e424a10c8fd1b73b24470e",
00176 |           "reference_tshs": "a86e7942fe389cad7ab549ba8d80774f5098a8ff298167c78d31d8cb5debc54a",
00177 |           "run_fdf": "549772617404d776f2103dc81f2f41402696a6a4cf59a5f2744d68d80654b414",
00178 |           "run_output": "ba00e89b317d0aacb33ce04815c4edd31e001cb8d6b4b219b7bc895523daed31",
00179 |           "struct_out": "2ba10ead272a6f580178b6efb3331b9634fe6e6c3d5b886fd1b57f6f300882d2",
00180 |           "xv": "f40efa58359c198c90eb6350f42a178b5c4173e5a3eacf6c53840fb47e332b08"
00181 |         },
00182 |         "deeph_sample_id": "md_999",
00183 |         "graph2mat_sample_id": "md_999",
00184 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/metadata.json",
00185 |         "metadata_sha256": "d8f8ae74738976c57dbdf2151572c8d868de9828a7f04c7fefa3f2082d09f80b",
00186 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.ORB_INDX",
00187 |         "orb_indx_sha256": "dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5",
00188 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.HSX",
00189 |         "reference_hsx_sha256": "d9d64957733926c96ae4b820e067b9892c5fea3b065aa0c318714c075f3f94f3",
00190 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.TSDE",
00191 |         "reference_tsde_sha256": "a1cc7acb74773917e9eabf5c50d89be1bc3afad100e424a10c8fd1b73b24470e",
00192 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.TSHS",
00193 |         "reference_tshs_sha256": "a86e7942fe389cad7ab549ba8d80774f5098a8ff298167c78d31d8cb5debc54a",
00194 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/RUN.fdf",
00195 |         "run_fdf_sha256": "549772617404d776f2103dc81f2f41402696a6a4cf59a5f2744d68d80654b414",
00196 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/RUN.out",
00197 |         "run_output_sha256": "ba00e89b317d0aacb33ce04815c4edd31e001cb8d6b4b219b7bc895523daed31",
00198 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999",
00199 |         "sample_id": "md_999",
00200 |         "split": "test",
00201 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.STRUCT_OUT",
00202 |         "struct_out_sha256": "2ba10ead272a6f580178b6efb3331b9634fe6e6c3d5b886fd1b57f6f300882d2",
00203 |         "system_label": "graphene",
00204 |         "valid": true,
00205 |         "validation_problems": [],
00206 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits/test/999/graphene.XV",
00207 |         "xv_sha256": "f40efa58359c198c90eb6350f42a178b5c4173e5a3eacf6c53840fb47e332b08"
00208 |       }
00209 |     ]
00210 |   },
00211 |   "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
00212 |   "split_counts": {
00213 |     "test": 100,
00214 |     "train": 798,
00215 |     "validation": 100
00216 |   },
00217 |   "split_hash": "acc00551ed4a5ce2be843e7c0280c22fa76e0434f9d5ea6cd83314651b1e8aa6",
00218 |   "split_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/splits",
00219 |   "valid": true,
00220 |   "warnings": []
00221 | }
```

## `Comparison/datasets/graphene_5x5_vacancy/artifact_validation.json` — vista compacta

SHA-256 del JSON completo: `a5686d06080b189cc38366d863e226fe3f096667533240025dae5a97bef59a69`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "basis_present": true,
00003 |   "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00004 |   "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy",
00005 |   "errors": [],
00006 |   "invalid_snapshots": 0,
00007 |   "material_identity_present": true,
00008 |   "pseudopotential_provenance_present": true,
00009 |   "repair_required_snapshots": 0,
00010 |   "siesta_command_line_provenance_present": true,
00011 |   "siesta_environment_provenance_present": true,
00012 |   "siesta_execution_log_present": true,
00013 |   "siesta_input_provenance_present": true,
00014 |   "siesta_version_provenance_present": true,
00015 |   "snapshots": {
00016 |     "_list_length": 20,
00017 |     "_first_two": [
00018 |       {
00019 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00020 |         "errors": [],
00021 |         "missing_required": [],
00022 |         "present_artifacts": {
00023 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.HSX",
00024 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/metadata.json",
00025 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.ORB_INDX",
00026 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.fdf",
00027 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00028 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.STRUCT_OUT",
00029 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSDE",
00030 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSHS",
00031 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.XV"
00032 |         },
00033 |         "repair_required": false,
00034 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0",
00035 |         "system_label": "graphene_5x5_vacancy",
00036 |         "valid": true,
00037 |         "warnings": []
00038 |       },
00039 |       {
00040 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00041 |         "errors": [],
00042 |         "missing_required": [],
00043 |         "present_artifacts": {
00044 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.HSX",
00045 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/metadata.json",
00046 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.ORB_INDX",
00047 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.fdf",
00048 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.out",
00049 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.STRUCT_OUT",
00050 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSDE",
00051 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSHS",
00052 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.XV"
00053 |         },
00054 |         "repair_required": false,
00055 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1",
00056 |         "system_label": "graphene_5x5_vacancy",
00057 |         "valid": true,
00058 |         "warnings": []
00059 |       }
00060 |     ],
00061 |     "_last_two": [
00062 |       {
00063 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00064 |         "errors": [],
00065 |         "missing_required": [],
00066 |         "present_artifacts": {
00067 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.HSX",
00068 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/metadata.json",
00069 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.ORB_INDX",
00070 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.fdf",
00071 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.out",
00072 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.STRUCT_OUT",
00073 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSDE",
00074 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSHS",
00075 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.XV"
00076 |         },
00077 |         "repair_required": false,
00078 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18",
00079 |         "system_label": "graphene_5x5_vacancy",
00080 |         "valid": true,
00081 |         "warnings": []
00082 |       },
00083 |       {
00084 |         "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00085 |         "errors": [],
00086 |         "missing_required": [],
00087 |         "present_artifacts": {
00088 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.HSX",
00089 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/metadata.json",
00090 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.ORB_INDX",
00091 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.fdf",
00092 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.out",
00093 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.STRUCT_OUT",
00094 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSDE",
00095 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSHS",
00096 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.XV"
00097 |         },
00098 |         "repair_required": false,
00099 |         "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19",
00100 |         "system_label": "graphene_5x5_vacancy",
00101 |         "valid": true,
00102 |         "warnings": []
00103 |       }
00104 |     ]
00105 |   },
00106 |   "total_snapshots": 20,
00107 |   "valid": true,
00108 |   "valid_snapshots": 20,
00109 |   "warnings": []
00110 | }
```

## `Comparison/datasets/graphene_5x5_vacancy/benchmark_dataset_manifest.json` — vista compacta

SHA-256 del JSON completo: `029c490e497c65bdad3dc76b4aa9f8fd6d1a05b44023b1756c1627e11c41cb98`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
00003 |   "artifact_validation": {
00004 |     "basis_present": true,
00005 |     "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00006 |     "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy",
00007 |     "errors": [],
00008 |     "invalid_snapshots": 0,
00009 |     "material_identity_present": true,
00010 |     "pseudopotential_provenance_present": true,
00011 |     "repair_required_snapshots": 0,
00012 |     "siesta_command_line_provenance_present": true,
00013 |     "siesta_environment_provenance_present": true,
00014 |     "siesta_execution_log_present": true,
00015 |     "siesta_input_provenance_present": true,
00016 |     "siesta_version_provenance_present": true,
00017 |     "snapshots": {
00018 |       "_list_length": 20,
00019 |       "_first_two": [
00020 |         {
00021 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00022 |           "errors": [],
00023 |           "missing_required": [],
00024 |           "present_artifacts": {
00025 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.HSX",
00026 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/metadata.json",
00027 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.ORB_INDX",
00028 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.fdf",
00029 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00030 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.STRUCT_OUT",
00031 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSDE",
00032 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSHS",
00033 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.XV"
00034 |           },
00035 |           "repair_required": false,
00036 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0",
00037 |           "system_label": "graphene_5x5_vacancy",
00038 |           "valid": true,
00039 |           "warnings": []
00040 |         },
00041 |         {
00042 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00043 |           "errors": [],
00044 |           "missing_required": [],
00045 |           "present_artifacts": {
00046 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.HSX",
00047 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/metadata.json",
00048 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.ORB_INDX",
00049 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.fdf",
00050 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.out",
00051 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.STRUCT_OUT",
00052 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSDE",
00053 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSHS",
00054 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.XV"
00055 |           },
00056 |           "repair_required": false,
00057 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1",
00058 |           "system_label": "graphene_5x5_vacancy",
00059 |           "valid": true,
00060 |           "warnings": []
00061 |         }
00062 |       ],
00063 |       "_last_two": [
00064 |         {
00065 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00066 |           "errors": [],
00067 |           "missing_required": [],
00068 |           "present_artifacts": {
00069 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.HSX",
00070 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/metadata.json",
00071 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.ORB_INDX",
00072 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.fdf",
00073 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.out",
00074 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.STRUCT_OUT",
00075 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSDE",
00076 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSHS",
00077 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.XV"
00078 |           },
00079 |           "repair_required": false,
00080 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18",
00081 |           "system_label": "graphene_5x5_vacancy",
00082 |           "valid": true,
00083 |           "warnings": []
00084 |         },
00085 |         {
00086 |           "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
00087 |           "errors": [],
00088 |           "missing_required": [],
00089 |           "present_artifacts": {
00090 |             "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.HSX",
00091 |             "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/metadata.json",
00092 |             "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.ORB_INDX",
00093 |             "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.fdf",
00094 |             "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.out",
00095 |             "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.STRUCT_OUT",
00096 |             "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSDE",
00097 |             "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSHS",
00098 |             "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.XV"
00099 |           },
00100 |           "repair_required": false,
00101 |           "snapshot_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19",
00102 |           "system_label": "graphene_5x5_vacancy",
00103 |           "valid": true,
00104 |           "warnings": []
00105 |         }
00106 |       ]
00107 |     },
00108 |     "total_snapshots": 20,
00109 |     "valid": true,
00110 |     "valid_snapshots": 20,
00111 |     "warnings": []
00112 |   },
00113 |   "basis_hashes": {
00114 |     "C.ion.xml": "6740d9f56df9f2d42ff27e6a7abd9b7b0224a49cbda58b52b7f38136bcfc8b6f",
00115 |     "Ghost-H.ion.xml": "bd23b909b0cb2de1f9c9cfa421ca2448e6e2934f4e0102a55ceb0357bcf45ca5"
00116 |   },
00117 |   "benchmark_dataset_id": "joint_graph2mat_deeph_77416e667d79d121",
00118 |   "benchmark_ready": true,
00119 |   "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy",
00120 |   "deeph_pack_commit": "",
00121 |   "environment": {
00122 |     "executable": "/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python",
00123 |     "package_versions": {
00124 |       "graph2mat": "0.0.13",
00125 |       "numpy": "2.4.4",
00126 |       "sisl": "0.16.4",
00127 |       "torch": "2.11.0"
00128 |     },
00129 |     "platform": "Linux-6.17.0-35-generic-x86_64-with-glibc2.39",
00130 |     "python_version": "3.12.3"
00131 |   },
00132 |   "frozen_split_manifest": {
00133 |     "path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/frozen_split_manifest.json",
00134 |     "split_counts": {
00135 |       "test": 20,
00136 |       "train": 0,
00137 |       "validation": 0
00138 |     },
00139 |     "split_hash": "f9900fe3862c46741671de61311e8a138b8bba02cd32eb298faf6e9fd4a13563",
00140 |     "valid": true
00141 |   },
00142 |   "generation_mode": "derived_pristine_monovacancy_static_siesta",
00143 |   "graph2mat_commit": "",
00144 |   "kpoint_summary": {
00145 |     "kgrid_monkhorst_pack": [],
00146 |     "present": false
00147 |   },
00148 |   "material_label": "graphene_5x5_vacancy",
00149 |   "material_source": {
00150 |     "absolute_paths_used": false,
00151 |     "basis_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/basis",
00152 |     "basis_file_sha256": {
00153 |       "C.ion.xml": "6740d9f56df9f2d42ff27e6a7abd9b7b0224a49cbda58b52b7f38136bcfc8b6f",
00154 |       "Ghost-H.ion.xml": "bd23b909b0cb2de1f9c9cfa421ca2448e6e2934f4e0102a55ceb0357bcf45ca5"
00155 |     },
00156 |     "defect": {
00157 |       "relaxed": false,
00158 |       "source_num_atoms": 50,
00159 |       "spin_polarized": false,
00160 |       "target_num_atoms": 49,
00161 |       "type": "monovacancy"
00162 |     },
00163 |     "environment": {
00164 |       "executable": "/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python",
00165 |       "package_versions": {
00166 |         "graph2mat": "0.0.13",
00167 |         "numpy": "2.4.4",
00168 |         "sisl": "0.16.4",
00169 |         "torch": "2.11.0"
00170 |       },
00171 |       "platform": "Linux-6.17.0-35-generic-x86_64-with-glibc2.39",
00172 |       "python_version": "3.12.3"
00173 |     },
00174 |     "fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene_5x5_vacancy/RUN.fdf",
00175 |     "fdf_sha256": "3354707c38b410cb4c75e7472535ee2494a585184fddcec4768059c49046b992",
00176 |     "graph2mat_basis_files": {
00177 |       "C": {
00178 |         "action": "copied",
00179 |         "file_name": "C.ion.xml",
00180 |         "sha256": "6740d9f56df9f2d42ff27e6a7abd9b7b0224a49cbda58b52b7f38136bcfc8b6f",
00181 |         "source": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/basis/C.ion.xml",
00182 |         "target": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600/material_basis/C.ion.xml"
00183 |       }
00184 |     },
00185 |     "label": "graphene_5x5_vacancy",
00186 |     "material_source": "derived_monovacancy_from_pristine_test_snapshots",
00187 |     "preset": "graphene_5x5_vacancy",
00188 |     "pseudopotential_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/pseudos",
00189 |     "pseudopotential_sha256": {
00190 |       "C": "835287a894d851d30c2c613edcc4124c493a08dad74a2aa1a982266c15d1a0e6"
00191 |     },
00192 |     "pseudopotentials": {
00193 |       "C": "/home/christian/repositorios/MD_vs_AtomicDisplacement/materials/graphene/pseudos/C.psf"
00194 |     },
00195 |     "pseudopotentials_copied_to_dataset": {
00196 |       "C": "C.psf"
00197 |     },
00198 |     "pseudopotentials_verified_in_dataset": {},
00199 |     "run_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00200 |     "siesta_build_info": "Authorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nExecutable      : siesta\nVersion         : 5.4.2-11-g4e9a46060\nArchitecture    : x86_64\nCompiler version: GNU-13.3.0\nCompiler flags  : -fallow-argument-mismatch -O3 -march=native\nParallelisations: MPI\nNetCDF support\nNetCDF-4 support\nLua support\nELSI support. Solvers:\n   ELPA (internal) \n   NTPoly\n   OMM\nDFT-D3 support",
00201 |     "siesta_command_line": "/home/christian/bin/siesta",
00202 |     "siesta_executable": "siesta",
00203 |     "siesta_stdout_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00204 |     "siesta_version": "5.4.2-11-g4e9a46060",
00205 |     "siesta_version_probe": {
00206 |       "attempts": [
00207 |         {
00208 |           "command": [
00209 |             "siesta",
00210 |             "--version"
00211 |           ],
00212 |           "output": "Authorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nExecutable      : siesta\nVersion         : 5.4.2-11-g4e9a46060\nArchitecture    : x86_64\nCompiler version: GNU-13.3.0\nCompiler flags  : -fallow-argument-mismatch -O3 -march=native\nParallelisations: MPI\nNetCDF support\nNetCDF-4 support\nLua support\nELSI support. Solvers:\n   ELPA (internal) \n   NTPoly\n   OMM\nDFT-D3 support",
00213 |           "returncode": 0
00214 |         }
00215 |       ],
00216 |       "status": "detected"
00217 |     },
00218 |     "source_dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600",
00219 |     "species": [
00220 |       {
00221 |         "atomic_number": 6,
00222 |         "index": 1,
00223 |         "label": "C"
00224 |       }
00225 |     ],
00226 |     "structure_type": "crystal",
00227 |     "warning": null
00228 |   },
00229 |   "provenance_status": {
00230 |     "basis_provenance": true,
00231 |     "material_identity": true,
00232 |     "missing": [],
00233 |     "pseudopotential_provenance": true,
00234 |     "siesta_command_line_provenance": true,
00235 |     "siesta_environment_provenance": true,
00236 |     "siesta_execution_log_provenance": true,
00237 |     "siesta_input_provenance": true,
00238 |     "siesta_version_provenance": true,
00239 |     "strict_paper_ready": true,
00240 |     "valid": true
00241 |   },
00242 |   "pseudopotential_hashes": {
00243 |     "C": "835287a894d851d30c2c613edcc4124c493a08dad74a2aa1a982266c15d1a0e6"
00244 |   },
00245 |   "samples": {
00246 |     "_list_length": 20,
00247 |     "_first_two": [
00248 |       {
00249 |         "artifact_paths": {
00250 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.HSX",
00251 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/metadata.json",
00252 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.ORB_INDX",
00253 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.fdf",
00254 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00255 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.STRUCT_OUT",
00256 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSDE",
00257 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSHS",
00258 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.XV"
00259 |         },
00260 |         "artifact_sha256": {
00261 |           "hsx": "4a507a994a12de5c2c23f67bb2d091367c5a75175edc286096478e52df960030",
00262 |           "metadata": "aaad8cde7f0b086ae06ca22911a57c78e0824270f6ae166d1b458ea855be2744",
00263 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00264 |           "run_fdf": "75309217eff0fc3810e8cb0d5258253c088fe9b58af197b434f37b74e7b35aba",
00265 |           "run_output": "47f759c6bc158e3b2e3cd548af1281825d7cf154b60d9848c4aca8bce27bf69d",
00266 |           "struct_out": "b6b8fee0b437000e80fa355fe5d889ef4060ba8f7692e2eac5c11d54eaf5ec72",
00267 |           "tsde": "575075aaaccab5cbab872f1b2eaf69ba44f91bd635ffd81c8cef0752c0124402",
00268 |           "tshs": "b07a43a2b81d5a2a5fd7b8efc04c4bf235d5edfaec0eb774d55f64b7ff27ef65",
00269 |           "xv": "a92a9b6af2b18e9c0ca31eeeec97906fb6c48f109e432f66defb9c4408fc93b4"
00270 |         },
00271 |         "errors": [],
00272 |         "missing_required": [],
00273 |         "repair_required": false,
00274 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0",
00275 |         "system_label": "graphene_5x5_vacancy",
00276 |         "valid": true,
00277 |         "warnings": []
00278 |       },
00279 |       {
00280 |         "artifact_paths": {
00281 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.HSX",
00282 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/metadata.json",
00283 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.ORB_INDX",
00284 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.fdf",
00285 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.out",
00286 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.STRUCT_OUT",
00287 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSDE",
00288 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSHS",
00289 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.XV"
00290 |         },
00291 |         "artifact_sha256": {
00292 |           "hsx": "3f3407be59a0ab82aab525654668c67e131026b39600539ea4ca4b7d312b92c3",
00293 |           "metadata": "e6d2b1e1bc7d67ee164a4a9fe508b068a7480cfc70f4eaf9edb5bb96c7e789b3",
00294 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00295 |           "run_fdf": "a41344b34c946c8f3c3b6ffcb07bb316600e21951bb3890dca905bdff8e64382",
00296 |           "run_output": "7db9cbce2d95cdc9d87db6d5ece3a47c5bc716235abfbd9fd1339a2b57b7de53",
00297 |           "struct_out": "35e3c4fbf0c3c8af5e03216238bea0e3bef8a3390705db0df6823a8b46d0d200",
00298 |           "tsde": "b88d8df4904935dc4cff84a79b2299da1627914911899a07558f8962fdd7eb82",
00299 |           "tshs": "80c295d880ff35f251748e854adcf0fdaa1abb5d6147577c2ec41bc5e7abd5b4",
00300 |           "xv": "9eba5ca8ce8c5a5578ba20668d20b9b33f71a9d13d2eb2e786be64285eb0c361"
00301 |         },
00302 |         "errors": [],
00303 |         "missing_required": [],
00304 |         "repair_required": false,
00305 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1",
00306 |         "system_label": "graphene_5x5_vacancy",
00307 |         "valid": true,
00308 |         "warnings": []
00309 |       }
00310 |     ],
00311 |     "_last_two": [
00312 |       {
00313 |         "artifact_paths": {
00314 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/graphene_5x5_vacancy.HSX",
00315 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/metadata.json",
00316 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/graphene_5x5_vacancy.ORB_INDX",
00317 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/RUN.fdf",
00318 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/RUN.out",
00319 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/graphene_5x5_vacancy.STRUCT_OUT",
00320 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/graphene_5x5_vacancy.TSDE",
00321 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/graphene_5x5_vacancy.TSHS",
00322 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8/graphene_5x5_vacancy.XV"
00323 |         },
00324 |         "artifact_sha256": {
00325 |           "hsx": "8abc78258e4da5e99d53dc48c43a649ec50667166e2dad210ad3ffd871f85d20",
00326 |           "metadata": "24c332ff27ea34bf4070532f5b9ff1c7dd0bab1c3ff17e55005c994d55574ff1",
00327 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00328 |           "run_fdf": "5d2906c2e404288614e527ae2bc53d17990b335b043771114d53deea7c260cfb",
00329 |           "run_output": "ebba4656c2d38fd951cd94a4475fd48296ac884f2b05f4715a32ba5e5b879ed3",
00330 |           "struct_out": "daeaecdf00acee3268ffcf58ccba03a7f9a08c3a1c498c1a9148e1a88ceea81d",
00331 |           "tsde": "41d83e2b2dc464d2e19a7bc7faf80a2c0bc89f1e0e2a7f00f004f0babe753ce9",
00332 |           "tshs": "6bd8dacdfcb7d190616ef769622b308ade4b9d8cc415b427850651fc438959e7",
00333 |           "xv": "b3cdcad24769ada4c941ca1deed1ef42d8699c6728d7fdab1c32c681338f7488"
00334 |         },
00335 |         "errors": [],
00336 |         "missing_required": [],
00337 |         "repair_required": false,
00338 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/8",
00339 |         "system_label": "graphene_5x5_vacancy",
00340 |         "valid": true,
00341 |         "warnings": []
00342 |       },
00343 |       {
00344 |         "artifact_paths": {
00345 |           "hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/graphene_5x5_vacancy.HSX",
00346 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/metadata.json",
00347 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/graphene_5x5_vacancy.ORB_INDX",
00348 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/RUN.fdf",
00349 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/RUN.out",
00350 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/graphene_5x5_vacancy.STRUCT_OUT",
00351 |           "tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/graphene_5x5_vacancy.TSDE",
00352 |           "tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/graphene_5x5_vacancy.TSHS",
00353 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9/graphene_5x5_vacancy.XV"
00354 |         },
00355 |         "artifact_sha256": {
00356 |           "hsx": "f92e7ad98c173cfe551f64a45008326060dc22ad79320967b76cf9c98b14876d",
00357 |           "metadata": "106ec25459b1fcc910a4dbe6396b16cd884c8dc713e5ac38282647426d2dbdfd",
00358 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00359 |           "run_fdf": "db08f8a710eea49d9158d708ed44b17d817b39ec2fce2eae3812688b265f0af8",
00360 |           "run_output": "47378290a09ab848da397c25ec18284bcb96b901e711abec670c5644e0c9fe8c",
00361 |           "struct_out": "1755f1b05e15be01f949e0ec4a15f500605f8ad0731f8f756615a1ca4d7cb62f",
00362 |           "tsde": "0caa84a819f8c1e57afcfc23081cfae523a52ecb876056d41caf47346b1004d9",
00363 |           "tshs": "07c278aac6fc346e9d13363e27b644faa10718151b0ab40810019bdd2bbdf114",
00364 |           "xv": "642afe89e0cde92b892c33218e9c4f971e6bbec95e459813d2a056d7123101e0"
00365 |         },
00366 |         "errors": [],
00367 |         "missing_required": [],
00368 |         "repair_required": false,
00369 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/9",
00370 |         "system_label": "graphene_5x5_vacancy",
00371 |         "valid": true,
00372 |         "warnings": []
00373 |       }
00374 |     ]
00375 |   },
00376 |   "schema": "joint_graph2mat_deeph_benchmark_manifest_v1",
00377 |   "siesta_build_info": "Authorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nAuthorization required, but no authorization protocol specified\n\nExecutable      : siesta\nVersion         : 5.4.2-11-g4e9a46060\nArchitecture    : x86_64\nCompiler version: GNU-13.3.0\nCompiler flags  : -fallow-argument-mismatch -O3 -march=native\nParallelisations: MPI\nNetCDF support\nNetCDF-4 support\nLua support\nELSI support. Solvers:\n   ELPA (internal) \n   NTPoly\n   OMM\nDFT-D3 support",
00378 |   "siesta_command_line": "/home/christian/bin/siesta",
00379 |   "siesta_executable": "siesta",
00380 |   "siesta_flags": {},
00381 |   "siesta_input_path": "",
00382 |   "siesta_input_sha256": "",
00383 |   "siesta_returncode": null,
00384 |   "siesta_stdout_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00385 |   "siesta_version": "5.4.2-11-g4e9a46060",
00386 |   "siesta_version_source_file": "",
00387 |   "spin_summary": {},
00388 |   "system_label": "graphene_5x5_vacancy",
00389 |   "validation_status": "valid",
00390 |   "warnings": []
00391 | }
```

## `Comparison/datasets/graphene_5x5_vacancy/frozen_split_manifest.json` — vista compacta

SHA-256 del JSON completo: `3b48b66f04ecb08855567d8c4d7bb6218a49dcd76fd43bc9595e3e4bd9b12f63`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
00003 |   "dataset_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy",
00004 |   "rows": {
00005 |     "_list_length": 20,
00006 |     "_first_two": [
00007 |       {
00008 |         "artifact_paths": {
00009 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/metadata.json",
00010 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.ORB_INDX",
00011 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.HSX",
00012 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSDE",
00013 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSHS",
00014 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.fdf",
00015 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00016 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.STRUCT_OUT",
00017 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.XV"
00018 |         },
00019 |         "artifact_sha256": {
00020 |           "metadata": "aaad8cde7f0b086ae06ca22911a57c78e0824270f6ae166d1b458ea855be2744",
00021 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00022 |           "reference_hsx": "4a507a994a12de5c2c23f67bb2d091367c5a75175edc286096478e52df960030",
00023 |           "reference_tsde": "575075aaaccab5cbab872f1b2eaf69ba44f91bd635ffd81c8cef0752c0124402",
00024 |           "reference_tshs": "b07a43a2b81d5a2a5fd7b8efc04c4bf235d5edfaec0eb774d55f64b7ff27ef65",
00025 |           "run_fdf": "75309217eff0fc3810e8cb0d5258253c088fe9b58af197b434f37b74e7b35aba",
00026 |           "run_output": "47f759c6bc158e3b2e3cd548af1281825d7cf154b60d9848c4aca8bce27bf69d",
00027 |           "struct_out": "b6b8fee0b437000e80fa355fe5d889ef4060ba8f7692e2eac5c11d54eaf5ec72",
00028 |           "xv": "a92a9b6af2b18e9c0ca31eeeec97906fb6c48f109e432f66defb9c4408fc93b4"
00029 |         },
00030 |         "deeph_sample_id": "vacancy_md_540",
00031 |         "graph2mat_sample_id": "vacancy_md_540",
00032 |         "hamiltonian_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSHS",
00033 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/metadata.json",
00034 |         "metadata_sha256": "aaad8cde7f0b086ae06ca22911a57c78e0824270f6ae166d1b458ea855be2744",
00035 |         "method": "md_vacancy",
00036 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.ORB_INDX",
00037 |         "orb_indx_sha256": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00038 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.HSX",
00039 |         "reference_hsx_sha256": "4a507a994a12de5c2c23f67bb2d091367c5a75175edc286096478e52df960030",
00040 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSDE",
00041 |         "reference_tsde_sha256": "575075aaaccab5cbab872f1b2eaf69ba44f91bd635ffd81c8cef0752c0124402",
00042 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.TSHS",
00043 |         "reference_tshs_sha256": "b07a43a2b81d5a2a5fd7b8efc04c4bf235d5edfaec0eb774d55f64b7ff27ef65",
00044 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.fdf",
00045 |         "run_fdf_sha256": "75309217eff0fc3810e8cb0d5258253c088fe9b58af197b434f37b74e7b35aba",
00046 |         "run_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00047 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.out",
00048 |         "run_output_sha256": "47f759c6bc158e3b2e3cd548af1281825d7cf154b60d9848c4aca8bce27bf69d",
00049 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0",
00050 |         "sample_id": "vacancy_md_540",
00051 |         "source_run": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600",
00052 |         "source_sample_id": "md_540",
00053 |         "split": "test",
00054 |         "status": "completed",
00055 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.STRUCT_OUT",
00056 |         "struct_out_sha256": "b6b8fee0b437000e80fa355fe5d889ef4060ba8f7692e2eac5c11d54eaf5ec72",
00057 |         "structure_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/RUN.fdf",
00058 |         "system_label": "graphene_5x5_vacancy",
00059 |         "valid": true,
00060 |         "validation_problems": [],
00061 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/0/graphene_5x5_vacancy.XV",
00062 |         "xv_sha256": "a92a9b6af2b18e9c0ca31eeeec97906fb6c48f109e432f66defb9c4408fc93b4"
00063 |       },
00064 |       {
00065 |         "artifact_paths": {
00066 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/metadata.json",
00067 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.ORB_INDX",
00068 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.HSX",
00069 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSDE",
00070 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSHS",
00071 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.fdf",
00072 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.out",
00073 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.STRUCT_OUT",
00074 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.XV"
00075 |         },
00076 |         "artifact_sha256": {
00077 |           "metadata": "e6d2b1e1bc7d67ee164a4a9fe508b068a7480cfc70f4eaf9edb5bb96c7e789b3",
00078 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00079 |           "reference_hsx": "3f3407be59a0ab82aab525654668c67e131026b39600539ea4ca4b7d312b92c3",
00080 |           "reference_tsde": "b88d8df4904935dc4cff84a79b2299da1627914911899a07558f8962fdd7eb82",
00081 |           "reference_tshs": "80c295d880ff35f251748e854adcf0fdaa1abb5d6147577c2ec41bc5e7abd5b4",
00082 |           "run_fdf": "a41344b34c946c8f3c3b6ffcb07bb316600e21951bb3890dca905bdff8e64382",
00083 |           "run_output": "7db9cbce2d95cdc9d87db6d5ece3a47c5bc716235abfbd9fd1339a2b57b7de53",
00084 |           "struct_out": "35e3c4fbf0c3c8af5e03216238bea0e3bef8a3390705db0df6823a8b46d0d200",
00085 |           "xv": "9eba5ca8ce8c5a5578ba20668d20b9b33f71a9d13d2eb2e786be64285eb0c361"
00086 |         },
00087 |         "deeph_sample_id": "vacancy_md_541",
00088 |         "graph2mat_sample_id": "vacancy_md_541",
00089 |         "hamiltonian_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSHS",
00090 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/metadata.json",
00091 |         "metadata_sha256": "e6d2b1e1bc7d67ee164a4a9fe508b068a7480cfc70f4eaf9edb5bb96c7e789b3",
00092 |         "method": "md_vacancy",
00093 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.ORB_INDX",
00094 |         "orb_indx_sha256": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00095 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.HSX",
00096 |         "reference_hsx_sha256": "3f3407be59a0ab82aab525654668c67e131026b39600539ea4ca4b7d312b92c3",
00097 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSDE",
00098 |         "reference_tsde_sha256": "b88d8df4904935dc4cff84a79b2299da1627914911899a07558f8962fdd7eb82",
00099 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.TSHS",
00100 |         "reference_tshs_sha256": "80c295d880ff35f251748e854adcf0fdaa1abb5d6147577c2ec41bc5e7abd5b4",
00101 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.fdf",
00102 |         "run_fdf_sha256": "a41344b34c946c8f3c3b6ffcb07bb316600e21951bb3890dca905bdff8e64382",
00103 |         "run_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.out",
00104 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.out",
00105 |         "run_output_sha256": "7db9cbce2d95cdc9d87db6d5ece3a47c5bc716235abfbd9fd1339a2b57b7de53",
00106 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1",
00107 |         "sample_id": "vacancy_md_541",
00108 |         "source_run": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600",
00109 |         "source_sample_id": "md_541",
00110 |         "split": "test",
00111 |         "status": "completed",
00112 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.STRUCT_OUT",
00113 |         "struct_out_sha256": "35e3c4fbf0c3c8af5e03216238bea0e3bef8a3390705db0df6823a8b46d0d200",
00114 |         "structure_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/RUN.fdf",
00115 |         "system_label": "graphene_5x5_vacancy",
00116 |         "valid": true,
00117 |         "validation_problems": [],
00118 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/1/graphene_5x5_vacancy.XV",
00119 |         "xv_sha256": "9eba5ca8ce8c5a5578ba20668d20b9b33f71a9d13d2eb2e786be64285eb0c361"
00120 |       }
00121 |     ],
00122 |     "_last_two": [
00123 |       {
00124 |         "artifact_paths": {
00125 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/metadata.json",
00126 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.ORB_INDX",
00127 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.HSX",
00128 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSDE",
00129 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSHS",
00130 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.fdf",
00131 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.out",
00132 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.STRUCT_OUT",
00133 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.XV"
00134 |         },
00135 |         "artifact_sha256": {
00136 |           "metadata": "bbf1345608367acf5e2d085070c7dcbea97c7654c2cc78bd3e3340ac448fd829",
00137 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00138 |           "reference_hsx": "7bd6346d62167901c24e31b050856b8544351989ce2601f8c109268bf3263c4f",
00139 |           "reference_tsde": "fe124b0b41e0468e5b3020831581621d41515b78a5969a9c16daac29b81060d3",
00140 |           "reference_tshs": "cf32ff726ecb9133a945dfec7daa10e780c6e539a5bbe9f4d7b31f2de8980b39",
00141 |           "run_fdf": "5c4aeda100072d462f5fdaa6f62e71874f326435de57b5d775885b95408a99d1",
00142 |           "run_output": "4664abde5e4e1c877eab09f7a629cbc3fb70670a536cc31ca1194eb96fb5cc32",
00143 |           "struct_out": "c7b614d93eac70db15ee49e1146fc3d81a59dea4c43fc34a5c9f8ad041d41d86",
00144 |           "xv": "9d11054b0d5c9d6d3c40bf5e2c286d8ce2cb15254f92aeabf5cc8362322d6b4e"
00145 |         },
00146 |         "deeph_sample_id": "vacancy_md_558",
00147 |         "graph2mat_sample_id": "vacancy_md_558",
00148 |         "hamiltonian_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSHS",
00149 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/metadata.json",
00150 |         "metadata_sha256": "bbf1345608367acf5e2d085070c7dcbea97c7654c2cc78bd3e3340ac448fd829",
00151 |         "method": "md_vacancy",
00152 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.ORB_INDX",
00153 |         "orb_indx_sha256": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00154 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.HSX",
00155 |         "reference_hsx_sha256": "7bd6346d62167901c24e31b050856b8544351989ce2601f8c109268bf3263c4f",
00156 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSDE",
00157 |         "reference_tsde_sha256": "fe124b0b41e0468e5b3020831581621d41515b78a5969a9c16daac29b81060d3",
00158 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.TSHS",
00159 |         "reference_tshs_sha256": "cf32ff726ecb9133a945dfec7daa10e780c6e539a5bbe9f4d7b31f2de8980b39",
00160 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.fdf",
00161 |         "run_fdf_sha256": "5c4aeda100072d462f5fdaa6f62e71874f326435de57b5d775885b95408a99d1",
00162 |         "run_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.out",
00163 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.out",
00164 |         "run_output_sha256": "4664abde5e4e1c877eab09f7a629cbc3fb70670a536cc31ca1194eb96fb5cc32",
00165 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18",
00166 |         "sample_id": "vacancy_md_558",
00167 |         "source_run": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600",
00168 |         "source_sample_id": "md_558",
00169 |         "split": "test",
00170 |         "status": "completed",
00171 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.STRUCT_OUT",
00172 |         "struct_out_sha256": "c7b614d93eac70db15ee49e1146fc3d81a59dea4c43fc34a5c9f8ad041d41d86",
00173 |         "structure_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/RUN.fdf",
00174 |         "system_label": "graphene_5x5_vacancy",
00175 |         "valid": true,
00176 |         "validation_problems": [],
00177 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/18/graphene_5x5_vacancy.XV",
00178 |         "xv_sha256": "9d11054b0d5c9d6d3c40bf5e2c286d8ce2cb15254f92aeabf5cc8362322d6b4e"
00179 |       },
00180 |       {
00181 |         "artifact_paths": {
00182 |           "metadata": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/metadata.json",
00183 |           "orb_indx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.ORB_INDX",
00184 |           "reference_hsx": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.HSX",
00185 |           "reference_tsde": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSDE",
00186 |           "reference_tshs": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSHS",
00187 |           "run_fdf": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.fdf",
00188 |           "run_output": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.out",
00189 |           "struct_out": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.STRUCT_OUT",
00190 |           "xv": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.XV"
00191 |         },
00192 |         "artifact_sha256": {
00193 |           "metadata": "9e8e8d145310647ac8c16f85ab14e7d99028250fe8faa230613d5ed288b8af98",
00194 |           "orb_indx": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00195 |           "reference_hsx": "fd390c537aca8bfe46cdd7cd528c0f40847810eb9aa1498b55c35e5a5818dd1e",
00196 |           "reference_tsde": "e4b5a940a443e8ffad594bc30423c331283bd0803acef70ebecff36239fdc2d1",
00197 |           "reference_tshs": "f3514ca7d1b40cf056562043269acace5d261f4ee1b65692882daa1ca347200d",
00198 |           "run_fdf": "19da1c1c7c628b8576873ed081973779b51ccc39142b6788187286924b7c997b",
00199 |           "run_output": "1f20bb624275396a4bc397265444ae2c42a0633747787894d2903e93d4c840ae",
00200 |           "struct_out": "eb9f61dcc9719e2bcb7d6f39d5532c07cc583e4e3b80513ee4f30b2924e27098",
00201 |           "xv": "3cab774fe6d3b61e2534ad13002ab26ed595e7affa67e9f02214048630017640"
00202 |         },
00203 |         "deeph_sample_id": "vacancy_md_559",
00204 |         "graph2mat_sample_id": "vacancy_md_559",
00205 |         "hamiltonian_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSHS",
00206 |         "metadata_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/metadata.json",
00207 |         "metadata_sha256": "9e8e8d145310647ac8c16f85ab14e7d99028250fe8faa230613d5ed288b8af98",
00208 |         "method": "md_vacancy",
00209 |         "orb_indx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.ORB_INDX",
00210 |         "orb_indx_sha256": "e3654ce5a3d0efb03bec39f7f49bfb3f2053ff2738053d7d90d741a3a0610e2f",
00211 |         "reference_hsx_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.HSX",
00212 |         "reference_hsx_sha256": "fd390c537aca8bfe46cdd7cd528c0f40847810eb9aa1498b55c35e5a5818dd1e",
00213 |         "reference_tsde_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSDE",
00214 |         "reference_tsde_sha256": "e4b5a940a443e8ffad594bc30423c331283bd0803acef70ebecff36239fdc2d1",
00215 |         "reference_tshs_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.TSHS",
00216 |         "reference_tshs_sha256": "f3514ca7d1b40cf056562043269acace5d261f4ee1b65692882daa1ca347200d",
00217 |         "run_fdf_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.fdf",
00218 |         "run_fdf_sha256": "19da1c1c7c628b8576873ed081973779b51ccc39142b6788187286924b7c997b",
00219 |         "run_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.out",
00220 |         "run_output_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.out",
00221 |         "run_output_sha256": "1f20bb624275396a4bc397265444ae2c42a0633747787894d2903e93d4c840ae",
00222 |         "sample_dir": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19",
00223 |         "sample_id": "vacancy_md_559",
00224 |         "source_run": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/graphene_5x5_scale_iid600",
00225 |         "source_sample_id": "md_559",
00226 |         "split": "test",
00227 |         "status": "completed",
00228 |         "struct_out_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.STRUCT_OUT",
00229 |         "struct_out_sha256": "eb9f61dcc9719e2bcb7d6f39d5532c07cc583e4e3b80513ee4f30b2924e27098",
00230 |         "structure_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/RUN.fdf",
00231 |         "system_label": "graphene_5x5_vacancy",
00232 |         "valid": true,
00233 |         "validation_problems": [],
00234 |         "xv_path": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits/test/19/graphene_5x5_vacancy.XV",
00235 |         "xv_sha256": "3cab774fe6d3b61e2534ad13002ab26ed595e7affa67e9f02214048630017640"
00236 |       }
00237 |     ]
00238 |   },
00239 |   "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
00240 |   "split_counts": {
00241 |     "test": 20,
00242 |     "train": 0,
00243 |     "validation": 0
00244 |   },
00245 |   "split_hash": "f9900fe3862c46741671de61311e8a138b8bba02cd32eb298faf6e9fd4a13563",
00246 |   "split_root": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_5x5_vacancy/splits",
00247 |   "valid": true,
00248 |   "warnings": []
00249 | }
```
