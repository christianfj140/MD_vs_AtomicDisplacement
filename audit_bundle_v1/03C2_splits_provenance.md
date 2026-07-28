# Dossier 2B — Splits, referencias y procedencia

## Objeto de revisión

Auditar aislamiento train/validation/test, referencias prohibidas, identidad material, contratos de artefactos y estado reproducible.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `Comparison/scripts/deeph_fair_utils.py`

SHA-256: `348246fafa394ca1344a6b77634bef784f2de3bc99d4c6d7e4994fb38c80bab9`

```py
00001 | #!/usr/bin/env python3
00002 | """Utilities for fair Graph2Mat-vs-DeepH SIESTA benchmark harnesses."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import csv
00007 | import hashlib
00008 | import json
00009 | import math
00010 | import os
00011 | import re
00012 | import shutil
00013 | import subprocess
00014 | from configparser import ConfigParser
00015 | from dataclasses import dataclass
00016 | from pathlib import Path
00017 | from typing import Any, Iterable
00018 | 
00019 | 
00020 | REPO_ROOT = Path(__file__).resolve().parents[2]
00021 | COMPARISON_ROOT = REPO_ROOT / "Comparison"
00022 | DEFAULT_DEEPH_REPO = Path(os.environ["DEEPH_PACK_ROOT"]).expanduser() if os.environ.get("DEEPH_PACK_ROOT") else Path("deeph-pack")
00023 | FORBIDDEN_REFERENCE_NAMES = {"ML_prediction.HSX"}
00024 | DEEPH_REQUIRED_SIESTA_SUFFIXES = (".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX")
00025 | 
00026 | 
00027 | class DeepHFairBenchmarkError(RuntimeError):
00028 |     """Raised when the fair-comparison contract cannot be satisfied."""
00029 | 
00030 | 
00031 | @dataclass(frozen=True)
00032 | class SplitSample:
00033 |     sample: str
00034 |     split: str
00035 |     sample_id: str
00036 |     sample_dir: Path
00037 |     structure_path: Path | None
00038 |     hamiltonian_path: Path | None
00039 |     metadata_path: Path | None
00040 |     source_row: dict[str, Any]
00041 | 
00042 | 
00043 | def sha256_file(path: Path | None) -> str | None:
00044 |     if path is None or not path.exists() or not path.is_file():
00045 |         return None
00046 |     digest = hashlib.sha256()
00047 |     with path.open("rb") as handle:
00048 |         for chunk in iter(lambda: handle.read(1024 * 1024), b""):
00049 |             digest.update(chunk)
00050 |     return digest.hexdigest()
00051 | 
00052 | 
00053 | def sha256_text(text: str) -> str:
00054 |     return hashlib.sha256(text.encode("utf-8")).hexdigest()
00055 | 
00056 | 
00057 | def read_json(path: Path) -> dict[str, Any]:
00058 |     if not path.exists():
00059 |         return {}
00060 |     payload = json.loads(path.read_text(encoding="utf-8"))
00061 |     return payload if isinstance(payload, dict) else {}
00062 | 
00063 | 
00064 | def write_json(path: Path, payload: dict[str, Any]) -> None:
00065 |     path.parent.mkdir(parents=True, exist_ok=True)
00066 |     path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
00067 | 
00068 | 
00069 | def json_safe(value: Any) -> Any:
00070 |     if isinstance(value, Path):
00071 |         return str(value)
00072 |     if isinstance(value, float):
00073 |         return value if math.isfinite(value) else None
00074 |     if isinstance(value, dict):
00075 |         return {str(key): json_safe(item) for key, item in value.items()}
00076 |     if isinstance(value, (list, tuple)):
00077 |         return [json_safe(item) for item in value]
00078 |     return value
00079 | 
00080 | 
00081 | def read_csv_rows(path: Path) -> list[dict[str, str]]:
00082 |     if not path.exists():
00083 |         return []
00084 |     with path.open(encoding="utf-8", newline="") as handle:
00085 |         return list(csv.DictReader(handle))
00086 | 
00087 | 
00088 | def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
00089 |     path.parent.mkdir(parents=True, exist_ok=True)
00090 |     if fieldnames is None:
00091 |         fieldnames = sorted({key for row in rows for key in row}) or ["status"]
00092 |     with path.open("w", encoding="utf-8", newline="") as handle:
00093 |         writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
00094 |         writer.writeheader()
00095 |         writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])
00096 | 
00097 | 
00098 | def run_git_commit(repo: Path) -> dict[str, Any]:
00099 |     if not (repo / ".git").exists():
00100 |         return {"path": str(repo), "commit": None, "dirty": None}
00101 |     commit = subprocess.run(
00102 |         ["git", "-C", str(repo), "rev-parse", "HEAD"],
00103 |         check=False,
00104 |         text=True,
00105 |         capture_output=True,
00106 |     )
00107 |     dirty = subprocess.run(
00108 |         ["git", "-C", str(repo), "status", "--porcelain"],
00109 |         check=False,
00110 |         text=True,
00111 |         capture_output=True,
00112 |     )
00113 |     branch = subprocess.run(
00114 |         ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
00115 |         check=False,
00116 |         text=True,
00117 |         capture_output=True,
00118 |     )
00119 |     return {
00120 |         "path": str(repo),
00121 |         "commit": commit.stdout.strip() if commit.returncode == 0 else None,
00122 |         "branch": branch.stdout.strip() if branch.returncode == 0 else None,
00123 |         "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
00124 |     }
00125 | 
00126 | 
00127 | def deeph_subprocess_env(deeph_repo: Path) -> dict[str, str]:
00128 |     """Return an environment that imports the editable DeepH checkout first."""
00129 |     env = os.environ.copy()
00130 |     repo_path = str(deeph_repo.resolve())
00131 |     current = env.get("PYTHONPATH")
00132 |     env["PYTHONPATH"] = repo_path if not current else repo_path + os.pathsep + current
00133 |     env["PYTHONUNBUFFERED"] = "1"
00134 |     return env
00135 | 
00136 | 
00137 | def run_subprocess_streaming(
00138 |     command: list[str],
00139 |     *,
00140 |     cwd: Path,
00141 |     env: dict[str, str],
00142 |     stdout_path: Path,
00143 |     stderr_path: Path,
00144 |     prefix: str,
00145 |     epoch_total: int | None = None,
00146 | ) -> int:
00147 |     """Run a subprocess while teeing combined output to disk and parent stdout."""
00148 |     stdout_path.parent.mkdir(parents=True, exist_ok=True)
00149 |     stderr_path.parent.mkdir(parents=True, exist_ok=True)
00150 |     merged_env = dict(env)
00151 |     merged_env["PYTHONUNBUFFERED"] = "1"
00152 |     process = subprocess.Popen(
00153 |         command,
00154 |         cwd=cwd,
00155 |         env=merged_env,
00156 |         text=True,
00157 |         stdout=subprocess.PIPE,
00158 |         stderr=subprocess.STDOUT,
00159 |         bufsize=1,
00160 |     )
00161 |     epoch_pattern = re.compile(r"Epoch\s*#\s*(\d+)", re.IGNORECASE)
00162 |     with stdout_path.open("w", encoding="utf-8") as stdout_log:
00163 |         stdout_log.write("$ " + " ".join(command) + "\n")
00164 |         stdout_log.flush()
00165 |         if process.stdout is not None:
00166 |             for line in process.stdout:
00167 |                 stdout_log.write(line)
00168 |                 stdout_log.flush()
00169 |                 print(f"{prefix}{line}", end="", flush=True)
00170 |                 if epoch_total is not None:
00171 |                     match = epoch_pattern.search(line)
00172 |                     if match:
00173 |                         print(
00174 |                             f"[DEEPh-FAIR][epoch] reported_epoch={match.group(1)}/{epoch_total}",
00175 |                             flush=True,
00176 |                         )
00177 |             process.stdout.close()
00178 |     returncode = process.wait()
00179 |     stderr_path.write_text(
00180 |         "stderr was merged into stdout for live UI streaming; see "
00181 |         f"{stdout_path.name}\nreturncode={returncode}\n",
00182 |         encoding="utf-8",
00183 |     )
00184 |     return returncode
00185 | 
00186 | 
00187 | def stable_sample_from_row(row: dict[str, str]) -> str:
00188 |     sample_id = str(row.get("sample_id") or "").strip()
00189 |     if sample_id:
00190 |         return sample_id
00191 |     for key in ("frame_index", "source_frame_index", "global_sample_id"):
00192 |         value = str(row.get(key) or "").strip()
00193 |         if value and value.lower() != "nan":
00194 |             return value
00195 |     for key in ("sample_dir", "structure_path", "hamiltonian_path"):
00196 |         value = str(row.get(key) or "").strip()
00197 |         if value:
00198 |             return Path(value).parent.name if key != "sample_dir" else Path(value).name
00199 |     match = re.search(r"(\d+)$", sample_id)
00200 |     if match:
00201 |         return match.group(1)
00202 |     raise DeepHFairBenchmarkError("Cannot infer stable sample id from split manifest row.")
00203 | 
00204 | 
00205 | def split_manifest_paths(graph2mat_result_dir: Path) -> dict[str, Path]:
00206 |     split_root = graph2mat_result_dir / "splits"
00207 |     return {
00208 |         "train": split_root / "train_manifest.csv",
00209 |         "validation": split_root / "validation_manifest.csv",
00210 |         "test": split_root / "test_manifest.csv",
00211 |     }
00212 | 
00213 | 
00214 | def load_split_samples(graph2mat_result_dir: Path) -> list[SplitSample]:
00215 |     samples: list[SplitSample] = []
00216 |     for split, manifest_path in split_manifest_paths(graph2mat_result_dir).items():
00217 |         rows = read_csv_rows(manifest_path)
00218 |         if not rows:
00219 |             raise DeepHFairBenchmarkError(f"Missing or empty split manifest: {manifest_path}")
00220 |         for row in rows:
00221 |             sample = stable_sample_from_row(row)
00222 |             sample_dir = Path(row.get("sample_dir") or graph2mat_result_dir / "structures" / sample)
00223 |             structure_path = Path(row["structure_path"]) if row.get("structure_path") else sample_dir / "RUN.fdf"
00224 |             hamiltonian_path = Path(row["hamiltonian_path"]) if row.get("hamiltonian_path") else None
00225 |             metadata_path = Path(row["metadata_path"]) if row.get("metadata_path") else sample_dir / "metadata.json"
00226 |             samples.append(
00227 |                 SplitSample(
00228 |                     sample=sample,
00229 |                     split=split,
00230 |                     sample_id=row.get("sample_id") or sample,
00231 |                     sample_dir=sample_dir,
00232 |                     structure_path=structure_path,
00233 |                     hamiltonian_path=hamiltonian_path,
00234 |                     metadata_path=metadata_path,
00235 |                     source_row=row,
00236 |                 )
00237 |             )
00238 |     return samples
00239 | 
00240 | 
00241 | def sample_limit_by_split(samples: list[SplitSample], limit: int | None) -> list[SplitSample]:
00242 |     if limit is None:
00243 |         return samples
00244 |     selected: list[SplitSample] = []
00245 |     counts: dict[str, int] = {}
00246 |     for sample in samples:
00247 |         count = counts.get(sample.split, 0)
00248 |         if count < limit:
00249 |             selected.append(sample)
00250 |             counts[sample.split] = count + 1
00251 |     return selected
00252 | 
00253 | 
00254 | def find_named_file(search_dirs: Iterable[Path], suffix: str, system_label: str = "graphene") -> Path | None:
00255 |     preferred = f"{system_label}{suffix}"
00256 |     for directory in search_dirs:
00257 |         candidate = directory / preferred
00258 |         if candidate.exists() and candidate.is_file():
00259 |             return candidate
00260 |     for directory in search_dirs:
00261 |         matches = sorted(path for path in directory.glob(f"*{suffix}") if path.name not in FORBIDDEN_REFERENCE_NAMES)
00262 |         if matches:
00263 |             return matches[0]
00264 |     return None
00265 | 
00266 | 
00267 | def detect_forbidden_references(paths: Iterable[Path | None]) -> list[str]:
00268 |     return [str(path) for path in paths if path is not None and path.name in FORBIDDEN_REFERENCE_NAMES]
00269 | 
00270 | 
00271 | def copy_or_symlink(src: Path, dst: Path, *, symlink: bool) -> None:
00272 |     dst.parent.mkdir(parents=True, exist_ok=True)
00273 |     if dst.exists() or dst.is_symlink():
00274 |         dst.unlink()
00275 |     if symlink:
00276 |         os.symlink(src, dst)
00277 |     else:
00278 |         shutil.copy2(src, dst)
00279 | 
00280 | 
00281 | def ensure_static_siesta_flags(run_fdf: Path) -> None:
00282 |     """Add DeepH/SIESTA export flags without changing physics controls."""
00283 |     text = run_fdf.read_text(encoding="utf-8", errors="replace")
00284 |     additions = []
00285 |     lower = text.lower()
00286 |     if "savehs" not in lower:
00287 |         additions.append("SaveHS                           true")
00288 |     if "save.hs" not in lower:
00289 |         additions.append("Save.HS                          T")
00290 |     if "ts.hs.save" not in lower:
00291 |         additions.append("TS.HS.Save                       T")
00292 |     if "xml.write" not in lower:
00293 |         additions.append("XML.Write                        T")
00294 |     if additions:
00295 |         run_fdf.write_text(text.rstrip() + "\n\n# DeepH fair-comparison export flags\n" + "\n".join(additions) + "\n", encoding="utf-8")
00296 | 
00297 | 
00298 | def infer_pseudo_dir_from_manifest(graph2mat_result_dir: Path) -> Path | None:
00299 |     manifest = read_json(graph2mat_result_dir / "manifest.json")
00300 |     provenance = manifest.get("material_provenance") if isinstance(manifest.get("material_provenance"), dict) else {}
00301 |     bundle_path = provenance.get("material_bundle_path") or manifest.get("material_bundle_path")
00302 |     if bundle_path:
00303 |         candidate = Path(bundle_path).parent / "pseudos"
00304 |         if candidate.exists():
00305 |             return candidate
00306 |     candidate = REPO_ROOT / "materials" / "graphene" / "pseudos"
00307 |     return candidate if candidate.exists() else None
00308 | 
00309 | 
00310 | def run_siesta_static(work_dir: Path, siesta_command: str, graph2mat_result_dir: Path) -> dict[str, Any]:
00311 |     pseudo_dir = infer_pseudo_dir_from_manifest(graph2mat_result_dir)
00312 |     if pseudo_dir is not None:
00313 |         for pseudo in pseudo_dir.iterdir():
00314 |             if pseudo.is_file() and pseudo.suffix.lower() in {".psf", ".psml", ".psp"}:
00315 |                 copy_or_symlink(pseudo, work_dir / pseudo.name, symlink=False)
00316 |     ensure_static_siesta_flags(work_dir / "RUN.fdf")
00317 |     with (work_dir / "RUN.out").open("w", encoding="utf-8") as stdout:
00318 |         completed = subprocess.run(
00319 |             siesta_command,
00320 |             cwd=work_dir,
00321 |             stdin=(work_dir / "RUN.fdf").open("r", encoding="utf-8"),
00322 |             stdout=stdout,
00323 |             stderr=subprocess.STDOUT,
00324 |             shell=True,
00325 |             check=False,
00326 |             text=True,
00327 |         )
00328 |     return {"command": siesta_command, "returncode": completed.returncode}
00329 | 
00330 | 
00331 | def count_orbitals_from_orbital_types(path: Path) -> list[int]:
00332 |     counts: list[int] = []
00333 |     for line in path.read_text(encoding="utf-8").splitlines():
00334 |         tokens = [token for token in line.split() if token.strip()]
00335 |         if not tokens:
00336 |             continue
00337 |         counts.append(sum(2 * int(token) + 1 for token in tokens))
00338 |     return counts
00339 | 
00340 | 
00341 | def orbital_config_from_processed_sample(sample_dir: Path) -> list[dict[str, list[int]]]:
00342 |     elements = [int(float(value)) for value in (sample_dir / "element.dat").read_text(encoding="utf-8").split()]
00343 |     orbital_counts = count_orbitals_from_orbital_types(sample_dir / "orbital_types.dat")
00344 |     if len(elements) != len(orbital_counts):
00345 |         raise DeepHFairBenchmarkError(
00346 |             f"element/orbital count mismatch in {sample_dir}: {len(elements)} vs {len(orbital_counts)}"
00347 |         )
00348 |     max_orbitals_by_element: dict[int, int] = {}
00349 |     for element, count in zip(elements, orbital_counts, strict=True):
00350 |         max_orbitals_by_element[element] = max(max_orbitals_by_element.get(element, 0), count)
00351 |     entries: list[dict[str, list[int]]] = []
00352 |     for row_element in sorted(max_orbitals_by_element):
00353 |         for col_element in sorted(max_orbitals_by_element):
00354 |             pair = f"{row_element} {col_element}"
00355 |             for row_orbital in range(max_orbitals_by_element[row_element]):
00356 |                 for col_orbital in range(max_orbitals_by_element[col_element]):
00357 |                     entries.append({pair: [row_orbital, col_orbital]})
00358 |     return entries
00359 | 
00360 | 
00361 | def max_l_from_orbital_types(path: Path) -> int:
00362 |     values: list[int] = []
00363 |     for line in path.read_text(encoding="utf-8").splitlines():
00364 |         values.extend(int(token) for token in line.split())
00365 |     return max(values) if values else 0
00366 | 
00367 | 
00368 | def configparser_to_dict(config: ConfigParser) -> dict[str, dict[str, str]]:
00369 |     return {section: dict(config.items(section)) for section in config.sections()}
00370 | 
00371 | 
00372 | def write_deeph_config(path: Path, config: ConfigParser) -> None:
00373 |     path.parent.mkdir(parents=True, exist_ok=True)
00374 |     with path.open("w", encoding="utf-8") as handle:
00375 |         config.write(handle)
00376 | 
00377 | 
00378 | def make_split_ordered_processed_dir(
00379 |     *,
00380 |     processed_dir: Path,
00381 |     ordered_dir: Path,
00382 |     samples: list[SplitSample],
00383 |     seed: int,
00384 |     symlink: bool = True,
00385 |     shuffled_indices: list[int] | None = None,
00386 | ) -> dict[str, Any]:
00387 |     split_groups = {split: [sample.sample for sample in samples if sample.split == split] for split in ("train", "validation", "test")}
00388 |     total = sum(len(values) for values in split_groups.values())
00389 |     if total == 0:
00390 |         raise DeepHFairBenchmarkError("No split samples available for DeepH split ordering.")
00391 |     train_size = len(split_groups["train"])
00392 |     validation_size = len(split_groups["validation"])
00393 |     test_size = len(split_groups["test"])
00394 |     if shuffled_indices is None:
00395 |         try:
00396 |             import numpy as np
00397 |         except ModuleNotFoundError as exc:
00398 |             raise DeepHFairBenchmarkError(
00399 |                 "numpy is required to reproduce DeepH's np.random.shuffle split ordering; "
00400 |                 "run this step in the DeepH environment or pass explicit shuffled_indices."
00401 |             ) from exc
00402 |         indices = list(range(total))
00403 |         np.random.seed(seed)
00404 |         np.random.shuffle(indices)
00405 |         indices = [int(index) for index in indices]
00406 |     else:
00407 |         indices = list(shuffled_indices)
00408 |         if sorted(indices) != list(range(total)):
00409 |             raise DeepHFairBenchmarkError(
00410 |                 f"Invalid shuffled_indices for {total} samples: expected a permutation of 0..{total - 1}."
00411 |             )
00412 |     desired_by_index: dict[int, str] = {}
00413 |     for index, sample_id in zip(indices[:train_size], split_groups["train"], strict=True):
00414 |         desired_by_index[index] = sample_id
00415 |     offset = train_size
00416 |     for index, sample_id in zip(indices[offset:offset + validation_size], split_groups["validation"], strict=True):
00417 |         desired_by_index[index] = sample_id
00418 |     offset += validation_size
00419 |     for index, sample_id in zip(indices[offset:offset + test_size], split_groups["test"], strict=True):
00420 |         desired_by_index[index] = sample_id
00421 |     ordered_dir.mkdir(parents=True, exist_ok=True)
00422 |     mapping_rows: list[dict[str, Any]] = []
00423 |     for index in range(total):
00424 |         sample_id = desired_by_index[index]
00425 |         src = processed_dir / sample_id
00426 |         if not src.exists():
00427 |             raise DeepHFairBenchmarkError(f"Processed DeepH sample missing: {src}")
00428 |         dst = ordered_dir / f"{index:06d}__sample_{sample_id}"
00429 |         if dst.exists() or dst.is_symlink():
00430 |             if dst.is_dir() and not dst.is_symlink():
00431 |                 shutil.rmtree(dst)
00432 |             else:
00433 |                 dst.unlink()
00434 |         if symlink:
00435 |             dst.mkdir(parents=True, exist_ok=True)
00436 |             for child in src.iterdir():
00437 |                 os.symlink(child, dst / child.name, target_is_directory=child.is_dir())
00438 |         else:
00439 |             shutil.copytree(src, dst)
00440 |         split = next(split for split, ids in split_groups.items() if sample_id in ids)
00441 |         mapping_rows.append({"ordered_index": index, "ordered_name": dst.name, "sample": sample_id, "split": split, "source": str(src)})
00442 |     return {
00443 |         "seed": seed,
00444 |         "ordered_dir": str(ordered_dir),
00445 |         "train_size": train_size,
00446 |         "validation_size": validation_size,
00447 |         "test_size": test_size,
00448 |         "total": total,
00449 |         "train_ratio": train_size / total,
00450 |         "validation_ratio": validation_size / total,
00451 |         "test_ratio": test_size / total,
00452 |         "mapping_rows": mapping_rows,
00453 |     }
```

## `Comparison/scripts/deeph_split_audit.py`

SHA-256: `a01e31ef5eceafab5a5d19226365dc737692e2ae82d10797b43ac47b3505287e`

```py
00001 | #!/usr/bin/env python3
00002 | """Audit DeepH train/validation/test splits against a frozen benchmark split."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import configparser
00007 | import csv
00008 | import json
00009 | import os
00010 | from pathlib import Path
00011 | from typing import Any
00012 | 
00013 | 
00014 | SCHEMA = "graph2mat_deeph_deeph_split_audit_v1"
00015 | STATUS_VALID = "valid"
00016 | STATUS_UNVERIFIED = "invalid_unverified_deeph_split"
00017 | STATUS_INCOMPATIBLE = "invalid_incompatible_splits"
00018 | SPLITS = ("train", "validation", "test")
00019 | 
00020 | 
00021 | def write_json(path: Path, payload: dict[str, Any]) -> None:
00022 |     path.parent.mkdir(parents=True, exist_ok=True)
00023 |     path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
00024 | 
00025 | 
00026 | def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
00027 |     path.parent.mkdir(parents=True, exist_ok=True)
00028 |     fields = [
00029 |         "sample_id",
00030 |         "deeph_dataset_index",
00031 |         "processed_dir",
00032 |         "frozen_split",
00033 |         "actual_deeph_split",
00034 |         "status",
00035 |     ]
00036 |     with path.open("w", encoding="utf-8", newline="") as handle:
00037 |         writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
00038 |         writer.writeheader()
00039 |         writer.writerows(rows)
00040 | 
00041 | 
00042 | def load_train_split_config(train_config_path: Path) -> tuple[int | None, dict[str, float], list[str]]:
00043 |     errors: list[str] = []
00044 |     if not train_config_path.exists():
00045 |         return None, {}, [f"DeepH train config is missing: {train_config_path}"]
00046 |     config = configparser.ConfigParser()
00047 |     config.read(train_config_path)
00048 |     try:
00049 |         seed = config.getint("basic", "seed")
00050 |     except Exception as exc:  # configparser exceptions vary by missing section/key.
00051 |         seed = None
00052 |         errors.append(f"DeepH train config does not define [basic] seed: {exc}")
00053 |     ratios: dict[str, float] = {}
00054 |     for key in ("train_ratio", "val_ratio", "test_ratio"):
00055 |         try:
00056 |             ratios[key] = config.getfloat("train", key)
00057 |         except Exception as exc:
00058 |             errors.append(f"DeepH train config does not define [train] {key}: {exc}")
00059 |     return seed, ratios, errors
00060 | 
00061 | 
00062 | def processed_sample_dirs(processed_dir: Path) -> list[Path]:
00063 |     processed_dir = Path(processed_dir)
00064 |     if not processed_dir.exists():
00065 |         return []
00066 |     folders: list[Path] = []
00067 |     for root, _dirs, files in os.walk(processed_dir):
00068 |         if "rc.h5" in files:
00069 |             folders.append(Path(root))
00070 |     return sorted(folders, key=lambda path: str(path))
00071 | 
00072 | 
00073 | def deeph_index_split_map(dataset_size: int, ratios: dict[str, float], seed: int) -> tuple[dict[int, str], list[str]]:
00074 |     errors: list[str] = []
00075 |     try:
00076 |         import numpy as np  # type: ignore[import-not-found]
00077 |     except ImportError:
00078 |         return {}, ["NumPy is required to reproduce DeepH np.random.shuffle split indices."]
00079 |     sizes = {
00080 |         "train": int(float(ratios.get("train_ratio", 0.0)) * dataset_size),
00081 |         "validation": int(float(ratios.get("val_ratio", 0.0)) * dataset_size),
00082 |         "test": int(float(ratios.get("test_ratio", 0.0)) * dataset_size),
00083 |     }
00084 |     if sum(sizes.values()) > dataset_size:
00085 |         errors.append(f"DeepH split sizes exceed dataset size: {sizes} > {dataset_size}")
00086 |     if any(size <= 0 for size in sizes.values()):
00087 |         errors.append(f"DeepH split sizes must be non-empty for benchmark comparability: {sizes}")
00088 |     if errors:
00089 |         return {}, errors
00090 |     indices = list(range(dataset_size))
00091 |     np.random.seed(int(seed))
00092 |     np.random.shuffle(indices)
00093 |     actual: dict[int, str] = {}
00094 |     cursor = 0
00095 |     for split in SPLITS:
00096 |         count = sizes[split]
00097 |         for index in indices[cursor : cursor + count]:
00098 |             actual[int(index)] = split
00099 |         cursor += count
00100 |     return actual, []
00101 | 
00102 | 
00103 | def _relative_key(path: Path, root: Path) -> str:
00104 |     try:
00105 |         return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
00106 |     except ValueError:
00107 |         return path.name
00108 | 
00109 | 
00110 | def _raw_mirror_by_relative_dir(raw_mirror: dict[str, Any]) -> dict[str, dict[str, Any]]:
00111 |     raw_root = Path(str(raw_mirror.get("raw_dir") or ""))
00112 |     rows: dict[str, dict[str, Any]] = {}
00113 |     for row in raw_mirror.get("rows") or []:
00114 |         if not isinstance(row, dict):
00115 |             continue
00116 |         raw_dir = Path(str(row.get("raw_dir") or ""))
00117 |         rows[_relative_key(raw_dir, raw_root)] = dict(row)
00118 |     return rows
00119 | 
00120 | 
00121 | def _frozen_split_by_sample_id(frozen_split_manifest: dict[str, Any]) -> dict[str, str]:
00122 |     result: dict[str, str] = {}
00123 |     for row in frozen_split_manifest.get("rows") or []:
00124 |         if not isinstance(row, dict):
00125 |             continue
00126 |         sample_id = str(row.get("sample_id") or row.get("graph2mat_sample_id") or row.get("deeph_sample_id") or "").strip()
00127 |         split = str(row.get("split") or "").strip()
00128 |         if sample_id and split:
00129 |             result[sample_id] = split
00130 |     return result
00131 | 
00132 | 
00133 | def audit_deeph_split(
00134 |     *,
00135 |     frozen_split_manifest: dict[str, Any],
00136 |     raw_mirror: dict[str, Any],
00137 |     processed_dir: Path,
00138 |     train_config_path: Path,
00139 |     output_json: Path | None = None,
00140 |     output_csv: Path | None = None,
00141 | ) -> dict[str, Any]:
00142 |     errors: list[str] = []
00143 |     warnings: list[str] = []
00144 |     seed, ratios, config_errors = load_train_split_config(train_config_path)
00145 |     errors.extend(config_errors)
00146 |     raw_seed = raw_mirror.get("seed")
00147 |     if seed is not None and raw_seed is not None and int(raw_seed) != int(seed):
00148 |         errors.append(f"DeepH raw mirror seed {raw_seed} does not match train config seed {seed}.")
00149 | 
00150 |     raw_by_dir = _raw_mirror_by_relative_dir(raw_mirror)
00151 |     frozen_by_sample = _frozen_split_by_sample_id(frozen_split_manifest)
00152 |     folders = processed_sample_dirs(processed_dir)
00153 |     if not folders:
00154 |         errors.append(f"No DeepH processed sample directories with rc.h5 found under {processed_dir}.")
00155 | 
00156 |     processed_root = Path(processed_dir)
00157 |     unknown_processed: list[str] = []
00158 |     rows: list[dict[str, Any]] = []
00159 |     if seed is not None and ratios and folders:
00160 |         split_by_index, split_errors = deeph_index_split_map(len(folders), ratios, seed)
00161 |         errors.extend(split_errors)
00162 |         if not split_errors:
00163 |             for index, folder in enumerate(folders):
00164 |                 key = _relative_key(folder, processed_root)
00165 |                 mirror_row = raw_by_dir.get(key)
00166 |                 if mirror_row is None:
00167 |                     unknown_processed.append(str(folder))
00168 |                     continue
00169 |                 sample_id = str(mirror_row.get("sample_id") or "").strip()
00170 |                 frozen_split = frozen_by_sample.get(sample_id, "")
00171 |                 actual_split = split_by_index.get(index, "")
00172 |                 status = "ok" if frozen_split and actual_split == frozen_split else "mismatch"
00173 |                 rows.append(
00174 |                     {
00175 |                         "sample_id": sample_id,
00176 |                         "deeph_dataset_index": index,
00177 |                         "processed_dir": str(folder),
00178 |                         "frozen_split": frozen_split,
00179 |                         "actual_deeph_split": actual_split,
00180 |                         "status": status,
00181 |                     }
00182 |                 )
00183 |     if unknown_processed:
00184 |         errors.append("Processed DeepH samples cannot be mapped to raw mirror rows: " + ", ".join(unknown_processed[:10]))
00185 | 
00186 |     raw_sample_ids = {str(row.get("sample_id") or "") for row in raw_by_dir.values()}
00187 |     processed_sample_ids = {row["sample_id"] for row in rows if row.get("sample_id")}
00188 |     missing_processed_sample_ids = sorted(raw_sample_ids - processed_sample_ids)
00189 |     if missing_processed_sample_ids:
00190 |         errors.append(
00191 |             "DeepH processed output is missing raw mirror samples: "
00192 |             + ", ".join(missing_processed_sample_ids[:10])
00193 |         )
00194 |     if set(frozen_by_sample) != processed_sample_ids:
00195 |         errors.append(
00196 |             "DeepH processed sample IDs do not match frozen split IDs: "
00197 |             f"missing={sorted(set(frozen_by_sample) - processed_sample_ids)[:10]} "
00198 |             f"extra={sorted(processed_sample_ids - set(frozen_by_sample))[:10]}"
00199 |         )
00200 | 
00201 |     mismatches = [row for row in rows if row.get("status") != "ok"]
00202 |     if mismatches:
00203 |         warnings.append(f"{len(mismatches)} DeepH split assignments differ from frozen split.")
00204 | 
00205 |     if errors and not mismatches:
00206 |         status = STATUS_UNVERIFIED
00207 |     elif errors or mismatches:
00208 |         status = STATUS_INCOMPATIBLE
00209 |     else:
00210 |         status = STATUS_VALID
00211 |     valid = status == STATUS_VALID
00212 |     audit = {
00213 |         "schema": SCHEMA,
00214 |         "status": status,
00215 |         "valid": valid,
00216 |         "comparability_status": "valid" if valid else status,
00217 |         "scientific_status": "valid" if valid else STATUS_INCOMPATIBLE,
00218 |         "robust_winner_allowed": valid,
00219 |         "frozen_split_hash": frozen_split_manifest.get("split_hash"),
00220 |         "raw_mirror_seed": raw_seed,
00221 |         "train_config_seed": seed,
00222 |         "split_ratios": ratios,
00223 |         "processed_dir": str(processed_dir),
00224 |         "train_config_path": str(train_config_path),
00225 |         "dataset_size": len(folders),
00226 |         "rows": rows,
00227 |         "mismatched_rows": mismatches,
00228 |         "errors": errors,
00229 |         "warnings": warnings,
00230 |     }
00231 |     if output_json is not None:
00232 |         write_json(output_json, audit)
00233 |         audit["path"] = str(output_json)
00234 |     if output_csv is not None:
00235 |         write_csv(output_csv, rows)
00236 |         audit["csv_path"] = str(output_csv)
00237 |     return audit
```

## `Comparison/scripts/g2m_deeph_test_blindness.py`

SHA-256: `0e1a5ca629a87f5584949b74c3ae94aea84ad990684c8ec3d29e01d133bed3ff`

```py
00001 | #!/usr/bin/env python3
00002 | """Protocol-level test blindness helpers for Graph2Mat-vs-DeepH final benchmarks."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import json
00007 | import time
00008 | from pathlib import Path
00009 | from typing import Any
00010 | 
00011 | 
00012 | FINAL_BENCHMARK_MODES = {"final", "final_publication", "paper", "paper_ready", "publicable"}
00013 | SEARCH_STAGE = "search"
00014 | ROBUST_VALIDATION_STAGE = "robust_validation"
00015 | FINAL_TEST_STAGE = "final_test"
00016 | EXPLORATORY_STAGE = "exploratory"
00017 | VALIDATION_SPLITS = {"validation", "val"}
00018 | TEST_SPLITS = {"test"}
00019 | 
00020 | 
00021 | def _parse_bool(value: Any, default: bool = False) -> bool:
00022 |     if value is None or value == "":
00023 |         return default
00024 |     if isinstance(value, bool):
00025 |         return value
00026 |     if isinstance(value, (int, float)):
00027 |         return bool(value)
00028 |     if isinstance(value, str):
00029 |         return value.strip().lower() in {"1", "true", "yes", "y", "on", "si", "sí"}
00030 |     return default
00031 | 
00032 | 
00033 | def is_final_benchmark_mode(payload: dict[str, Any] | None) -> bool:
00034 |     """Return true when payload requests the strict final/publicable benchmark protocol."""
00035 | 
00036 |     payload = payload or {}
00037 |     for key in ("benchmark_mode", "protocol_mode", "mode"):
00038 |         value = str(payload.get(key) or "").strip().lower()
00039 |         if value in FINAL_BENCHMARK_MODES:
00040 |             return True
00041 |     if _parse_bool(payload.get("final_benchmark"), False) or _parse_bool(payload.get("paper_ready"), False):
00042 |         return True
00043 |     protocol = payload.get("protocol")
00044 |     if isinstance(protocol, dict):
00045 |         final_policy = protocol.get("final_test_policy") if isinstance(protocol.get("final_test_policy"), dict) else {}
00046 |         if final_policy.get("policy") == "locked_until_final" and final_policy.get("locked_during_search") is True:
00047 |             return True
00048 |     return False
00049 | 
00050 | 
00051 | def protocol_stage_from_payload(payload: dict[str, Any] | None, *, default: str | None = None) -> str:
00052 |     payload = payload or {}
00053 |     raw = str(payload.get("protocol_stage") or "").strip().lower()
00054 |     if raw:
00055 |         return raw
00056 |     if default:
00057 |         return default
00058 |     return SEARCH_STAGE if is_final_benchmark_mode(payload) else EXPLORATORY_STAGE
00059 | 
00060 | 
00061 | def metric_split(row: dict[str, Any]) -> str:
00062 |     for key in ("metric_split", "evaluation_split", "selection_split", "split", "dataset_split"):
00063 |         value = str(row.get(key) or "").strip().lower()
00064 |         if value:
00065 |             return value
00066 |     stage = str(row.get("protocol_stage") or row.get("stage") or "").strip().lower()
00067 |     if stage == FINAL_TEST_STAGE:
00068 |         return "test"
00069 |     return ""
00070 | 
00071 | 
00072 | def row_contains_test_metric(row: dict[str, Any]) -> bool:
00073 |     split = metric_split(row)
00074 |     if split in TEST_SPLITS:
00075 |         return True
00076 |     if row.get("uses_test_metrics") is True:
00077 |         return True
00078 |     scope = str(row.get("metric_scope") or row.get("scope") or "").strip().lower()
00079 |     return scope in TEST_SPLITS
00080 | 
00081 | 
00082 | def assert_no_test_metrics_for_search(rows: list[dict[str, Any]], *, stage: str = SEARCH_STAGE) -> None:
00083 |     """Fail closed if search/robust-validation inputs contain test metrics."""
00084 | 
00085 |     normalized_stage = str(stage or SEARCH_STAGE).strip().lower()
00086 |     if normalized_stage not in {SEARCH_STAGE, ROBUST_VALIDATION_STAGE}:
00087 |         return
00088 |     offenders = [
00089 |         str(row.get("config_id") or row.get("run_id") or index)
00090 |         for index, row in enumerate(rows)
00091 |         if isinstance(row, dict) and row_contains_test_metric(row)
00092 |     ]
00093 |     if offenders:
00094 |         raise RuntimeError(
00095 |             "Test metrics are locked during "
00096 |             f"{normalized_stage}; offending rows: {', '.join(offenders[:10])}"
00097 |         )
00098 | 
00099 | 
00100 | def _metric_value(row: dict[str, Any], metric: str) -> float | None:
00101 |     for key in (metric, f"{metric}_mean"):
00102 |         value = row.get(key)
00103 |         if value in (None, ""):
00104 |             continue
00105 |         try:
00106 |             number = float(value)
00107 |         except (TypeError, ValueError):
00108 |             continue
00109 |         if number == number and number not in (float("inf"), float("-inf")):
00110 |             return number
00111 |     return None
00112 | 
00113 | 
00114 | def select_top_k_validation_only(
00115 |     rows: list[dict[str, Any]],
00116 |     *,
00117 |     metric: str,
00118 |     mode: str,
00119 |     k_per_model: int,
00120 |     stage: str = SEARCH_STAGE,
00121 | ) -> list[dict[str, Any]]:
00122 |     """Select top-k configs per model using validation metrics only."""
00123 | 
00124 |     if k_per_model <= 0:
00125 |         raise RuntimeError("k_per_model must be positive.")
00126 |     normalized_mode = str(mode or "").strip().lower()
00127 |     if normalized_mode not in {"min", "max"}:
00128 |         raise RuntimeError("mode must be min or max.")
00129 |     assert_no_test_metrics_for_search(rows, stage=stage)
00130 |     validation_rows = [
00131 |         row
00132 |         for row in rows
00133 |         if isinstance(row, dict)
00134 |         and metric_split(row) in VALIDATION_SPLITS
00135 |         and _metric_value(row, metric) is not None
00136 |     ]
00137 |     if not validation_rows:
00138 |         raise RuntimeError(
00139 |             "No validation metric rows are available for top-k selection; "
00140 |             "search/top-k selection must not use test metrics."
00141 |         )
00142 |     reverse = normalized_mode == "max"
00143 |     selected: list[dict[str, Any]] = []
00144 |     models = sorted({str(row.get("model") or "") for row in validation_rows if row.get("model")})
00145 |     for model in models:
00146 |         model_rows = [row for row in validation_rows if str(row.get("model") or "") == model]
00147 |         model_rows.sort(key=lambda row: (_metric_value(row, metric), str(row.get("config_id") or "")), reverse=reverse)
00148 |         selected.extend(model_rows[:k_per_model])
00149 |     return selected
00150 | 
00151 | 
00152 | def validate_final_evaluation_inputs(
00153 |     *,
00154 |     selected_runs: list[dict[str, Any]],
00155 |     metric_rows: list[dict[str, Any]],
00156 |     stage: str,
00157 |     metric: str,
00158 | ) -> None:
00159 |     """Validate stage-specific metric availability for strict final protocols."""
00160 | 
00161 |     normalized_stage = str(stage or "").strip().lower()
00162 |     if normalized_stage in {SEARCH_STAGE, ROBUST_VALIDATION_STAGE}:
00163 |         assert_no_test_metrics_for_search(metric_rows, stage=normalized_stage)
00164 |         return
00165 |     if normalized_stage != FINAL_TEST_STAGE:
00166 |         raise RuntimeError(f"Unsupported final benchmark protocol stage: {stage}")
00167 |     if not selected_runs:
00168 |         raise RuntimeError("final_test requires selected final runs.")
00169 |     test_rows = [
00170 |         row
00171 |         for row in metric_rows
00172 |         if isinstance(row, dict)
00173 |         and metric_split(row) in TEST_SPLITS
00174 |         and _metric_value(row, metric) is not None
00175 |     ]
00176 |     if not test_rows:
00177 |         raise RuntimeError("final_test requires test metrics for selected final runs.")
00178 |     selected_keys = {
00179 |         (str(row.get("model") or ""), str(row.get("config_id") or ""))
00180 |         for row in selected_runs
00181 |     }
00182 |     measured_keys = {
00183 |         (str(row.get("model") or ""), str(row.get("config_id") or ""))
00184 |         for row in test_rows
00185 |     }
00186 |     missing = sorted(selected_keys - measured_keys)
00187 |     if missing:
00188 |         raise RuntimeError(
00189 |             "final_test is missing test metrics for selected runs: "
00190 |             + ", ".join(f"{model}/{config}" for model, config in missing[:10])
00191 |         )
00192 | 
00193 | 
00194 | def search_stage_record_fields() -> dict[str, Any]:
00195 |     return {
00196 |         "protocol_stage": SEARCH_STAGE,
00197 |         "test_metrics_locked": True,
00198 |         "test_metrics_status": "locked_until_final",
00199 |         "metric_split": "validation",
00200 |         "final_test_evaluation_allowed": False,
00201 |     }
00202 | 
00203 | 
00204 | def build_search_stage_manifest(
00205 |     *,
00206 |     run_root: Path,
00207 |     summary: dict[str, Any],
00208 |     payload: dict[str, Any] | None = None,
00209 | ) -> dict[str, Any]:
00210 |     payload = payload or {}
00211 |     manifest = {
00212 |         "schema": "graph2mat_deeph_test_blindness_v1",
00213 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
00214 |         "run_root": str(run_root),
00215 |         "protocol_stage": SEARCH_STAGE,
00216 |         "final_benchmark_mode": is_final_benchmark_mode(payload),
00217 |         "final_test_locked": True,
00218 |         "search_may_compute_splits": ["train", "validation"],
00219 |         "search_must_not_compute_splits": ["test"],
00220 |         "top_k_selection_split": "validation",
00221 |         "final_test_stage": FINAL_TEST_STAGE,
00222 |         "training_sweep_status": summary.get("status"),
00223 |         "completed_runs": len([row for row in summary.get("runs") or [] if row.get("status") == "completed"]),
00224 |         "failed_runs": len(summary.get("failed_runs") or []),
00225 |         "selected_final_runs": [],
00226 |         "final_test_status": "pending_selection",
00227 |     }
00228 |     path = run_root / "summary" / "test_blindness_manifest.json"
00229 |     path.parent.mkdir(parents=True, exist_ok=True)
00230 |     path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
00231 |     manifest["path"] = str(path)
00232 |     return manifest
00233 | 
00234 | 
00235 | def build_final_test_stage_manifest(
00236 |     *,
00237 |     run_root: Path,
00238 |     selected_runs: list[dict[str, Any]],
00239 |     metric_rows: list[dict[str, Any]],
00240 |     metric: str,
00241 | ) -> dict[str, Any]:
00242 |     """Write a manifest proving final-test metrics exist only for selected final runs."""
00243 | 
00244 |     validate_final_evaluation_inputs(
00245 |         selected_runs=selected_runs,
00246 |         metric_rows=metric_rows,
00247 |         stage=FINAL_TEST_STAGE,
00248 |         metric=metric,
00249 |     )
00250 |     selected_keys = {
00251 |         (str(row.get("model") or ""), str(row.get("config_id") or ""))
00252 |         for row in selected_runs
00253 |     }
00254 |     final_rows = [
00255 |         row
00256 |         for row in metric_rows
00257 |         if isinstance(row, dict)
00258 |         and metric_split(row) in TEST_SPLITS
00259 |         and (str(row.get("model") or ""), str(row.get("config_id") or "")) in selected_keys
00260 |     ]
00261 |     manifest = {
00262 |         "schema": "graph2mat_deeph_test_blindness_v1",
00263 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
00264 |         "run_root": str(run_root),
00265 |         "protocol_stage": FINAL_TEST_STAGE,
00266 |         "final_test_locked": False,
00267 |         "selection_required_before_final_test": True,
00268 |         "selected_final_runs": selected_runs,
00269 |         "final_test_metric": metric,
00270 |         "final_test_metric_rows": len(final_rows),
00271 |         "final_test_status": "completed",
00272 |     }
00273 |     path = run_root / "summary" / "final_test_manifest.json"
00274 |     path.parent.mkdir(parents=True, exist_ok=True)
00275 |     path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
00276 |     manifest["path"] = str(path)
00277 |     return manifest
```

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

## `Comparison/scripts/material_provenance.py`

SHA-256: `0b1588ff193d2ccadbfb45e1e31a707d136c8ed6d23a0b7257f59cba5ba6d71f`

```py
00001 | #!/usr/bin/env python3
00002 | """Small helpers for material provenance in manifests and aggregate rows."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import hashlib
00007 | import json
00008 | from pathlib import Path
00009 | from typing import Any
00010 | 
00011 | 
00012 | INCOMPATIBLE_MATERIAL_WARNING_CODE = "INCOMPATIBLE_MATERIAL_PROVENANCE"
00013 | 
00014 | MATERIAL_FLAT_FIELDS = (
00015 |     "material_label",
00016 |     "material_bundle_path",
00017 |     "material_source",
00018 |     "material_preset",
00019 |     "material_structure_type",
00020 |     "material_species",
00021 |     "material_atom_count",
00022 |     "material_cell_summary",
00023 |     "fdf_sha256",
00024 |     "pseudopotential_sha256_by_species",
00025 |     "basis_sha256_by_species",
00026 |     "siesta_settings_hash",
00027 |     "siesta_output_flags",
00028 |     "graph2mat_config_hash",
00029 |     "split_manifest_hash",
00030 |     "dataset_recipe",
00031 |     "dataset_recipe_parameters",
00032 |     "reference_matrix_sha256",
00033 |     "prediction_matrix_sha256",
00034 |     "material_identity_hash",
00035 |     "material_compatibility_hash",
00036 | )
00037 | 
00038 | MATERIAL_MAP_FIELDS = (
00039 |     "material_label_by_method",
00040 |     "material_identity_hash_by_method",
00041 |     "material_compatibility_hash_by_method",
00042 |     "fdf_sha256_by_method",
00043 |     "pseudopotential_sha256_by_method",
00044 |     "basis_sha256_by_method",
00045 | )
00046 | 
00047 | 
00048 | def read_json_file(path: Path) -> dict[str, Any]:
00049 |     if not path.exists() or not path.is_file():
00050 |         return {}
00051 |     with path.open("r", encoding="utf-8") as handle:
00052 |         payload = json.load(handle)
00053 |     return payload if isinstance(payload, dict) else {}
00054 | 
00055 | 
00056 | def stable_json_text(value: Any) -> str:
00057 |     return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
00058 | 
00059 | 
00060 | def stable_json_hash(value: Any) -> str:
00061 |     return hashlib.sha256(stable_json_text(value).encode("utf-8")).hexdigest()
00062 | 
00063 | 
00064 | def file_collection_hash(paths: list[Path]) -> str:
00065 |     entries = []
00066 |     for path in sorted({item.resolve() for item in paths if item.exists() and item.is_file()}):
00067 |         digest = hashlib.sha256()
00068 |         with path.open("rb") as handle:
00069 |             for chunk in iter(lambda: handle.read(1024 * 1024), b""):
00070 |                 digest.update(chunk)
00071 |         entries.append((path.name, digest.hexdigest()))
00072 |     return stable_json_hash(entries) if entries else ""
00073 | 
00074 | 
00075 | def _first_present(*values: Any) -> Any:
00076 |     for value in values:
00077 |         if value not in (None, "", {}, [], False):
00078 |             return value
00079 |     return None
00080 | 
00081 | 
00082 | def _as_mapping(value: Any) -> dict[str, Any]:
00083 |     return value if isinstance(value, dict) else {}
00084 | 
00085 | 
00086 | def _normalise_species(species: Any) -> list[dict[str, Any]] | list[str]:
00087 |     if not isinstance(species, list):
00088 |         return []
00089 |     if all(isinstance(item, dict) for item in species):
00090 |         return sorted(
00091 |             (
00092 |                 {
00093 |                     key: item[key]
00094 |                     for key in ("index", "atomic_number", "label")
00095 |                     if key in item
00096 |                 }
00097 |                 for item in species
00098 |             ),
00099 |             key=lambda item: (int(item.get("index") or 0), str(item.get("label") or "")),
00100 |         )
00101 |     return sorted(str(item) for item in species if item not in (None, ""))
00102 | 
00103 | 
00104 | def _atom_count_from_material(material: dict[str, Any]) -> int | None:
00105 |     for key in ("atom_count", "number_of_atoms", "n_atoms"):
00106 |         try:
00107 |             value = material.get(key)
00108 |             if value not in (None, ""):
00109 |                 return int(value)
00110 |         except (TypeError, ValueError):
00111 |             pass
00112 |     atoms = material.get("atoms")
00113 |     if isinstance(atoms, list):
00114 |         return len(atoms)
00115 |     return None
00116 | 
00117 | 
00118 | def _basis_hashes_from_graph2mat(graph2mat: dict[str, Any]) -> dict[str, str]:
00119 |     by_species = graph2mat.get("basis_files_by_species")
00120 |     if not isinstance(by_species, dict):
00121 |         return {}
00122 |     hashes: dict[str, str] = {}
00123 |     for species, item in by_species.items():
00124 |         if isinstance(item, dict) and item.get("sha256"):
00125 |             hashes[str(species)] = str(item["sha256"])
00126 |     return dict(sorted(hashes.items()))
00127 | 
00128 | 
00129 | def _first_material_mapping(sources: tuple[dict[str, Any], ...]) -> dict[str, Any]:
00130 |     for source in sources:
00131 |         material = source.get("material")
00132 |         if isinstance(material, dict):
00133 |             return material
00134 |     for source in sources:
00135 |         if source.get("label") or source.get("material_label"):
00136 |             return source
00137 |     return {}
00138 | 
00139 | 
00140 | def flatten_material_provenance(*sources: dict[str, Any]) -> dict[str, Any]:
00141 |     """Return a deterministic, JSON-safe material provenance summary.
00142 | 
00143 |     The current repository has material metadata in several sidecars. This
00144 |     helper accepts any combination of those dictionaries and extracts the common
00145 |     fields without requiring legacy archives to contain all of them.
00146 |     """
00147 | 
00148 |     valid_sources = tuple(source for source in sources if isinstance(source, dict) and source)
00149 |     if not valid_sources:
00150 |         return {}
00151 | 
00152 |     material = _first_material_mapping(valid_sources)
00153 |     graph2mat = next(
00154 |         (_as_mapping(source.get("graph2mat")) for source in valid_sources if isinstance(source.get("graph2mat"), dict)),
00155 |         {},
00156 |     )
00157 |     reference = next(
00158 |         (_as_mapping(source.get("reference_matrix")) for source in valid_sources if isinstance(source.get("reference_matrix"), dict)),
00159 |         {},
00160 |     )
00161 |     prediction = next(
00162 |         (_as_mapping(source.get("prediction_matrix")) for source in valid_sources if isinstance(source.get("prediction_matrix"), dict)),
00163 |         {},
00164 |     )
00165 | 
00166 |     material_species = _normalise_species(
00167 |         _first_present(
00168 |             material.get("species"),
00169 |             *(source.get("material_species") for source in valid_sources),
00170 |         )
00171 |     )
00172 |     basis_hashes = _first_present(
00173 |         *(source.get("basis_sha256_by_species") for source in valid_sources),
00174 |         material.get("basis_sha256_by_species"),
00175 |         material.get("basis_file_sha256"),
00176 |         _basis_hashes_from_graph2mat(graph2mat),
00177 |     ) or {}
00178 |     pseudo_hashes = _first_present(
00179 |         *(source.get("pseudopotential_sha256_by_species") for source in valid_sources),
00180 |         material.get("pseudopotential_sha256_by_species"),
00181 |         material.get("pseudopotential_sha256"),
00182 |     ) or {}
00183 |     split_hash = _first_present(
00184 |         *(source.get("split_manifest_hash") for source in valid_sources),
00185 |         graph2mat.get("split_manifest_hash"),
00186 |         stable_json_hash(graph2mat.get("split_file_sha256"))
00187 |         if isinstance(graph2mat.get("split_file_sha256"), dict) and graph2mat.get("split_file_sha256")
00188 |         else None,
00189 |     )
00190 |     dataset_recipe = _first_present(*(source.get("dataset_recipe") for source in valid_sources))
00191 |     dataset_recipe_parameters = _first_present(
00192 |         *(source.get("dataset_recipe_parameters") for source in valid_sources),
00193 |         _as_mapping(dataset_recipe).get("parameters") if isinstance(dataset_recipe, dict) else None,
00194 |         _as_mapping(dataset_recipe).get("generation_parameters") if isinstance(dataset_recipe, dict) else None,
00195 |     )
00196 | 
00197 |     provenance = {
00198 |         "material_label": _first_present(
00199 |             *(source.get("material_label") for source in valid_sources),
00200 |             material.get("label"),
00201 |         ),
00202 |         "material_bundle_path": _first_present(
00203 |             *(source.get("material_bundle_path") for source in valid_sources),
00204 |             material.get("material_bundle_path"),
00205 |             material.get("material_yaml"),
00206 |             material.get("fdf"),
00207 |         ),
00208 |         "material_source": _first_present(
00209 |             *(source.get("material_source") for source in valid_sources),
00210 |             material.get("material_source"),
00211 |         ),
00212 |         "material_preset": _first_present(
00213 |             *(source.get("material_preset") for source in valid_sources),
00214 |             material.get("preset"),
00215 |         ),
00216 |         "material_structure_type": _first_present(
00217 |             *(source.get("material_structure_type") for source in valid_sources),
00218 |             material.get("structure_type"),
00219 |         ),
00220 |         "material_species": material_species,
00221 |         "material_atom_count": _first_present(
00222 |             *(source.get("material_atom_count") for source in valid_sources),
00223 |             _atom_count_from_material(material),
00224 |         ),
00225 |         "material_cell_summary": _first_present(
00226 |             *(source.get("material_cell_summary") for source in valid_sources),
00227 |             material.get("cell_summary"),
00228 |             material.get("lattice_vectors"),
00229 |         ),
00230 |         "fdf_sha256": _first_present(
00231 |             *(source.get("fdf_sha256") for source in valid_sources),
00232 |             material.get("fdf_sha256"),
00233 |             material.get("base_fdf_sha256"),
00234 |         ),
00235 |         "pseudopotential_sha256_by_species": dict(sorted(_as_mapping(pseudo_hashes).items())),
00236 |         "basis_sha256_by_species": dict(sorted(_as_mapping(basis_hashes).items())),
00237 |         "siesta_settings_hash": _first_present(*(source.get("siesta_settings_hash") for source in valid_sources)),
00238 |         "siesta_output_flags": _first_present(
00239 |             *(source.get("siesta_output_flags") for source in valid_sources),
00240 |             material.get("siesta_output_flags"),
00241 |             material.get("required_output_flags"),
00242 |             *(source.get("required_output_flags") for source in valid_sources),
00243 |         ),
00244 |         "graph2mat_config_hash": _first_present(
00245 |             *(source.get("graph2mat_config_hash") for source in valid_sources),
00246 |             graph2mat.get("config_sha256"),
00247 |         ),
00248 |         "split_manifest_hash": split_hash,
00249 |         "dataset_recipe": dataset_recipe,
00250 |         "dataset_recipe_parameters": dataset_recipe_parameters,
00251 |         "reference_matrix_sha256": _first_present(
00252 |             *(source.get("reference_matrix_sha256") for source in valid_sources),
00253 |             reference.get("sha256"),
00254 |         ),
00255 |         "prediction_matrix_sha256": _first_present(
00256 |             *(source.get("prediction_matrix_sha256") for source in valid_sources),
00257 |             prediction.get("sha256"),
00258 |         ),
00259 |     }
00260 |     identity_payload = {
00261 |         "label": provenance["material_label"],
00262 |         "structure_type": provenance["material_structure_type"],
00263 |         "species": provenance["material_species"],
00264 |         "fdf_sha256": provenance["fdf_sha256"],
00265 |         "pseudopotential_sha256_by_species": provenance["pseudopotential_sha256_by_species"],
00266 |         "basis_sha256_by_species": provenance["basis_sha256_by_species"],
00267 |     }
00268 |     compatibility_payload = {
00269 |         **identity_payload,
00270 |         "siesta_settings_hash": provenance["siesta_settings_hash"],
00271 |         "siesta_output_flags": provenance["siesta_output_flags"],
00272 |     }
00273 |     identity_present = any(value not in (None, "", {}, [], False) for value in identity_payload.values())
00274 |     if identity_present:
00275 |         provenance["material_identity_hash"] = stable_json_hash(identity_payload)
00276 |         provenance["material_compatibility_hash"] = stable_json_hash(compatibility_payload)
00277 |     return {key: value for key, value in provenance.items() if value not in (None, "", {}, [], False)}
00278 | 
00279 | 
00280 | def material_maps_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
00281 |     maps: dict[str, dict[str, Any]] = {
00282 |         key: _as_mapping(manifest.get(key)).copy()
00283 |         for key in MATERIAL_MAP_FIELDS
00284 |     }
00285 |     method_provenance = _as_mapping(manifest.get("method_provenance"))
00286 |     for method, entry in method_provenance.items():
00287 |         if not isinstance(entry, dict):
00288 |             continue
00289 |         flat = flatten_material_provenance(entry.get("material_provenance") or entry)
00290 |         if flat.get("material_label"):
00291 |             maps["material_label_by_method"].setdefault(str(method), flat["material_label"])
00292 |         if flat.get("material_identity_hash"):
00293 |             maps["material_identity_hash_by_method"].setdefault(str(method), flat["material_identity_hash"])
00294 |         if flat.get("material_compatibility_hash"):
00295 |             maps["material_compatibility_hash_by_method"].setdefault(
00296 |                 str(method), flat["material_compatibility_hash"]
00297 |             )
00298 |         if flat.get("fdf_sha256"):
00299 |             maps["fdf_sha256_by_method"].setdefault(str(method), flat["fdf_sha256"])
00300 |         if flat.get("pseudopotential_sha256_by_species"):
00301 |             maps["pseudopotential_sha256_by_method"].setdefault(
00302 |                 str(method), flat["pseudopotential_sha256_by_species"]
00303 |             )
00304 |         if flat.get("basis_sha256_by_species"):
00305 |             maps["basis_sha256_by_method"].setdefault(str(method), flat["basis_sha256_by_species"])
00306 |     return {key: dict(sorted(value.items())) for key, value in maps.items()}
00307 | 
00308 | 
00309 | def material_compatibility_warning(material_maps: dict[str, Any]) -> str:
00310 |     hashes = _as_mapping(material_maps.get("material_compatibility_hash_by_method"))
00311 |     known = {str(method): str(value) for method, value in hashes.items() if value not in (None, "", False)}
00312 |     if len(set(known.values())) <= 1:
00313 |         return ""
00314 |     detail = ", ".join(f"{method}={value[:12]}" for method, value in sorted(known.items()))
00315 |     return (
00316 |         f"{INCOMPATIBLE_MATERIAL_WARNING_CODE}: material compatibility hashes differ across "
00317 |         f"methods ({detail}); do not pool these runs as one benchmark."
00318 |     )
```

## `shared/joint_artifact_contract.py`

SHA-256: `c5df779a666393fb11909f2b775e8bae9786b5c878ec6d972ded63e61baa6270`

```py
00001 | """Validate joint Graph2Mat/DeepH SIESTA benchmark artifacts.
00002 | 
00003 | The contract is intentionally filesystem-only: it never runs SIESTA and never
00004 | repairs a dataset. It answers whether an existing snapshot has the artifacts
00005 | needed to be used as shared SIESTA ground truth by both Graph2Mat and DeepH.
00006 | """
00007 | 
00008 | from __future__ import annotations
00009 | 
00010 | import json
00011 | from dataclasses import asdict, dataclass, field
00012 | from pathlib import Path
00013 | from typing import Any
00014 | 
00015 | 
00016 | CONTRACT_NAME = "joint_graph2mat_deeph_artifact_contract_v1"
00017 | G2M_DEEPH_BENCHMARK_PROFILE = "g2m_deeph_benchmark"
00018 | 
00019 | SYSTEM_LABEL_SUFFIXES = (
00020 |     ".TSHS",
00021 |     ".TSDE",
00022 |     ".HSX",
00023 |     ".STRUCT_OUT",
00024 |     ".XV",
00025 |     ".ORB_INDX",
00026 | )
00027 | FORBIDDEN_REFERENCE_NAMES = {"ML_prediction.HSX"}
00028 | 
00029 | 
00030 | @dataclass(frozen=True)
00031 | class ArtifactRequirement:
00032 |     key: str
00033 |     description: str
00034 |     required: bool = True
00035 |     filenames: tuple[str, ...] = ()
00036 |     system_label_suffix: str | None = None
00037 |     category: str = "snapshot"
00038 | 
00039 | 
00040 | @dataclass
00041 | class SnapshotValidationResult:
00042 |     snapshot_dir: Path
00043 |     contract_name: str = CONTRACT_NAME
00044 |     valid: bool = False
00045 |     repair_required: bool = False
00046 |     system_label: str | None = None
00047 |     missing_required: list[str] = field(default_factory=list)
00048 |     present_artifacts: dict[str, str] = field(default_factory=dict)
00049 |     errors: list[str] = field(default_factory=list)
00050 |     warnings: list[str] = field(default_factory=list)
00051 | 
00052 |     def to_dict(self) -> dict[str, Any]:
00053 |         data = asdict(self)
00054 |         data["snapshot_dir"] = str(self.snapshot_dir)
00055 |         return data
00056 | 
00057 | 
00058 | @dataclass
00059 | class DatasetValidationResult:
00060 |     dataset_root: Path
00061 |     contract_name: str = CONTRACT_NAME
00062 |     valid: bool = False
00063 |     total_snapshots: int = 0
00064 |     valid_snapshots: int = 0
00065 |     invalid_snapshots: int = 0
00066 |     repair_required_snapshots: int = 0
00067 |     basis_present: bool | None = None
00068 |     pseudopotential_provenance_present: bool | None = None
00069 |     material_identity_present: bool | None = None
00070 |     siesta_input_provenance_present: bool | None = None
00071 |     siesta_version_provenance_present: bool | None = None
00072 |     siesta_command_line_provenance_present: bool | None = None
00073 |     siesta_environment_provenance_present: bool | None = None
00074 |     siesta_execution_log_present: bool | None = None
00075 |     errors: list[str] = field(default_factory=list)
00076 |     warnings: list[str] = field(default_factory=list)
00077 |     snapshots: list[SnapshotValidationResult] = field(default_factory=list)
00078 | 
00079 |     def to_dict(self) -> dict[str, Any]:
00080 |         data = asdict(self)
00081 |         data["dataset_root"] = str(self.dataset_root)
00082 |         data["snapshots"] = [result.to_dict() for result in self.snapshots]
00083 |         return data
00084 | 
00085 | 
00086 | def read_system_label_from_fdf(path: Path) -> str | None:
00087 |     if not path.exists():
00088 |         return None
00089 |     for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
00090 |         clean = line.split("#", 1)[0].strip()
00091 |         if not clean:
00092 |             continue
00093 |         parts = clean.split()
00094 |         if len(parts) >= 2 and parts[0].lower() == "systemlabel":
00095 |             return parts[1]
00096 |     return None
00097 | 
00098 | 
00099 | def read_system_label_from_metadata(path: Path) -> str | None:
00100 |     if not path.exists():
00101 |         return None
00102 |     try:
00103 |         metadata = json.loads(path.read_text(encoding="utf-8"))
00104 |     except json.JSONDecodeError:
00105 |         return None
00106 |     if not isinstance(metadata, dict):
00107 |         return None
00108 |     for key in ("system_label", "siesta_system_label", "SystemLabel"):
00109 |         value = metadata.get(key)
00110 |         if isinstance(value, str) and value.strip():
00111 |             return value.strip()
00112 |     return None
00113 | 
00114 | 
00115 | def _system_labels_from_filenames(snapshot_dir: Path) -> set[str]:
00116 |     labels: set[str] = set()
00117 |     if not snapshot_dir.exists():
00118 |         return labels
00119 |     for path in snapshot_dir.iterdir():
00120 |         if not path.is_file() or path.name in FORBIDDEN_REFERENCE_NAMES:
00121 |             continue
00122 |         for suffix in SYSTEM_LABEL_SUFFIXES:
00123 |             if path.name.endswith(suffix) and path.name != suffix:
00124 |                 labels.add(path.name[: -len(suffix)])
00125 |                 break
00126 |     return labels
00127 | 
00128 | 
00129 | def resolve_system_label(snapshot_dir: Path, default: str | None = None) -> tuple[str | None, list[str], list[str]]:
00130 |     """Resolve SystemLabel from RUN.fdf, metadata, filenames or explicit default."""
00131 | 
00132 |     errors: list[str] = []
00133 |     warnings: list[str] = []
00134 |     labels_by_source: dict[str, str] = {}
00135 | 
00136 |     run_fdf_label = read_system_label_from_fdf(snapshot_dir / "RUN.fdf")
00137 |     if run_fdf_label:
00138 |         labels_by_source["RUN.fdf"] = run_fdf_label
00139 | 
00140 |     metadata_label = read_system_label_from_metadata(snapshot_dir / "metadata.json")
00141 |     if metadata_label:
00142 |         labels_by_source["metadata.json"] = metadata_label
00143 | 
00144 |     filename_labels = _system_labels_from_filenames(snapshot_dir)
00145 |     if len(filename_labels) == 1:
00146 |         labels_by_source["filenames"] = next(iter(filename_labels))
00147 |     elif len(filename_labels) > 1:
00148 |         errors.append(f"ambiguous SystemLabel from filenames: {sorted(filename_labels)}")
00149 | 
00150 |     if default:
00151 |         labels_by_source["default"] = default
00152 | 
00153 |     if errors:
00154 |         return None, errors, warnings
00155 | 
00156 |     unique_labels = sorted(set(labels_by_source.values()))
00157 |     if len(unique_labels) > 1:
00158 |         errors.append(f"ambiguous SystemLabel across sources: {labels_by_source}")
00159 |         return None, errors, warnings
00160 |     if not unique_labels:
00161 |         errors.append("could not resolve SystemLabel from RUN.fdf, metadata, filenames or default")
00162 |         return None, errors, warnings
00163 |     return unique_labels[0], errors, warnings
00164 | 
00165 | 
00166 | def snapshot_requirements(
00167 |     system_label: str,
00168 |     *,
00169 |     require_tshs: bool = True,
00170 |     require_tsde: bool = True,
00171 |     require_run_output: bool = True,
00172 | ) -> list[ArtifactRequirement]:
00173 |     requirements = [
00174 |         ArtifactRequirement("run_fdf", "SIESTA input FDF", filenames=("RUN.fdf",)),
00175 |         ArtifactRequirement("metadata", "snapshot metadata", filenames=("metadata.json",)),
00176 |         ArtifactRequirement(
00177 |             "run_output",
00178 |             "SIESTA text output",
00179 |             required=require_run_output,
00180 |             filenames=("RUN.out", "siesta.out"),
00181 |         ),
00182 |         ArtifactRequirement("hsx", "DeepH SIESTA Hamiltonian/overlap input", system_label_suffix=".HSX"),
00183 |         ArtifactRequirement("struct_out", "DeepH SIESTA structure output", system_label_suffix=".STRUCT_OUT"),
00184 |         ArtifactRequirement("xv", "SIESTA XV geometry/velocity output", system_label_suffix=".XV"),
00185 |         ArtifactRequirement("orb_indx", "DeepH SIESTA orbital index file", system_label_suffix=".ORB_INDX"),
00186 |         ArtifactRequirement(
00187 |             "tshs",
00188 |             "Graph2Mat/evaluator transport Hamiltonian",
00189 |             required=require_tshs,
00190 |             system_label_suffix=".TSHS",
00191 |         ),
00192 |         ArtifactRequirement(
00193 |             "tsde",
00194 |             "Graph2Mat transport density/energy artifact",
00195 |             required=require_tsde,
00196 |             system_label_suffix=".TSDE",
00197 |         ),
00198 |     ]
00199 |     # Materialize the label into descriptions only indirectly; matching happens
00200 |     # in find_artifact so tests can inspect requirement metadata.
00201 |     return requirements
00202 | 
00203 | 
00204 | def find_artifact(snapshot_dir: Path, requirement: ArtifactRequirement, system_label: str) -> Path | None:
00205 |     for name in requirement.filenames:
00206 |         candidate = snapshot_dir / name
00207 |         if candidate.exists():
00208 |             return candidate
00209 |     if requirement.system_label_suffix:
00210 |         candidate = snapshot_dir / f"{system_label}{requirement.system_label_suffix}"
00211 |         if candidate.exists():
00212 |             return candidate
00213 |     return None
00214 | 
00215 | 
00216 | def validate_snapshot(
00217 |     snapshot_dir: Path,
00218 |     *,
00219 |     system_label: str | None = None,
00220 |     require_tshs: bool = True,
00221 |     require_tsde: bool = True,
00222 |     require_run_output: bool = True,
00223 | ) -> SnapshotValidationResult:
00224 |     snapshot_dir = Path(snapshot_dir)
00225 |     result = SnapshotValidationResult(snapshot_dir=snapshot_dir)
00226 |     if not snapshot_dir.exists():
00227 |         result.errors.append(f"snapshot directory does not exist: {snapshot_dir}")
00228 |         result.repair_required = True
00229 |         return result
00230 |     if not snapshot_dir.is_dir():
00231 |         result.errors.append(f"snapshot path is not a directory: {snapshot_dir}")
00232 |         result.repair_required = True
00233 |         return result
00234 | 
00235 |     resolved_label, label_errors, label_warnings = resolve_system_label(snapshot_dir, default=system_label)
00236 |     result.errors.extend(label_errors)
00237 |     result.warnings.extend(label_warnings)
00238 |     result.system_label = resolved_label
00239 |     if resolved_label is None:
00240 |         result.repair_required = True
00241 |         return result
00242 | 
00243 |     for forbidden_name in sorted(FORBIDDEN_REFERENCE_NAMES):
00244 |         if (snapshot_dir / forbidden_name).exists():
00245 |             result.warnings.append(f"forbidden prediction artifact present but ignored: {forbidden_name}")
00246 | 
00247 |     for requirement in snapshot_requirements(
00248 |         resolved_label,
00249 |         require_tshs=require_tshs,
00250 |         require_tsde=require_tsde,
00251 |         require_run_output=require_run_output,
00252 |     ):
00253 |         artifact = find_artifact(snapshot_dir, requirement, resolved_label)
00254 |         if artifact is not None:
00255 |             result.present_artifacts[requirement.key] = str(artifact)
00256 |         elif requirement.required:
00257 |             result.missing_required.append(requirement.key)
00258 | 
00259 |     result.valid = not result.errors and not result.missing_required
00260 |     result.repair_required = not result.valid
00261 |     return result
00262 | 
00263 | 
00264 | def discover_snapshot_dirs(dataset_root: Path) -> list[Path]:
00265 |     dataset_root = Path(dataset_root)
00266 |     if (dataset_root / "RUN.fdf").exists() or (dataset_root / "metadata.json").exists():
00267 |         return [dataset_root]
00268 |     if not dataset_root.exists():
00269 |         return []
00270 |     return sorted(
00271 |         path
00272 |         for path in dataset_root.iterdir()
00273 |         if path.is_dir() and ((path / "RUN.fdf").exists() or (path / "metadata.json").exists())
00274 |     )
00275 | 
00276 | 
00277 | def _has_files(path: Path) -> bool:
00278 |     return path.exists() and path.is_dir() and any(child.is_file() for child in path.iterdir())
00279 | 
00280 | 
00281 | def _json_objects(paths: list[Path]) -> list[dict[str, Any]]:
00282 |     payloads: list[dict[str, Any]] = []
00283 |     for path in paths:
00284 |         if not path.exists() or not path.is_file():
00285 |             continue
00286 |         try:
00287 |             payload = json.loads(path.read_text(encoding="utf-8"))
00288 |         except (OSError, json.JSONDecodeError):
00289 |             continue
00290 |         if isinstance(payload, dict):
00291 |             payloads.append(payload)
00292 |     return payloads
00293 | 
00294 | 
00295 | def _non_empty_mapping(payloads: list[dict[str, Any]], *keys: str) -> bool:
00296 |     for payload in payloads:
00297 |         for key in keys:
00298 |             value = payload.get(key)
00299 |             if isinstance(value, dict) and bool(value):
00300 |                 return True
00301 |     return False
00302 | 
00303 | 
00304 | def _non_empty_text(payloads: list[dict[str, Any]], *keys: str) -> bool:
00305 |     for payload in payloads:
00306 |         for key in keys:
00307 |             value = payload.get(key)
00308 |             if isinstance(value, str) and value.strip():
00309 |                 return True
00310 |     return False
00311 | 
00312 | 
00313 | def _non_empty_text_or_sequence(payloads: list[dict[str, Any]], *keys: str) -> bool:
00314 |     for payload in payloads:
00315 |         for key in keys:
00316 |             value = payload.get(key)
00317 |             if isinstance(value, str) and value.strip():
00318 |                 return True
00319 |             if isinstance(value, (list, tuple)) and any(str(item).strip() for item in value):
00320 |                 return True
00321 |     return False
00322 | 
00323 | 
00324 | def _existing_path_value(payloads: list[dict[str, Any]], dataset_root: Path, *keys: str) -> bool:
00325 |     for payload in payloads:
00326 |         for key in keys:
00327 |             value = payload.get(key)
00328 |             if not isinstance(value, str) or not value.strip():
00329 |                 continue
00330 |             path = Path(value)
00331 |             if not path.is_absolute():
00332 |                 path = dataset_root / path
00333 |             if path.exists() and path.is_file():
00334 |                 return True
00335 |     return False
00336 | 
00337 | 
00338 | def _environment_provenance_present(payloads: list[dict[str, Any]]) -> bool:
00339 |     for payload in payloads:
00340 |         environment = payload.get("environment")
00341 |         if not isinstance(environment, dict):
00342 |             continue
00343 |         python_version = environment.get("python_version")
00344 |         platform_value = environment.get("platform")
00345 |         if isinstance(python_version, str) and python_version.strip() and isinstance(platform_value, str) and platform_value.strip():
00346 |             return True
00347 |     return False
00348 | 
00349 | 
00350 | def validate_dataset(
00351 |     dataset_root: Path,
00352 |     *,
00353 |     snapshot_dirs: list[Path] | None = None,
00354 |     system_label: str | None = None,
00355 |     require_tshs: bool = True,
00356 |     require_tsde: bool = True,
00357 |     require_run_output: bool = True,
00358 |     basis_dirs: list[Path] | None = None,
00359 |     pseudopotential_provenance_paths: list[Path] | None = None,
00360 |     material_identity_paths: list[Path] | None = None,
00361 |     siesta_input_paths: list[Path] | None = None,
00362 |     require_basis: bool = False,
00363 |     require_pseudopotential_provenance: bool = False,
00364 |     require_material_identity: bool = False,
00365 |     require_siesta_input_provenance: bool = False,
00366 |     require_siesta_version_provenance: bool = False,
00367 |     require_siesta_command_line_provenance: bool = False,
00368 |     require_siesta_environment_provenance: bool = False,
00369 |     require_siesta_execution_log: bool = False,
00370 |     require_dataset_provenance: bool = False,
00371 |     validation_profile: str | None = None,
00372 | ) -> DatasetValidationResult:
00373 |     dataset_root = Path(dataset_root)
00374 |     if validation_profile == G2M_DEEPH_BENCHMARK_PROFILE or require_dataset_provenance:
00375 |         require_basis = True
00376 |         require_pseudopotential_provenance = True
00377 |         require_material_identity = True
00378 |         require_siesta_input_provenance = True
00379 |         require_siesta_version_provenance = True
00380 |         require_siesta_command_line_provenance = True
00381 |         require_siesta_environment_provenance = True
00382 |         require_siesta_execution_log = True
00383 |     result = DatasetValidationResult(dataset_root=dataset_root)
00384 |     snapshots = snapshot_dirs if snapshot_dirs is not None else discover_snapshot_dirs(dataset_root)
00385 |     result.snapshots = [
00386 |         validate_snapshot(
00387 |             path,
00388 |             system_label=system_label,
00389 |             require_tshs=require_tshs,
00390 |             require_tsde=require_tsde,
00391 |             require_run_output=require_run_output,
00392 |         )
00393 |         for path in snapshots
00394 |     ]
00395 |     result.total_snapshots = len(result.snapshots)
00396 |     result.valid_snapshots = sum(1 for item in result.snapshots if item.valid)
00397 |     result.invalid_snapshots = result.total_snapshots - result.valid_snapshots
00398 |     result.repair_required_snapshots = sum(1 for item in result.snapshots if item.repair_required)
00399 |     if result.total_snapshots == 0:
00400 |         result.errors.append(f"no snapshot directories found under {dataset_root}")
00401 | 
00402 |     default_material_provenance = dataset_root / "material_provenance.json"
00403 |     basis_candidates = [Path(path) for path in (basis_dirs or [])] or [dataset_root / "basis"]
00404 |     pseudo_candidates = [Path(path) for path in (pseudopotential_provenance_paths or [])] or [
00405 |         default_material_provenance
00406 |     ]
00407 |     material_candidates = [Path(path) for path in (material_identity_paths or [])] or [
00408 |         default_material_provenance
00409 |     ]
00410 |     siesta_candidates = [Path(path) for path in (siesta_input_paths or [])] or [
00411 |         dataset_root / "RUN.fdf",
00412 |         default_material_provenance,
00413 |     ]
00414 |     provenance_payloads = _json_objects(
00415 |         sorted({*pseudo_candidates, *material_candidates, *siesta_candidates}, key=lambda path: str(path))
00416 |     )
00417 | 
00418 |     result.basis_present = any(_has_files(path) for path in basis_candidates) or _non_empty_mapping(
00419 |         provenance_payloads,
00420 |         "basis_file_sha256",
00421 |         "basis_hashes",
00422 |     )
00423 |     result.pseudopotential_provenance_present = _non_empty_mapping(
00424 |         provenance_payloads,
00425 |         "pseudopotential_sha256",
00426 |         "pseudopotential_hashes",
00427 |         "pseudopotential_sha256_by_species",
00428 |     )
00429 |     result.material_identity_present = _non_empty_text(
00430 |         provenance_payloads,
00431 |         "label",
00432 |         "material_label",
00433 |         "material_id",
00434 |     )
00435 |     result.siesta_input_provenance_present = any(path.exists() and path.is_file() for path in siesta_candidates) or _non_empty_text(
00436 |         provenance_payloads,
00437 |         "fdf_sha256",
00438 |         "siesta_input_sha256",
00439 |     )
00440 |     result.siesta_version_provenance_present = _non_empty_text(
00441 |         provenance_payloads,
00442 |         "siesta_version",
00443 |     ) or _existing_path_value(
00444 |         provenance_payloads,
00445 |         dataset_root,
00446 |         "siesta_version_source_file",
00447 |     )
00448 |     result.siesta_command_line_provenance_present = _non_empty_text_or_sequence(
00449 |         provenance_payloads,
00450 |         "siesta_command_line",
00451 |     )
00452 |     result.siesta_environment_provenance_present = _environment_provenance_present(provenance_payloads)
00453 |     result.siesta_execution_log_present = _existing_path_value(
00454 |         provenance_payloads,
00455 |         dataset_root,
00456 |         "siesta_stdout_path",
00457 |         "run_out_path",
00458 |     )
00459 | 
00460 |     if require_basis and not result.basis_present:
00461 |         result.errors.append("dataset-level basis provenance or basis file hashes are missing")
00462 |     if require_pseudopotential_provenance and not result.pseudopotential_provenance_present:
00463 |         result.errors.append("dataset-level pseudopotential provenance or hashes are missing")
00464 |     if require_material_identity and not result.material_identity_present:
00465 |         result.errors.append("dataset-level material identity is missing")
00466 |     if require_siesta_input_provenance and not result.siesta_input_provenance_present:
00467 |         result.errors.append("dataset-level SIESTA/FDF input provenance is missing")
00468 |     if require_siesta_version_provenance and not result.siesta_version_provenance_present:
00469 |         result.errors.append("dataset-level SIESTA version provenance is missing")
00470 |     if require_siesta_command_line_provenance and not result.siesta_command_line_provenance_present:
00471 |         result.errors.append("dataset-level SIESTA command-line provenance is missing")
00472 |     if require_siesta_environment_provenance and not result.siesta_environment_provenance_present:
00473 |         result.errors.append("dataset-level execution environment provenance is missing")
00474 |     if require_siesta_execution_log and not result.siesta_execution_log_present:
00475 |         result.errors.append("dataset-level SIESTA execution log provenance is missing")
00476 | 
00477 |     result.valid = not result.errors and result.invalid_snapshots == 0
00478 |     if result.invalid_snapshots:
00479 |         result.warnings.append(f"{result.invalid_snapshots} snapshots are not benchmark-ready")
00480 |     return result
```

## `shared/benchmark_manifest.py`

SHA-256: `1311aa8c867de71aabebd6c2fa1b391d26ab98dd28d01e3d3d46893060b37865`

```py
00001 | """Deterministic manifests for joint Graph2Mat/DeepH benchmark datasets."""
00002 | 
00003 | from __future__ import annotations
00004 | 
00005 | import csv
00006 | import hashlib
00007 | import json
00008 | import re
00009 | from pathlib import Path
00010 | from typing import Any
00011 | 
00012 | from joint_artifact_contract import CONTRACT_NAME, validate_snapshot
00013 | 
00014 | 
00015 | MANIFEST_SCHEMA = "joint_graph2mat_deeph_benchmark_manifest_v1"
00016 | FROZEN_SPLIT_SCHEMA = "joint_graph2mat_deeph_frozen_split_manifest_v1"
00017 | SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
00018 | ARTIFACT_KEYS = {
00019 |     "run_fdf": "run_fdf",
00020 |     "run_output": "run_output",
00021 |     "metadata": "metadata",
00022 |     "hsx": "reference_hsx",
00023 |     "tshs": "reference_tshs",
00024 |     "tsde": "reference_tsde",
00025 |     "struct_out": "struct_out",
00026 |     "xv": "xv",
00027 |     "orb_indx": "orb_indx",
00028 | }
00029 | SIESTA_FLAG_KEYS = (
00030 |     "SaveHS",
00031 |     "Save.HS",
00032 |     "TS.HS.Save",
00033 |     "TS.DE.Save",
00034 |     "XML.Write",
00035 |     "Write.OrbitalIndex",
00036 | )
00037 | SPIN_FLAG_KEYS = (
00038 |     "SpinPolarized",
00039 |     "FixSpin",
00040 |     "NonCollinearSpin",
00041 | )
00042 | ENVIRONMENT_PROVENANCE_KEYS = (
00043 |     "python_version",
00044 |     "platform",
00045 |     "executable",
00046 |     "package_versions",
00047 |     "conda_env_export_path",
00048 |     "pip_freeze_path",
00049 |     "container_image",
00050 |     "container_digest",
00051 | )
00052 | 
00053 | 
00054 | def file_sha256(path: Path) -> str:
00055 |     digest = hashlib.sha256()
00056 |     with path.open("rb") as handle:
00057 |         for chunk in iter(lambda: handle.read(1024 * 1024), b""):
00058 |             digest.update(chunk)
00059 |     return digest.hexdigest()
00060 | 
00061 | 
00062 | def canonical_sha256(payload: Any) -> str:
00063 |     encoded = json.dumps(
00064 |         payload,
00065 |         sort_keys=True,
00066 |         separators=(",", ":"),
00067 |         ensure_ascii=True,
00068 |     ).encode("utf-8")
00069 |     return hashlib.sha256(encoded).hexdigest()
00070 | 
00071 | 
00072 | def read_json(path: Path) -> dict[str, Any]:
00073 |     if not path.exists():
00074 |         return {}
00075 |     payload = json.loads(path.read_text(encoding="utf-8"))
00076 |     return payload if isinstance(payload, dict) else {}
00077 | 
00078 | 
00079 | def write_json(path: Path, payload: dict[str, Any]) -> None:
00080 |     path.parent.mkdir(parents=True, exist_ok=True)
00081 |     path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
00082 | 
00083 | 
00084 | def read_csv_rows(path: Path) -> list[dict[str, str]]:
00085 |     if not path.exists():
00086 |         return []
00087 |     with path.open("r", encoding="utf-8", newline="") as handle:
00088 |         return list(csv.DictReader(handle))
00089 | 
00090 | 
00091 | def _strip_fdf_comment(line: str) -> str:
00092 |     return line.split("#", 1)[0].strip()
00093 | 
00094 | 
00095 | def fdf_directives(path: Path, keys: tuple[str, ...] = SIESTA_FLAG_KEYS) -> dict[str, str]:
00096 |     if not path.exists():
00097 |         return {}
00098 |     wanted = {key.lower(): key for key in keys}
00099 |     found: dict[str, str] = {}
00100 |     for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
00101 |         clean = _strip_fdf_comment(line)
00102 |         if not clean:
00103 |             continue
00104 |         parts = clean.split(None, 1)
00105 |         if not parts:
00106 |             continue
00107 |         canonical = wanted.get(parts[0].lower())
00108 |         if canonical:
00109 |             found[canonical] = parts[1].strip() if len(parts) > 1 else ""
00110 |     return {key: found[key] for key in keys if key in found}
00111 | 
00112 | 
00113 | def _artifact_hashes(paths: dict[str, str]) -> dict[str, str]:
00114 |     hashes: dict[str, str] = {}
00115 |     for key, value in sorted(paths.items()):
00116 |         path = Path(value)
00117 |         if path.exists() and path.is_file():
00118 |             hashes[key] = file_sha256(path)
00119 |     return hashes
00120 | 
00121 | 
00122 | def _relative_or_absolute(path: str, root: Path) -> str:
00123 |     candidate = Path(path)
00124 |     try:
00125 |         return str(candidate.relative_to(root))
00126 |     except ValueError:
00127 |         return str(candidate)
00128 | 
00129 | 
00130 | def _snapshot_artifacts(sample_dir: Path) -> tuple[dict[str, str], dict[str, str], str | None, bool, list[str]]:
00131 |     validation = validate_snapshot(sample_dir)
00132 |     artifact_paths = {
00133 |         output_key: validation.present_artifacts[input_key]
00134 |         for input_key, output_key in ARTIFACT_KEYS.items()
00135 |         if input_key in validation.present_artifacts
00136 |     }
00137 |     return (
00138 |         artifact_paths,
00139 |         _artifact_hashes(artifact_paths),
00140 |         validation.system_label,
00141 |         validation.valid,
00142 |         validation.missing_required + validation.errors,
00143 |     )
00144 | 
00145 | 
00146 | def split_manifest_rows(split_root: Path) -> list[dict[str, str]]:
00147 |     rows: list[dict[str, str]] = []
00148 |     for split in ("train", "validation", "test"):
00149 |         for row in read_csv_rows(split_root / f"{split}_manifest.csv"):
00150 |             merged = dict(row)
00151 |             merged["split"] = split
00152 |             rows.append(merged)
00153 |     return sorted(
00154 |         rows,
00155 |         key=lambda row: (
00156 |             SPLIT_ORDER.get(str(row.get("split") or ""), 99),
00157 |             str(row.get("sample_id") or ""),
00158 |             str(row.get("sample_dir") or ""),
00159 |         ),
00160 |     )
00161 | 
00162 | 
00163 | def build_frozen_split_manifest(dataset_root: Path, split_root: Path) -> dict[str, Any]:
00164 |     dataset_root = Path(dataset_root)
00165 |     split_root = Path(split_root)
00166 |     frozen_rows: list[dict[str, Any]] = []
00167 |     warnings: list[str] = []
00168 |     for row in split_manifest_rows(split_root):
00169 |         sample_dir = Path(row.get("sample_dir") or "")
00170 |         artifact_paths, artifact_hashes, system_label, valid, problems = _snapshot_artifacts(sample_dir)
00171 |         sample_id = str(row.get("sample_id") or sample_dir.name)
00172 |         frozen_row: dict[str, Any] = {
00173 |             "sample_id": sample_id,
00174 |             "graph2mat_sample_id": sample_id,
00175 |             "deeph_sample_id": sample_id,
00176 |             "split": row.get("split"),
00177 |             "sample_dir": str(sample_dir),
00178 |             "system_label": system_label,
00179 |             "valid": valid,
00180 |             "validation_problems": problems,
00181 |             "artifact_paths": artifact_paths,
00182 |             "artifact_sha256": artifact_hashes,
00183 |         }
00184 |         for key, value in row.items():
00185 |             if key not in frozen_row and value not in (None, ""):
00186 |                 frozen_row[key] = value
00187 |         for artifact_key, artifact_path in sorted(artifact_paths.items()):
00188 |             frozen_row[f"{artifact_key}_path"] = artifact_path
00189 |             frozen_row[f"{artifact_key}_sha256"] = artifact_hashes.get(artifact_key, "")
00190 |         if not valid:
00191 |             warnings.append(f"{sample_id}: invalid joint artifacts: {problems}")
00192 |         frozen_rows.append(frozen_row)
00193 | 
00194 |     hash_rows = [
00195 |         {
00196 |             "sample_id": row["sample_id"],
00197 |             "split": row["split"],
00198 |             "artifact_sha256": row["artifact_sha256"],
00199 |         }
00200 |         for row in frozen_rows
00201 |     ]
00202 |     split_hash = canonical_sha256(hash_rows)
00203 |     split_counts = {
00204 |         split: sum(1 for row in frozen_rows if row.get("split") == split)
00205 |         for split in ("train", "validation", "test")
00206 |     }
00207 |     return {
00208 |         "schema": FROZEN_SPLIT_SCHEMA,
00209 |         "artifact_contract_version": CONTRACT_NAME,
00210 |         "dataset_root": str(dataset_root),
00211 |         "split_root": str(split_root),
00212 |         "split_hash": split_hash,
00213 |         "split_counts": split_counts,
00214 |         "valid": not warnings and bool(frozen_rows),
00215 |         "warnings": warnings,
00216 |         "rows": frozen_rows,
00217 |     }
00218 | 
00219 | 
00220 | def _dataset_sample_rows_from_validation(artifact_validation: dict[str, Any]) -> list[dict[str, Any]]:
00221 |     rows: list[dict[str, Any]] = []
00222 |     for snapshot in artifact_validation.get("snapshots") or []:
00223 |         if not isinstance(snapshot, dict):
00224 |             continue
00225 |         artifacts = dict(snapshot.get("present_artifacts") or {})
00226 |         rows.append(
00227 |             {
00228 |                 "sample_dir": snapshot.get("snapshot_dir"),
00229 |                 "system_label": snapshot.get("system_label"),
00230 |                 "valid": bool(snapshot.get("valid")),
00231 |                 "repair_required": bool(snapshot.get("repair_required")),
00232 |                 "missing_required": list(snapshot.get("missing_required") or []),
00233 |                 "errors": list(snapshot.get("errors") or []),
00234 |                 "warnings": list(snapshot.get("warnings") or []),
00235 |                 "artifact_paths": artifacts,
00236 |                 "artifact_sha256": _artifact_hashes(artifacts),
00237 |             }
00238 |         )
00239 |     return sorted(rows, key=lambda row: str(row.get("sample_dir") or ""))
00240 | 
00241 | 
00242 | def _non_empty_mapping(payload: dict[str, Any], *keys: str) -> bool:
00243 |     for key in keys:
00244 |         value = payload.get(key)
00245 |         if isinstance(value, dict) and bool(value):
00246 |             return True
00247 |     return False
00248 | 
00249 | 
00250 | def _non_empty_text(payload: dict[str, Any], *keys: str) -> bool:
00251 |     for key in keys:
00252 |         value = payload.get(key)
00253 |         if isinstance(value, str) and value.strip():
00254 |             return True
00255 |     return False
00256 | 
00257 | 
00258 | # A usable SIESTA version string must contain a dotted numeric version
00259 | # (e.g. "5.4.2-11-g4e9a46060"). Environment noise captured from stderr
00260 | # ("Authorization required, but no authorization protocol specified") has no
00261 | # such token, so it can never pass the provenance gate again.
00262 | _SIESTA_VERSION_TOKEN = re.compile(r"\d+\.\d+")
00263 | # Line emitted by `siesta --version` build info: "Version         : 5.4.2-...".
00264 | _SIESTA_BUILD_INFO_VERSION_LINE = re.compile(r"^\s*Version\s*:\s*(\S+)", re.MULTILINE)
00265 | 
00266 | 
00267 | def looks_like_siesta_version(text: Any) -> bool:
00268 |     """True when ``text`` plausibly names a SIESTA version (contains X.Y)."""
00269 |     return isinstance(text, str) and bool(_SIESTA_VERSION_TOKEN.search(text))
00270 | 
00271 | 
00272 | def extract_siesta_version_from_text(text: Any) -> str | None:
00273 |     """Extract a validated SIESTA version from probe output / build info.
00274 | 
00275 |     Prefers the build-info ``Version : <token>`` line; otherwise the first
00276 |     line that looks like a version. Returns None when nothing validates —
00277 |     callers must NOT fall back to arbitrary first lines (that is how X11
00278 |     noise got recorded as a version).
00279 |     """
00280 |     if not isinstance(text, str) or not text.strip():
00281 |         return None
00282 |     match = _SIESTA_BUILD_INFO_VERSION_LINE.search(text)
00283 |     if match and looks_like_siesta_version(match.group(1)):
00284 |         return match.group(1)
00285 |     for line in text.splitlines():
00286 |         line = line.strip()
00287 |         if line and looks_like_siesta_version(line):
00288 |             return line
00289 |     return None
00290 | 
00291 | 
00292 | def _non_empty_text_or_sequence(payload: dict[str, Any], *keys: str) -> bool:
00293 |     for key in keys:
00294 |         value = payload.get(key)
00295 |         if isinstance(value, str) and value.strip():
00296 |             return True
00297 |         if isinstance(value, (list, tuple)) and any(str(item).strip() for item in value):
00298 |             return True
00299 |     return False
00300 | 
00301 | 
00302 | def _existing_material_path(dataset_root: Path, material: dict[str, Any], *keys: str) -> bool:
00303 |     for key in keys:
00304 |         value = material.get(key)
00305 |         if not isinstance(value, str) or not value.strip():
00306 |             continue
00307 |         path = Path(value)
00308 |         if not path.is_absolute():
00309 |             path = dataset_root / path
00310 |         if path.exists() and path.is_file():
00311 |             return True
00312 |     return False
00313 | 
00314 | 
00315 | def sanitized_environment_provenance(material: dict[str, Any]) -> dict[str, Any]:
00316 |     environment = material.get("environment")
00317 |     if not isinstance(environment, dict):
00318 |         return {}
00319 |     sanitized: dict[str, Any] = {}
00320 |     for key in ENVIRONMENT_PROVENANCE_KEYS:
00321 |         value = environment.get(key)
00322 |         if value in (None, "", {}, []):
00323 |             continue
00324 |         sanitized[key] = value
00325 |     return sanitized
00326 | 
00327 | 
00328 | def _environment_provenance_present(material: dict[str, Any]) -> bool:
00329 |     environment = sanitized_environment_provenance(material)
00330 |     return _non_empty_text(environment, "python_version") and _non_empty_text(environment, "platform")
00331 | 
00332 | 
00333 | def fdf_block_lines(path: Path, block_name: str) -> list[str]:
00334 |     if not path.exists():
00335 |         return []
00336 |     lower_name = block_name.lower()
00337 |     lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
00338 |     inside = False
00339 |     output: list[str] = []
00340 |     for line in lines:
00341 |         clean = _strip_fdf_comment(line)
00342 |         if not clean:
00343 |             continue
00344 |         lower = clean.lower()
00345 |         if lower == f"%block {lower_name}":
00346 |             inside = True
00347 |             continue
00348 |         if inside and lower == f"%endblock {lower_name}":
00349 |             return output
00350 |         if inside:
00351 |             output.append(clean)
00352 |     return output
00353 | 
00354 | 
00355 | def kpoint_summary(path: Path) -> dict[str, Any]:
00356 |     rows = fdf_block_lines(path, "kgrid_Monkhorst_Pack")
00357 |     return {
00358 |         "kgrid_monkhorst_pack": rows,
00359 |         "present": bool(rows),
00360 |     }
00361 | 
00362 | 
00363 | def spin_summary(path: Path) -> dict[str, str]:
00364 |     return fdf_directives(path, keys=SPIN_FLAG_KEYS)
00365 | 
00366 | 
00367 | def provenance_status(
00368 |     dataset_root: Path,
00369 |     material: dict[str, Any],
00370 |     *,
00371 |     strict_paper_ready: bool = False,
00372 | ) -> dict[str, Any]:
00373 |     run_fdf_path = dataset_root / "RUN.fdf"
00374 |     status: dict[str, Any] = {
00375 |         "basis_provenance": _non_empty_mapping(material, "basis_file_sha256", "basis_hashes"),
00376 |         "pseudopotential_provenance": _non_empty_mapping(
00377 |             material,
00378 |             "pseudopotential_sha256",
00379 |             "pseudopotential_hashes",
00380 |             "pseudopotential_sha256_by_species",
00381 |             # Mixed datasets (mixed_dataset_materialize.py) record per-source
00382 |             # hashes here when small/large pseudopotentials differ (e.g. the
00383 |             # small pool carries Ghost-H, the large pool doesn't).
00384 |             "pseudopotential_sha256_by_source",
00385 |         ),
00386 |         "material_identity": _non_empty_text(material, "label", "material_label", "material_id"),
00387 |         "siesta_input_provenance": run_fdf_path.exists() or _non_empty_text(
00388 |             material,
00389 |             "fdf_sha256",
00390 |             "siesta_input_sha256",
00391 |         ),
00392 |         # A non-empty string is NOT enough: archived datasets carried X11
00393 |         # noise as siesta_version. The text must look like a real version.
00394 |         "siesta_version_provenance": looks_like_siesta_version(
00395 |             material.get("siesta_version")
00396 |         ) or _existing_material_path(dataset_root, material, "siesta_version_source_file"),
00397 |         "siesta_command_line_provenance": _non_empty_text_or_sequence(material, "siesta_command_line"),
00398 |         "siesta_environment_provenance": _environment_provenance_present(material),
00399 |         "siesta_execution_log_provenance": _existing_material_path(
00400 |             dataset_root,
00401 |             material,
00402 |             "siesta_stdout_path",
00403 |             "run_out_path",
00404 |         ),
00405 |     }
00406 |     required_keys = [
00407 |         "basis_provenance",
00408 |         "pseudopotential_provenance",
00409 |         "material_identity",
00410 |         "siesta_input_provenance",
00411 |     ]
00412 |     if strict_paper_ready:
00413 |         required_keys.extend(
00414 |             [
00415 |                 "siesta_version_provenance",
00416 |                 "siesta_command_line_provenance",
00417 |                 "siesta_environment_provenance",
00418 |                 "siesta_execution_log_provenance",
00419 |             ]
00420 |         )
00421 |     missing = [key for key in required_keys if not status.get(key)]
00422 |     return {
00423 |         **status,
00424 |         "strict_paper_ready": strict_paper_ready,
00425 |         "valid": not missing,
00426 |         "missing": missing,
00427 |     }
00428 | 
00429 | 
00430 | def build_benchmark_dataset_manifest(
00431 |     dataset_root: Path,
00432 |     *,
00433 |     artifact_validation: dict[str, Any],
00434 |     frozen_split_manifest: dict[str, Any] | None = None,
00435 |     material_provenance: dict[str, Any] | None = None,
00436 |     generation_mode: str = "clean_one_pass",
00437 |     strict_paper_ready_provenance: bool = False,
00438 | ) -> dict[str, Any]:
00439 |     dataset_root = Path(dataset_root)
00440 |     material = material_provenance or {}
00441 |     samples = _dataset_sample_rows_from_validation(artifact_validation)
00442 |     labels = sorted(
00443 |         {str(row.get("system_label")) for row in samples if row.get("system_label")}
00444 |     )
00445 |     system_label = labels[0] if len(labels) == 1 else None
00446 |     warnings = list(artifact_validation.get("warnings") or [])
00447 |     if len(labels) > 1:
00448 |         warnings.append(f"ambiguous dataset SystemLabel values: {labels}")
00449 | 
00450 |     run_fdf_path = dataset_root / "RUN.fdf"
00451 |     provenance = provenance_status(
00452 |         dataset_root,
00453 |         material,
00454 |         strict_paper_ready=strict_paper_ready_provenance,
00455 |     )
00456 |     for missing_key in provenance["missing"]:
00457 |         warnings.append(f"missing dataset-level {missing_key}")
00458 |     split_hash = (frozen_split_manifest or {}).get("split_hash")
00459 |     identity_payload = {
00460 |         "artifact_contract_version": CONTRACT_NAME,
00461 |         "generation_mode": generation_mode,
00462 |         "material_label": material.get("label"),
00463 |         "samples": [
00464 |             {
00465 |                 "sample_dir": row.get("sample_dir"),
00466 |                 "artifact_sha256": row.get("artifact_sha256"),
00467 |             }
00468 |             for row in samples
00469 |         ],
00470 |         "split_hash": split_hash,
00471 |     }
00472 |     benchmark_dataset_id = f"joint_graph2mat_deeph_{canonical_sha256(identity_payload)[:16]}"
00473 |     valid = (
00474 |         bool(artifact_validation.get("valid"))
00475 |         and not any(not row.get("valid") for row in samples)
00476 |         and bool(provenance["valid"])
00477 |     )
00478 |     if frozen_split_manifest is not None:
00479 |         valid = valid and bool(frozen_split_manifest.get("valid"))
00480 |     return {
00481 |         "schema": MANIFEST_SCHEMA,
00482 |         "benchmark_dataset_id": benchmark_dataset_id,
00483 |         "dataset_root": str(dataset_root),
00484 |         "artifact_contract_version": CONTRACT_NAME,
00485 |         "generation_mode": generation_mode,
00486 |         "validation_status": "valid" if valid else "invalid",
00487 |         "benchmark_ready": valid,
00488 |         "warnings": warnings,
00489 |         "material_label": material.get("label"),
00490 |         "material_source": material,
00491 |         "system_label": system_label,
00492 |         "siesta_input_path": str(run_fdf_path) if run_fdf_path.exists() else "",
00493 |         "siesta_input_sha256": file_sha256(run_fdf_path) if run_fdf_path.exists() else "",
00494 |         "siesta_flags": fdf_directives(run_fdf_path),
00495 |         "siesta_version": material.get("siesta_version", ""),
00496 |         "siesta_version_source_file": material.get("siesta_version_source_file", ""),
00497 |         "siesta_executable": material.get("siesta_executable", ""),
00498 |         "siesta_command_line": material.get("siesta_command_line", ""),
00499 |         "siesta_stdout_path": material.get("siesta_stdout_path") or material.get("run_out_path") or "",
00500 |         "siesta_returncode": material.get("siesta_returncode"),
00501 |         "siesta_build_info": material.get("siesta_build_info", ""),
00502 |         "kpoint_summary": kpoint_summary(run_fdf_path),
00503 |         "spin_summary": spin_summary(run_fdf_path),
00504 |         "environment": sanitized_environment_provenance(material),
00505 |         "graph2mat_commit": material.get("graph2mat_commit", ""),
00506 |         "deeph_pack_commit": material.get("deeph_pack_commit", ""),
00507 |         "basis_hashes": material.get("basis_file_sha256") or {},
00508 |         "pseudopotential_hashes": material.get("pseudopotential_sha256") or {},
00509 |         "provenance_status": provenance,
00510 |         "artifact_validation": artifact_validation,
00511 |         "samples": samples,
00512 |         "frozen_split_manifest": {
00513 |             "path": str(dataset_root / "frozen_split_manifest.json"),
00514 |             "split_hash": split_hash,
00515 |             "split_counts": (frozen_split_manifest or {}).get("split_counts", {}),
00516 |             "valid": (frozen_split_manifest or {}).get("valid"),
00517 |         },
00518 |     }
00519 | 
00520 | 
00521 | def write_benchmark_manifests(
00522 |     *,
00523 |     dataset_root: Path,
00524 |     split_root: Path,
00525 |     generation_mode: str = "clean_one_pass",
00526 |     artifact_validation_path: Path | None = None,
00527 |     material_provenance_path: Path | None = None,
00528 |     strict_paper_ready_provenance: bool = False,
00529 | ) -> tuple[dict[str, Any], dict[str, Any]]:
00530 |     dataset_root = Path(dataset_root)
00531 |     artifact_validation = read_json(artifact_validation_path or dataset_root / "artifact_validation.json")
00532 |     material_provenance = read_json(material_provenance_path or dataset_root / "material_provenance.json")
00533 |     frozen_split = build_frozen_split_manifest(dataset_root, split_root)
00534 |     dataset_manifest = build_benchmark_dataset_manifest(
00535 |         dataset_root,
00536 |         artifact_validation=artifact_validation,
00537 |         frozen_split_manifest=frozen_split,
00538 |         material_provenance=material_provenance,
00539 |         generation_mode=generation_mode,
00540 |         strict_paper_ready_provenance=strict_paper_ready_provenance,
00541 |     )
00542 |     write_json(dataset_root / "frozen_split_manifest.json", frozen_split)
00543 |     write_json(dataset_root / "benchmark_dataset_manifest.json", dataset_manifest)
00544 |     if not dataset_manifest["benchmark_ready"]:
00545 |         raise RuntimeError(
00546 |             "Benchmark dataset manifest is not valid; refusing to freeze dataset for training. "
00547 |             f"See {dataset_root / 'benchmark_dataset_manifest.json'}"
00548 |         )
00549 |     return dataset_manifest, frozen_split
```

## `shared/artifact_signature.py`

SHA-256: `a844ab3f4713c7f8ba3ca279b3d4ef26b3a68e2c35f679590b5761040e5cf9c2`

```py
00001 | """Deterministic input signatures for cached derivative artifacts (audit Fase 5).
00002 | 
00003 | A cached ``.npz`` may only be reused when its sidecar metadata carries an
00004 | ``input_signature_sha256`` that matches the signature recomputed from the
00005 | CURRENT inputs (checkpoint, code, structure, direction, dtype, method).
00006 | Anything else — no metadata, no signature (legacy), mismatch, unreadable or
00007 | non-finite payload — must be recomputed.
00008 | """
00009 | 
00010 | from __future__ import annotations
00011 | 
00012 | import hashlib
00013 | import json
00014 | from pathlib import Path
00015 | from typing import Any
00016 | 
00017 | INPUT_SIGNATURE_SCHEMA = "derivative_input_signature_v1"
00018 | 
00019 | CACHE_VALID = "valid"
00020 | CACHE_LEGACY_UNVERIFIED = "legacy_unverified"
00021 | CACHE_SIGNATURE_MISMATCH = "signature_mismatch"
00022 | CACHE_MISSING_METADATA = "missing_metadata"
00023 | CACHE_UNREADABLE = "unreadable"
00024 | CACHE_NON_FINITE = "non_finite"
00025 | 
00026 | 
00027 | def file_sha256(path: str | Path | None) -> str | None:
00028 |     if path in (None, ""):
00029 |         return None
00030 |     path = Path(path)
00031 |     if not path.is_file():
00032 |         return None
00033 |     digest = hashlib.sha256()
00034 |     with path.open("rb") as handle:
00035 |         for chunk in iter(lambda: handle.read(1 << 20), b""):
00036 |             digest.update(chunk)
00037 |     return digest.hexdigest()
00038 | 
00039 | 
00040 | def input_signature_sha256(payload: dict[str, Any]) -> str:
00041 |     """Canonical sha256 over the signature payload (order-independent)."""
00042 |     encoded = json.dumps(
00043 |         {"schema": INPUT_SIGNATURE_SCHEMA, **payload},
00044 |         sort_keys=True,
00045 |         ensure_ascii=True,
00046 |         default=str,
00047 |     ).encode("utf-8")
00048 |     return hashlib.sha256(encoded).hexdigest()
00049 | 
00050 | 
00051 | def cached_result_status(
00052 |     npz_path: str | Path,
00053 |     metadata_path: str | Path,
00054 |     expected_signature: str,
00055 | ) -> str:
00056 |     """Classify an existing cached derivative for reuse (never raises)."""
00057 |     npz_path = Path(npz_path)
00058 |     metadata_path = Path(metadata_path)
00059 |     if not metadata_path.is_file():
00060 |         return CACHE_MISSING_METADATA
00061 |     try:
00062 |         metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
00063 |     except (ValueError, OSError):
00064 |         return CACHE_UNREADABLE
00065 |     stored = metadata.get("input_signature_sha256")
00066 |     if not stored:
00067 |         return CACHE_LEGACY_UNVERIFIED
00068 |     if str(stored) != str(expected_signature):
00069 |         return CACHE_SIGNATURE_MISMATCH
00070 |     try:
00071 |         import numpy as np
00072 |         from scipy import sparse
00073 | 
00074 |         matrix = sparse.load_npz(npz_path)
00075 |         if matrix.data.size and not bool(np.all(np.isfinite(matrix.data))):
00076 |             return CACHE_NON_FINITE
00077 |         expected_shape = metadata.get("matrix_shape")
00078 |         if expected_shape and [int(x) for x in expected_shape] != [int(x) for x in matrix.shape]:
00079 |             return CACHE_SIGNATURE_MISMATCH
00080 |     except Exception:  # noqa: BLE001 - unreadable/corrupt payloads must not be reused
00081 |         return CACHE_UNREADABLE
00082 |     return CACHE_VALID
```

## `shared/run_inventory.py`

SHA-256: `1ab7c7a542623e9232aa153eb403e8f9f44f28ad570e242b63b0dd075b1acc84`

```py
00001 | """Reproducible run inventory: which code (repo SHAs + imported checkouts) ran.
00002 | 
00003 | Single source of truth for the three-repo provenance block required by every
00004 | scientific manifest (training, derivatives, mixing, metrics, UI payloads).
00005 | 
00006 | ``reproducibility_status`` semantics:
00007 | 
00008 | - ``pinned_clean``: every repository has a resolvable SHA and a clean tree.
00009 | - ``pinned_dirty``: SHAs resolve but at least one tree has local changes.
00010 | - ``unpinned``: at least one repository path exists but has no resolvable SHA.
00011 | - ``unavailable``: at least one repository could not be inspected at all.
00012 | 
00013 | Only ``pinned_clean`` may aspire to ``paper_ready``; everything else is
00014 | diagnostic and must surface a visible warning downstream.
00015 | """
00016 | 
00017 | from __future__ import annotations
00018 | 
00019 | import json
00020 | import subprocess
00021 | import sys
00022 | from pathlib import Path
00023 | from typing import Any
00024 | 
00025 | REPO_ROOT = Path(__file__).resolve().parents[1]
00026 | DEFAULT_REPOSITORIES = {
00027 |     "MD_vs_AtomicDisplacement": REPO_ROOT,
00028 |     "graph2mat": REPO_ROOT.parent / "graph2mat",
00029 |     "DeepH-pack": REPO_ROOT.parent / "DeepH-pack",
00030 | }
00031 | 
00032 | RUN_INVENTORY_SCHEMA = "run_inventory_v1"
00033 | 
00034 | 
00035 | def _git_output(path: Path, args: list[str], timeout: float = 10.0) -> str | None:
00036 |     try:
00037 |         completed = subprocess.run(
00038 |             ["git", "-C", str(path), *args],
00039 |             check=False,
00040 |             capture_output=True,
00041 |             text=True,
00042 |             timeout=timeout,
00043 |         )
00044 |     except Exception:
00045 |         return None
00046 |     if completed.returncode != 0:
00047 |         return None
00048 |     return completed.stdout.strip()
00049 | 
00050 | 
00051 | def git_repository_state(path: str | Path | None) -> dict[str, Any]:
00052 |     """Commit/branch/dirty for one repository path (never raises)."""
00053 |     if path in (None, ""):
00054 |         return {"path": None, "commit": None, "branch": None, "dirty": None, "error": "no_path"}
00055 |     candidate = Path(path).expanduser()
00056 |     if not candidate.exists():
00057 |         return {
00058 |             "path": str(candidate),
00059 |             "commit": None,
00060 |             "branch": None,
00061 |             "dirty": None,
00062 |             "error": "path_missing",
00063 |         }
00064 |     root_text = _git_output(candidate, ["rev-parse", "--show-toplevel"])
00065 |     if not root_text:
00066 |         return {
00067 |             "path": str(candidate),
00068 |             "commit": None,
00069 |             "branch": None,
00070 |             "dirty": None,
00071 |             "error": "not_git_repo",
00072 |         }
00073 |     root = Path(root_text)
00074 |     commit = _git_output(root, ["rev-parse", "HEAD"])
00075 |     dirty_text = _git_output(root, ["status", "--porcelain"], timeout=30.0)
00076 |     return {
00077 |         "path": str(root),
00078 |         "commit": commit,
00079 |         "branch": _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
00080 |         "dirty": bool(dirty_text) if dirty_text is not None else None,
00081 |     }
00082 | 
00083 | 
00084 | def module_import_state(module_name: str, python_executable: str | Path | None = None) -> dict[str, Any]:
00085 |     """Where ``module_name`` actually imports from (this or another python)."""
00086 |     if python_executable in (None, "", str(sys.executable)):
00087 |         import importlib
00088 | 
00089 |         try:
00090 |             module = importlib.import_module(module_name)
00091 |             module_path = str(Path(getattr(module, "__file__", "") or "").resolve())
00092 |             return {"module": module_name, "module_path": module_path or None}
00093 |         except Exception as exc:  # noqa: BLE001 - inventory must never crash the run
00094 |             return {"module": module_name, "module_path": None, "error": repr(exc)}
00095 |     script = (
00096 |         f"import importlib, pathlib; "
00097 |         f"m = importlib.import_module({module_name!r}); "
00098 |         f"print(pathlib.Path(m.__file__).resolve())"
00099 |     )
00100 |     try:
00101 |         completed = subprocess.run(
00102 |             [str(python_executable), "-c", script],
00103 |             check=False,
00104 |             capture_output=True,
00105 |             text=True,
00106 |             timeout=60.0,
00107 |         )
00108 |     except Exception as exc:  # noqa: BLE001
00109 |         return {"module": module_name, "module_path": None, "error": repr(exc)}
00110 |     if completed.returncode != 0:
00111 |         return {
00112 |             "module": module_name,
00113 |             "module_path": None,
00114 |             "error": completed.stderr.strip() or "import_failed",
00115 |         }
00116 |     return {"module": module_name, "module_path": completed.stdout.strip()}
00117 | 
00118 | 
00119 | def _import_matches_repo(import_state: dict[str, Any], repo_state: dict[str, Any]) -> bool | None:
00120 |     module_path = import_state.get("module_path")
00121 |     repo_path = repo_state.get("path")
00122 |     if not module_path or not repo_path:
00123 |         return None
00124 |     return str(module_path).startswith(str(repo_path).rstrip("/") + "/")
00125 | 
00126 | 
00127 | def reproducibility_status(repositories: dict[str, dict[str, Any]]) -> str:
00128 |     statuses = list(repositories.values())
00129 |     if not statuses:
00130 |         return "unavailable"
00131 |     if any(s.get("error") in ("path_missing", "no_path") for s in statuses):
00132 |         return "unavailable"
00133 |     if any(not s.get("commit") for s in statuses):
00134 |         return "unpinned"
00135 |     if any(s.get("dirty") for s in statuses):
00136 |         return "pinned_dirty"
00137 |     if any(s.get("dirty") is None for s in statuses):
00138 |         return "unpinned"
00139 |     return "pinned_clean"
00140 | 
00141 | 
00142 | def collect_run_inventory(
00143 |     repositories: dict[str, str | Path] | None = None,
00144 |     *,
00145 |     deeph_python: str | Path | None = None,
00146 |     graph2mat_python: str | Path | None = None,
00147 | ) -> dict[str, Any]:
00148 |     """Full run inventory (repos + python + real import locations).
00149 | 
00150 |     ``deeph_python`` / ``graph2mat_python`` point at the interpreters that
00151 |     actually run each backend when they differ from ``sys.executable``.
00152 |     """
00153 |     repo_paths = {k: Path(v) for k, v in (repositories or DEFAULT_REPOSITORIES).items()}
00154 |     repo_states = {name: git_repository_state(path) for name, path in repo_paths.items()}
00155 | 
00156 |     try:
00157 |         import torch
00158 | 
00159 |         torch_version = torch.__version__
00160 |         default_dtype = str(torch.get_default_dtype()).replace("torch.", "")
00161 |     except Exception:  # noqa: BLE001 - torch-less callers still get an inventory
00162 |         torch_version = None
00163 |         default_dtype = None
00164 | 
00165 |     imports: dict[str, Any] = {}
00166 |     if "graph2mat" in repo_states:
00167 |         state = module_import_state("graph2mat", graph2mat_python)
00168 |         state["matches_inspected_repo"] = _import_matches_repo(state, repo_states["graph2mat"])
00169 |         imports["graph2mat"] = state
00170 |     if "DeepH-pack" in repo_states:
00171 |         state = module_import_state("deeph", deeph_python)
00172 |         state["matches_inspected_repo"] = _import_matches_repo(state, repo_states["DeepH-pack"])
00173 |         imports["deeph"] = state
00174 | 
00175 |     inventory = {
00176 |         "schema": RUN_INVENTORY_SCHEMA,
00177 |         "repositories": repo_states,
00178 |         "python": {
00179 |             "executable": sys.executable,
00180 |             "version": sys.version.split()[0],
00181 |             "torch_version": torch_version,
00182 |             "default_dtype": default_dtype,
00183 |         },
00184 |         "imports": imports,
00185 |         "reproducibility_status": reproducibility_status(repo_states),
00186 |     }
00187 |     mismatches = [
00188 |         name for name, state in imports.items() if state.get("matches_inspected_repo") is False
00189 |     ]
00190 |     if mismatches:
00191 |         inventory["warnings"] = [
00192 |             f"module '{name}' imports from outside the inspected repository "
00193 |             f"({imports[name].get('module_path')}); the inspected SHA does not "
00194 |             "describe the executed code"
00195 |             for name in mismatches
00196 |         ]
00197 |     return inventory
00198 | 
00199 | 
00200 | def main() -> int:
00201 |     print(json.dumps(collect_run_inventory(), indent=2))
00202 |     return 0
00203 | 
00204 | 
00205 | if __name__ == "__main__":
00206 |     raise SystemExit(main())
```

## `tests/test_g2m_deeph_test_blindness.py`

SHA-256: `67087b8de08986c8ec8b41ad702dc4ab4ce3ce858a4c15834864fe82f44c837f`

```py
00001 | import sys
00002 | import tempfile
00003 | import unittest
00004 | from pathlib import Path
00005 | 
00006 | 
00007 | REPO_ROOT = Path(__file__).resolve().parents[1]
00008 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00009 | if str(SCRIPTS_DIR) not in sys.path:
00010 |     sys.path.insert(0, str(SCRIPTS_DIR))
00011 | 
00012 | from g2m_deeph_test_blindness import (  # noqa: E402
00013 |     FINAL_TEST_STAGE,
00014 |     SEARCH_STAGE,
00015 |     assert_no_test_metrics_for_search,
00016 |     build_final_test_stage_manifest,
00017 |     build_search_stage_manifest,
00018 |     is_final_benchmark_mode,
00019 |     search_stage_record_fields,
00020 |     select_top_k_validation_only,
00021 |     validate_final_evaluation_inputs,
00022 | )
00023 | 
00024 | 
00025 | class Graph2MatDeepHTestBlindnessTests(unittest.TestCase):
00026 |     def test_final_benchmark_mode_detection(self) -> None:
00027 |         self.assertTrue(is_final_benchmark_mode({"benchmark_mode": "final_publication"}))
00028 |         self.assertTrue(is_final_benchmark_mode({"paper_ready": True}))
00029 |         self.assertTrue(
00030 |             is_final_benchmark_mode(
00031 |                 {
00032 |                     "protocol": {
00033 |                         "final_test_policy": {
00034 |                             "policy": "locked_until_final",
00035 |                             "locked_during_search": True,
00036 |                         }
00037 |                     }
00038 |                 }
00039 |             )
00040 |         )
00041 |         self.assertFalse(is_final_benchmark_mode({"benchmark_mode": "exploratory"}))
00042 | 
00043 |     def test_search_artifacts_do_not_require_test_metrics(self) -> None:
00044 |         validate_final_evaluation_inputs(
00045 |             selected_runs=[],
00046 |             metric_rows=[],
00047 |             stage=SEARCH_STAGE,
00048 |             metric="low_energy_rmse_eV",
00049 |         )
00050 | 
00051 |     def test_search_rejects_test_metric_rows(self) -> None:
00052 |         with self.assertRaisesRegex(RuntimeError, "Test metrics are locked"):
00053 |             assert_no_test_metrics_for_search(
00054 |                 [{"model": "graph2mat", "config_id": "g2m_a", "metric_split": "test"}],
00055 |                 stage=SEARCH_STAGE,
00056 |             )
00057 | 
00058 |     def test_top_k_selection_uses_validation_only(self) -> None:
00059 |         rows = [
00060 |             {
00061 |                 "model": "graph2mat",
00062 |                 "config_id": "g2m_a",
00063 |                 "metric_split": "validation",
00064 |                 "low_energy_rmse_eV_mean": 0.2,
00065 |             },
00066 |             {
00067 |                 "model": "graph2mat",
00068 |                 "config_id": "g2m_b",
00069 |                 "metric_split": "validation",
00070 |                 "low_energy_rmse_eV_mean": 0.1,
00071 |             },
00072 |             {
00073 |                 "model": "deeph",
00074 |                 "config_id": "dh_a",
00075 |                 "metric_split": "validation",
00076 |                 "low_energy_rmse_eV_mean": 0.3,
00077 |             },
00078 |         ]
00079 | 
00080 |         selected = select_top_k_validation_only(
00081 |             rows,
00082 |             metric="low_energy_rmse_eV",
00083 |             mode="min",
00084 |             k_per_model=1,
00085 |         )
00086 | 
00087 |         self.assertEqual(
00088 |             {(row["model"], row["config_id"]) for row in selected},
00089 |             {("graph2mat", "g2m_b"), ("deeph", "dh_a")},
00090 |         )
00091 | 
00092 |     def test_top_k_rejects_test_metrics_even_when_validation_exists(self) -> None:
00093 |         rows = [
00094 |             {
00095 |                 "model": "graph2mat",
00096 |                 "config_id": "g2m_a",
00097 |                 "metric_split": "validation",
00098 |                 "low_energy_rmse_eV_mean": 0.2,
00099 |             },
00100 |             {
00101 |                 "model": "graph2mat",
00102 |                 "config_id": "g2m_a",
00103 |                 "metric_split": "test",
00104 |                 "low_energy_rmse_eV_mean": 0.05,
00105 |             },
00106 |         ]
00107 | 
00108 |         with self.assertRaisesRegex(RuntimeError, "Test metrics are locked"):
00109 |             select_top_k_validation_only(
00110 |                 rows,
00111 |                 metric="low_energy_rmse_eV",
00112 |                 mode="min",
00113 |                 k_per_model=1,
00114 |             )
00115 | 
00116 |     def test_final_evaluation_requires_selected_runs_and_test_metrics(self) -> None:
00117 |         with self.assertRaisesRegex(RuntimeError, "selected final runs"):
00118 |             validate_final_evaluation_inputs(
00119 |                 selected_runs=[],
00120 |                 metric_rows=[],
00121 |                 stage=FINAL_TEST_STAGE,
00122 |                 metric="low_energy_rmse_eV",
00123 |             )
00124 | 
00125 |         selected = [{"model": "graph2mat", "config_id": "g2m_a"}]
00126 |         with self.assertRaisesRegex(RuntimeError, "requires test metrics"):
00127 |             validate_final_evaluation_inputs(
00128 |                 selected_runs=selected,
00129 |                 metric_rows=[],
00130 |                 stage=FINAL_TEST_STAGE,
00131 |                 metric="low_energy_rmse_eV",
00132 |             )
00133 | 
00134 |         with self.assertRaisesRegex(RuntimeError, "missing test metrics"):
00135 |             validate_final_evaluation_inputs(
00136 |                 selected_runs=selected,
00137 |                 metric_rows=[
00138 |                     {
00139 |                         "model": "deeph",
00140 |                         "config_id": "dh_a",
00141 |                         "metric_split": "test",
00142 |                         "low_energy_rmse_eV": 0.1,
00143 |                     }
00144 |                 ],
00145 |                 stage=FINAL_TEST_STAGE,
00146 |                 metric="low_energy_rmse_eV",
00147 |             )
00148 | 
00149 |         validate_final_evaluation_inputs(
00150 |             selected_runs=selected,
00151 |             metric_rows=[
00152 |                 {
00153 |                     "model": "graph2mat",
00154 |                     "config_id": "g2m_a",
00155 |                     "metric_split": "test",
00156 |                     "low_energy_rmse_eV": 0.1,
00157 |                 }
00158 |             ],
00159 |             stage=FINAL_TEST_STAGE,
00160 |             metric="low_energy_rmse_eV",
00161 |         )
00162 | 
00163 |     def test_search_stage_manifest_marks_final_test_locked(self) -> None:
00164 |         with tempfile.TemporaryDirectory() as tmp:
00165 |             manifest = build_search_stage_manifest(
00166 |                 run_root=Path(tmp),
00167 |                 summary={"status": "completed", "runs": [{"status": "completed"}], "failed_runs": []},
00168 |                 payload={"benchmark_mode": "final_publication"},
00169 |             )
00170 | 
00171 |             self.assertEqual(manifest["protocol_stage"], SEARCH_STAGE)
00172 |             self.assertTrue(manifest["final_test_locked"])
00173 |             self.assertEqual(manifest["final_test_status"], "pending_selection")
00174 |             self.assertTrue(Path(manifest["path"]).exists())
00175 | 
00176 |     def test_final_test_manifest_requires_selected_test_metrics(self) -> None:
00177 |         with tempfile.TemporaryDirectory() as tmp:
00178 |             manifest = build_final_test_stage_manifest(
00179 |                 run_root=Path(tmp),
00180 |                 selected_runs=[{"model": "graph2mat", "config_id": "g2m_a"}],
00181 |                 metric_rows=[
00182 |                     {
00183 |                         "model": "graph2mat",
00184 |                         "config_id": "g2m_a",
00185 |                         "metric_split": "test",
00186 |                         "low_energy_rmse_eV_mean": 0.1,
00187 |                     }
00188 |                 ],
00189 |                 metric="low_energy_rmse_eV",
00190 |             )
00191 | 
00192 |             self.assertEqual(manifest["protocol_stage"], FINAL_TEST_STAGE)
00193 |             self.assertFalse(manifest["final_test_locked"])
00194 |             self.assertEqual(manifest["final_test_metric_rows"], 1)
00195 |             self.assertTrue(Path(manifest["path"]).exists())
00196 | 
00197 |     def test_search_stage_record_fields_are_explicit(self) -> None:
00198 |         fields = search_stage_record_fields()
00199 | 
00200 |         self.assertEqual(fields["protocol_stage"], SEARCH_STAGE)
00201 |         self.assertTrue(fields["test_metrics_locked"])
00202 |         self.assertEqual(fields["test_metrics_status"], "locked_until_final")
00203 | 
00204 | 
00205 | if __name__ == "__main__":
00206 |     unittest.main()
```

## `tests/test_deeph_split_audit.py`

SHA-256: `924bd090962519ec2b8de4d9e604fa8471a3b415dbc2109052f39937525d8e16`

```py
00001 | import json
00002 | import random
00003 | import sys
00004 | import tempfile
00005 | import unittest
00006 | from contextlib import contextmanager
00007 | from pathlib import Path
00008 | 
00009 | 
00010 | REPO_ROOT = Path(__file__).resolve().parents[1]
00011 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00012 | for path in (SCRIPTS_DIR,):
00013 |     if str(path) not in sys.path:
00014 |         sys.path.insert(0, str(path))
00015 | 
00016 | from deeph_config import build_deeph_raw_mirror, render_train_config  # noqa: E402
00017 | from deeph_split_audit import (  # noqa: E402
00018 |     STATUS_INCOMPATIBLE,
00019 |     STATUS_UNVERIFIED,
00020 |     STATUS_VALID,
00021 |     audit_deeph_split,
00022 | )
00023 | 
00024 | 
00025 | class _FakeNumpyRandom:
00026 |     def __init__(self) -> None:
00027 |         self._rng = random.Random(0)
00028 | 
00029 |     def seed(self, seed: int) -> None:
00030 |         self._rng = random.Random(int(seed))
00031 | 
00032 |     def shuffle(self, values: list[int]) -> None:
00033 |         self._rng.shuffle(values)
00034 | 
00035 | 
00036 | class _FakeNumpy:
00037 |     def __init__(self) -> None:
00038 |         self.random = _FakeNumpyRandom()
00039 | 
00040 | 
00041 | @contextmanager
00042 | def fake_numpy():
00043 |     previous = sys.modules.get("numpy")
00044 |     sys.modules["numpy"] = _FakeNumpy()  # type: ignore[assignment]
00045 |     try:
00046 |         yield
00047 |     finally:
00048 |         if previous is None:
00049 |             sys.modules.pop("numpy", None)
00050 |         else:
00051 |             sys.modules["numpy"] = previous
00052 | 
00053 | 
00054 | def write_snapshot(path: Path) -> None:
00055 |     path.mkdir(parents=True, exist_ok=True)
00056 |     (path / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")
00057 |     (path / "RUN.out").write_text("Job completed\n", encoding="utf-8")
00058 |     (path / "metadata.json").write_text(json.dumps({"system_label": "graphene"}) + "\n", encoding="utf-8")
00059 |     for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
00060 |         (path / f"graphene{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
00061 | 
00062 | 
00063 | def frozen_split_for(root: Path, splits: list[str]) -> dict:
00064 |     rows = []
00065 |     for index, split in enumerate(splits):
00066 |         sample_dir = root / "source" / str(index)
00067 |         write_snapshot(sample_dir)
00068 |         rows.append(
00069 |             {
00070 |                 "sample_id": f"sample{index}",
00071 |                 "split": split,
00072 |                 "sample_dir": str(sample_dir),
00073 |             }
00074 |         )
00075 |     return {
00076 |         "valid": True,
00077 |         "split_hash": "unit-split",
00078 |         "split_counts": {
00079 |             "train": sum(1 for split in splits if split == "train"),
00080 |             "validation": sum(1 for split in splits if split == "validation"),
00081 |             "test": sum(1 for split in splits if split == "test"),
00082 |         },
00083 |         "rows": rows,
00084 |     }
00085 | 
00086 | 
00087 | def write_train_config(root: Path, raw_mirror: dict, *, seed: int) -> Path:
00088 |     path = root / "config" / "train.ini"
00089 |     render_train_config(
00090 |         path,
00091 |         processed_dir=root / "processed",
00092 |         graph_dir=root / "graph",
00093 |         save_dir=root / "train",
00094 |         dataset_name="graphene_unit",
00095 |         split_ratios=raw_mirror["split_ratios"],
00096 |         seed=seed,
00097 |         epochs=1,
00098 |         batch_size=1,
00099 |         learning_rate=0.001,
00100 |         disable_cuda=True,
00101 |         device="cpu",
00102 |     )
00103 |     return path
00104 | 
00105 | 
00106 | def write_processed_from_mirror(raw_mirror: dict, processed_dir: Path) -> None:
00107 |     raw_root = Path(str(raw_mirror["raw_dir"])).resolve(strict=False)
00108 |     for row in raw_mirror["rows"]:
00109 |         raw_dir = Path(str(row["raw_dir"])).resolve(strict=False)
00110 |         relative = raw_dir.relative_to(raw_root)
00111 |         target = processed_dir / relative
00112 |         target.mkdir(parents=True, exist_ok=True)
00113 |         (target / "rc.h5").write_text("rc\n", encoding="utf-8")
00114 | 
00115 | 
00116 | class DeepHSplitAuditTests(unittest.TestCase):
00117 |     def setUp(self) -> None:
00118 |         self.tmp = tempfile.TemporaryDirectory()
00119 |         self.root = Path(self.tmp.name)
00120 | 
00121 |     def tearDown(self) -> None:
00122 |         self.tmp.cleanup()
00123 | 
00124 |     def build_case(self, *, seed: int = 123) -> tuple[dict, dict, Path]:
00125 |         frozen = frozen_split_for(self.root, ["train", "validation", "test"])
00126 |         raw_mirror = build_deeph_raw_mirror(
00127 |             frozen,
00128 |             raw_dir=self.root / "deeph" / "raw",
00129 |             workspace_root=self.root / "deeph",
00130 |             seed=seed,
00131 |         )
00132 |         train_config = write_train_config(self.root / "deeph", raw_mirror, seed=seed)
00133 |         write_processed_from_mirror(raw_mirror, self.root / "deeph" / "processed")
00134 |         return frozen, raw_mirror, train_config
00135 | 
00136 |     def test_matching_deeph_split_audit_passes(self) -> None:
00137 |         with fake_numpy():
00138 |             frozen, raw_mirror, train_config = self.build_case()
00139 | 
00140 |             audit = audit_deeph_split(
00141 |                 frozen_split_manifest=frozen,
00142 |                 raw_mirror=raw_mirror,
00143 |                 processed_dir=self.root / "deeph" / "processed",
00144 |                 train_config_path=train_config,
00145 |                 output_json=self.root / "deeph" / "deeph_split_audit.json",
00146 |                 output_csv=self.root / "deeph" / "deeph_split_audit.csv",
00147 |             )
00148 | 
00149 |         self.assertEqual(audit["status"], STATUS_VALID)
00150 |         self.assertTrue(audit["robust_winner_allowed"])
00151 |         self.assertTrue((self.root / "deeph" / "deeph_split_audit.json").exists())
00152 |         self.assertTrue((self.root / "deeph" / "deeph_split_audit.csv").exists())
00153 | 
00154 |     def test_swapped_frozen_split_fails(self) -> None:
00155 |         with fake_numpy():
00156 |             frozen, raw_mirror, train_config = self.build_case()
00157 |             swapped = {**frozen, "rows": [dict(row) for row in frozen["rows"]]}
00158 |             swapped["rows"][0]["split"] = "test"
00159 |             swapped["rows"][2]["split"] = "train"
00160 | 
00161 |             audit = audit_deeph_split(
00162 |                 frozen_split_manifest=swapped,
00163 |                 raw_mirror=raw_mirror,
00164 |                 processed_dir=self.root / "deeph" / "processed",
00165 |                 train_config_path=train_config,
00166 |             )
00167 | 
00168 |         self.assertEqual(audit["status"], STATUS_INCOMPATIBLE)
00169 |         self.assertFalse(audit["robust_winner_allowed"])
00170 |         self.assertTrue(audit["mismatched_rows"])
00171 | 
00172 |     def test_unknown_processed_ordering_is_unverified(self) -> None:
00173 |         frozen, raw_mirror, train_config = self.build_case()
00174 |         processed_dir = self.root / "deeph" / "processed"
00175 |         for rc_file in processed_dir.rglob("rc.h5"):
00176 |             rc_file.unlink()
00177 | 
00178 |         audit = audit_deeph_split(
00179 |             frozen_split_manifest=frozen,
00180 |             raw_mirror=raw_mirror,
00181 |             processed_dir=processed_dir,
00182 |             train_config_path=train_config,
00183 |         )
00184 | 
00185 |         self.assertEqual(audit["status"], STATUS_UNVERIFIED)
00186 |         self.assertEqual(audit["comparability_status"], STATUS_UNVERIFIED)
00187 |         self.assertFalse(audit["robust_winner_allowed"])
00188 | 
00189 |     def test_different_train_seed_is_invalid(self) -> None:
00190 |         with fake_numpy():
00191 |             frozen, raw_mirror, _train_config = self.build_case(seed=123)
00192 |             train_config = write_train_config(self.root / "deeph", raw_mirror, seed=456)
00193 | 
00194 |             audit = audit_deeph_split(
00195 |                 frozen_split_manifest=frozen,
00196 |                 raw_mirror=raw_mirror,
00197 |                 processed_dir=self.root / "deeph" / "processed",
00198 |                 train_config_path=train_config,
00199 |             )
00200 | 
00201 |         self.assertIn(audit["status"], {STATUS_INCOMPATIBLE, STATUS_UNVERIFIED})
00202 |         self.assertFalse(audit["robust_winner_allowed"])
00203 |         self.assertTrue(any("seed" in error for error in audit["errors"]))
00204 | 
00205 | 
00206 | if __name__ == "__main__":
00207 |     unittest.main()
```

## `tests/test_joint_artifact_contract.py`

SHA-256: `ac44c71921d3416bae8175e8c18348ce5dcc55d3163029ef1cb5f85f2c433bcc`

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
00011 | SHARED_DIR = REPO_ROOT / "shared"
00012 | if str(SHARED_DIR) not in sys.path:
00013 |     sys.path.insert(0, str(SHARED_DIR))
00014 | 
00015 | from joint_artifact_contract import (  # noqa: E402
00016 |     CONTRACT_NAME,
00017 |     G2M_DEEPH_BENCHMARK_PROFILE,
00018 |     validate_dataset,
00019 |     validate_snapshot,
00020 | )
00021 | 
00022 | 
00023 | def write_snapshot(
00024 |     path: Path,
00025 |     *,
00026 |     label: str = "graphene",
00027 |     include_run_out: bool = True,
00028 |     include_tshs: bool = True,
00029 |     include_tsde: bool = True,
00030 |     include_hsx: bool = True,
00031 |     include_struct_out: bool = True,
00032 |     include_xv: bool = True,
00033 |     include_orb_indx: bool = True,
00034 | ) -> None:
00035 |     path.mkdir(parents=True, exist_ok=True)
00036 |     (path / "RUN.fdf").write_text(f"SystemLabel {label}\n", encoding="utf-8")
00037 |     (path / "metadata.json").write_text('{"system_label": "%s"}\n' % label, encoding="utf-8")
00038 |     if include_run_out:
00039 |         (path / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
00040 |     files = {
00041 |         ".TSHS": include_tshs,
00042 |         ".TSDE": include_tsde,
00043 |         ".HSX": include_hsx,
00044 |         ".STRUCT_OUT": include_struct_out,
00045 |         ".XV": include_xv,
00046 |         ".ORB_INDX": include_orb_indx,
00047 |     }
00048 |     for suffix, enabled in files.items():
00049 |         if enabled:
00050 |             (path / f"{label}{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
00051 | 
00052 | 
00053 | def write_dataset_provenance(
00054 |     dataset: Path,
00055 |     *,
00056 |     include_basis: bool = True,
00057 |     include_pseudo: bool = True,
00058 |     include_material: bool = True,
00059 |     include_fdf: bool = True,
00060 |     include_siesta_version: bool = True,
00061 |     include_siesta_command: bool = True,
00062 |     include_environment: bool = True,
00063 |     include_run_log: bool = True,
00064 | ) -> None:
00065 |     payload = {}
00066 |     if include_material:
00067 |         payload["label"] = "graphene"
00068 |     if include_basis:
00069 |         payload["basis_file_sha256"] = {"C.ion.xml": "basis-hash"}
00070 |     if include_pseudo:
00071 |         payload["pseudopotential_sha256"] = {"C": "pseudo-hash"}
00072 |     if include_fdf:
00073 |         payload["fdf_sha256"] = "fdf-hash"
00074 |     if include_siesta_version:
00075 |         payload["siesta_version"] = "SIESTA test-version"
00076 |     if include_siesta_command:
00077 |         payload["siesta_command_line"] = "bash -lc 'siesta < RUN.fdf'"
00078 |     if include_environment:
00079 |         payload["environment"] = {"python_version": "3.11.0", "platform": "test-platform"}
00080 |     if include_run_log:
00081 |         (dataset / "RUN.out").write_text("Job completed\n", encoding="utf-8")
00082 |         payload["run_out_path"] = str(dataset / "RUN.out")
00083 |     (dataset / "material_provenance.json").write_text(
00084 |         json.dumps(payload, sort_keys=True) + "\n",
00085 |         encoding="utf-8",
00086 |     )
00087 | 
00088 | 
00089 | class JointArtifactContractTests(unittest.TestCase):
00090 |     def setUp(self) -> None:
00091 |         self.tmp = tempfile.TemporaryDirectory()
00092 |         self.root = Path(self.tmp.name)
00093 | 
00094 |     def tearDown(self) -> None:
00095 |         self.tmp.cleanup()
00096 | 
00097 |     def test_valid_snapshot_with_all_artifacts(self) -> None:
00098 |         sample = self.root / "sample_0001"
00099 |         write_snapshot(sample)
00100 | 
00101 |         result = validate_snapshot(sample)
00102 | 
00103 |         self.assertEqual(result.contract_name, CONTRACT_NAME)
00104 |         self.assertTrue(result.valid)
00105 |         self.assertFalse(result.repair_required)
00106 |         self.assertEqual(result.system_label, "graphene")
00107 |         self.assertEqual(result.missing_required, [])
00108 |         self.assertIn("hsx", result.present_artifacts)
00109 | 
00110 |     def test_missing_hsx_is_invalid_and_repair_required(self) -> None:
00111 |         sample = self.root / "sample_0001"
00112 |         write_snapshot(sample, include_hsx=False)
00113 | 
00114 |         result = validate_snapshot(sample)
00115 | 
00116 |         self.assertFalse(result.valid)
00117 |         self.assertTrue(result.repair_required)
00118 |         self.assertIn("hsx", result.missing_required)
00119 | 
00120 |     def test_missing_struct_out_is_invalid_and_repair_required(self) -> None:
00121 |         sample = self.root / "sample_0001"
00122 |         write_snapshot(sample, include_struct_out=False)
00123 | 
00124 |         result = validate_snapshot(sample)
00125 | 
00126 |         self.assertFalse(result.valid)
00127 |         self.assertIn("struct_out", result.missing_required)
00128 | 
00129 |     def test_missing_orb_indx_is_invalid_and_repair_required(self) -> None:
00130 |         sample = self.root / "sample_0001"
00131 |         write_snapshot(sample, include_orb_indx=False)
00132 | 
00133 |         result = validate_snapshot(sample)
00134 | 
00135 |         self.assertFalse(result.valid)
00136 |         self.assertIn("orb_indx", result.missing_required)
00137 | 
00138 |     def test_missing_tshs_when_required_is_invalid(self) -> None:
00139 |         sample = self.root / "sample_0001"
00140 |         write_snapshot(sample, include_tshs=False)
00141 | 
00142 |         result = validate_snapshot(sample, require_tshs=True)
00143 | 
00144 |         self.assertFalse(result.valid)
00145 |         self.assertIn("tshs", result.missing_required)
00146 | 
00147 |     def test_missing_tsde_when_required_is_invalid(self) -> None:
00148 |         sample = self.root / "sample_0001"
00149 |         write_snapshot(sample, include_tsde=False)
00150 | 
00151 |         result = validate_snapshot(sample, require_tsde=True)
00152 | 
00153 |         self.assertFalse(result.valid)
00154 |         self.assertIn("tsde", result.missing_required)
00155 | 
00156 |     def test_ambiguous_system_label_fails_clearly(self) -> None:
00157 |         sample = self.root / "sample_0001"
00158 |         write_snapshot(sample, label="graphene")
00159 |         (sample / "other.HSX").write_text("other\n", encoding="utf-8")
00160 | 
00161 |         result = validate_snapshot(sample)
00162 | 
00163 |         self.assertFalse(result.valid)
00164 |         self.assertIsNone(result.system_label)
00165 |         self.assertTrue(any("ambiguous SystemLabel" in error for error in result.errors))
00166 | 
00167 |     def test_dataset_summary_counts_valid_and_invalid_snapshots(self) -> None:
00168 |         valid = self.root / "dataset" / "valid"
00169 |         invalid = self.root / "dataset" / "invalid"
00170 |         write_snapshot(valid)
00171 |         write_snapshot(invalid, include_hsx=False)
00172 | 
00173 |         result = validate_dataset(self.root / "dataset")
00174 | 
00175 |         self.assertFalse(result.valid)
00176 |         self.assertEqual(result.total_snapshots, 2)
00177 |         self.assertEqual(result.valid_snapshots, 1)
00178 |         self.assertEqual(result.invalid_snapshots, 1)
00179 |         self.assertEqual(result.repair_required_snapshots, 1)
00180 | 
00181 |     def test_old_graph2mat_only_snapshot_is_not_benchmark_ready(self) -> None:
00182 |         sample = self.root / "old_graph2mat_only"
00183 |         write_snapshot(sample, include_hsx=False, include_struct_out=False, include_orb_indx=False)
00184 | 
00185 |         result = validate_snapshot(sample)
00186 | 
00187 |         self.assertFalse(result.valid)
00188 |         self.assertTrue(result.repair_required)
00189 |         self.assertEqual(
00190 |             {"hsx", "struct_out", "orb_indx"},
00191 |             set(result.missing_required),
00192 |         )
00193 | 
00194 |     def test_strict_profile_requires_basis_provenance(self) -> None:
00195 |         dataset = self.root / "dataset"
00196 |         write_snapshot(dataset / "sample_0001")
00197 |         write_dataset_provenance(dataset, include_basis=False)
00198 | 
00199 |         result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)
00200 | 
00201 |         self.assertFalse(result.valid)
00202 |         self.assertIn("dataset-level basis provenance or basis file hashes are missing", result.errors)
00203 | 
00204 |     def test_strict_profile_requires_pseudopotential_provenance(self) -> None:
00205 |         dataset = self.root / "dataset"
00206 |         write_snapshot(dataset / "sample_0001")
00207 |         write_dataset_provenance(dataset, include_pseudo=False)
00208 | 
00209 |         result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)
00210 | 
00211 |         self.assertFalse(result.valid)
00212 |         self.assertIn("dataset-level pseudopotential provenance or hashes are missing", result.errors)
00213 | 
00214 |     def test_strict_profile_requires_material_identity(self) -> None:
00215 |         dataset = self.root / "dataset"
00216 |         write_snapshot(dataset / "sample_0001")
00217 |         write_dataset_provenance(dataset, include_material=False)
00218 | 
00219 |         result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)
00220 | 
00221 |         self.assertFalse(result.valid)
00222 |         self.assertIn("dataset-level material identity is missing", result.errors)
00223 | 
00224 |     def test_strict_profile_with_full_provenance_passes(self) -> None:
00225 |         dataset = self.root / "dataset"
00226 |         write_snapshot(dataset / "sample_0001")
00227 |         write_dataset_provenance(dataset)
00228 | 
00229 |         result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)
00230 | 
00231 |         self.assertTrue(result.valid)
00232 |         self.assertTrue(result.basis_present)
00233 |         self.assertTrue(result.pseudopotential_provenance_present)
00234 |         self.assertTrue(result.material_identity_present)
00235 |         self.assertTrue(result.siesta_input_provenance_present)
00236 |         self.assertTrue(result.siesta_version_provenance_present)
00237 |         self.assertTrue(result.siesta_command_line_provenance_present)
00238 |         self.assertTrue(result.siesta_environment_provenance_present)
00239 |         self.assertTrue(result.siesta_execution_log_present)
00240 | 
00241 |     def test_strict_profile_requires_siesta_version_provenance(self) -> None:
00242 |         dataset = self.root / "dataset"
00243 |         write_snapshot(dataset / "sample_0001")
00244 |         write_dataset_provenance(dataset, include_siesta_version=False)
00245 | 
00246 |         result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)
00247 | 
00248 |         self.assertFalse(result.valid)
00249 |         self.assertIn("dataset-level SIESTA version provenance is missing", result.errors)
00250 | 
00251 |     def test_strict_profile_requires_siesta_command_line_provenance(self) -> None:
00252 |         dataset = self.root / "dataset"
00253 |         write_snapshot(dataset / "sample_0001")
00254 |         write_dataset_provenance(dataset, include_siesta_command=False)
00255 | 
00256 |         result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)
00257 | 
00258 |         self.assertFalse(result.valid)
00259 |         self.assertIn("dataset-level SIESTA command-line provenance is missing", result.errors)
00260 | 
00261 |     def test_strict_profile_requires_environment_provenance(self) -> None:
00262 |         dataset = self.root / "dataset"
00263 |         write_snapshot(dataset / "sample_0001")
00264 |         write_dataset_provenance(dataset, include_environment=False)
00265 | 
00266 |         result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)
00267 | 
00268 |         self.assertFalse(result.valid)
00269 |         self.assertIn("dataset-level execution environment provenance is missing", result.errors)
00270 | 
00271 |     def test_non_strict_fixture_can_skip_dataset_provenance_explicitly(self) -> None:
00272 |         dataset = self.root / "dataset"
00273 |         write_snapshot(dataset / "sample_0001")
00274 | 
00275 |         result = validate_dataset(dataset, require_dataset_provenance=False)
00276 | 
00277 |         self.assertTrue(result.valid)
00278 |         self.assertFalse(result.basis_present)
00279 |         self.assertFalse(result.pseudopotential_provenance_present)
00280 |         self.assertFalse(result.material_identity_present)
00281 | 
00282 | 
00283 | if __name__ == "__main__":
00284 |     unittest.main()
```

## `tests/test_artifact_signature.py`

SHA-256: `469adda0b1c60f018bbb207d334c31fc1697307a142b9f7b1d3a4ac65880a942`

```py
00001 | """Fase 5 (audit): derivative cache signatures — stale results must not be reused."""
00002 | 
00003 | from __future__ import annotations
00004 | 
00005 | import json
00006 | import sys
00007 | from pathlib import Path
00008 | 
00009 | import numpy as np
00010 | from scipy import sparse
00011 | 
00012 | REPO_ROOT = Path(__file__).resolve().parents[1]
00013 | sys.path.insert(0, str(REPO_ROOT / "shared"))
00014 | 
00015 | from artifact_signature import (  # noqa: E402
00016 |     CACHE_LEGACY_UNVERIFIED,
00017 |     CACHE_MISSING_METADATA,
00018 |     CACHE_NON_FINITE,
00019 |     CACHE_SIGNATURE_MISMATCH,
00020 |     CACHE_UNREADABLE,
00021 |     CACHE_VALID,
00022 |     cached_result_status,
00023 |     input_signature_sha256,
00024 | )
00025 | 
00026 | BASE_PAYLOAD = {
00027 |     "model": "graph2mat",
00028 |     "checkpoint_sha256": "abc",
00029 |     "repository_commits": {"MD": "sha1"},
00030 |     "structure_fdf_sha256": "fdf",
00031 |     "dtype": "float64",
00032 |     "atom_index": 0,
00033 |     "axis_index": 1,
00034 | }
00035 | 
00036 | 
00037 | def _write_cache(tmp_path: Path, signature: str | None, data=None, shape=None):
00038 |     matrix = sparse.csr_matrix(
00039 |         np.asarray(data if data is not None else [[1.0, 0.0], [0.0, 2.0]])
00040 |     )
00041 |     npz = tmp_path / "d.npz"
00042 |     with npz.open("wb") as handle:
00043 |         sparse.save_npz(handle, matrix)
00044 |     metadata = {"matrix_shape": shape or list(matrix.shape)}
00045 |     if signature is not None:
00046 |         metadata["input_signature_sha256"] = signature
00047 |     meta = tmp_path / "d.json"
00048 |     meta.write_text(json.dumps(metadata), encoding="utf-8")
00049 |     return npz, meta
00050 | 
00051 | 
00052 | def test_signature_is_deterministic_and_sensitive():
00053 |     sig = input_signature_sha256(BASE_PAYLOAD)
00054 |     assert sig == input_signature_sha256(dict(BASE_PAYLOAD))
00055 |     for key, value in [
00056 |         ("checkpoint_sha256", "OTHER"),        # different checkpoint
00057 |         ("structure_fdf_sha256", "OTHER"),      # different coordinates
00058 |         ("dtype", "float32"),                   # different dtype
00059 |         ("repository_commits", {"MD": "sha2"}),  # different code
00060 |         ("atom_index", 1),
00061 |     ]:
00062 |         assert input_signature_sha256({**BASE_PAYLOAD, key: value}) != sig, key
00063 | 
00064 | 
00065 | def test_matching_signature_is_valid(tmp_path):
00066 |     sig = input_signature_sha256(BASE_PAYLOAD)
00067 |     npz, meta = _write_cache(tmp_path, sig)
00068 |     assert cached_result_status(npz, meta, sig) == CACHE_VALID
00069 | 
00070 | 
00071 | def test_mismatched_signature_rejected(tmp_path):
00072 |     npz, meta = _write_cache(tmp_path, "stale-signature")
00073 |     sig = input_signature_sha256(BASE_PAYLOAD)
00074 |     assert cached_result_status(npz, meta, sig) == CACHE_SIGNATURE_MISMATCH
00075 | 
00076 | 
00077 | def test_legacy_without_signature_rejected(tmp_path):
00078 |     npz, meta = _write_cache(tmp_path, None)
00079 |     sig = input_signature_sha256(BASE_PAYLOAD)
00080 |     assert cached_result_status(npz, meta, sig) == CACHE_LEGACY_UNVERIFIED
00081 | 
00082 | 
00083 | def test_missing_metadata_rejected(tmp_path):
00084 |     npz, meta = _write_cache(tmp_path, "x")
00085 |     meta.unlink()
00086 |     assert cached_result_status(npz, meta, "x") == CACHE_MISSING_METADATA
00087 | 
00088 | 
00089 | def test_non_finite_payload_rejected(tmp_path):
00090 |     sig = input_signature_sha256(BASE_PAYLOAD)
00091 |     npz, meta = _write_cache(tmp_path, sig, data=[[np.nan, 0.0], [0.0, 1.0]])
00092 |     assert cached_result_status(npz, meta, sig) == CACHE_NON_FINITE
00093 | 
00094 | 
00095 | def test_shape_mismatch_rejected(tmp_path):
00096 |     sig = input_signature_sha256(BASE_PAYLOAD)
00097 |     npz, meta = _write_cache(tmp_path, sig, shape=[4, 4])
00098 |     assert cached_result_status(npz, meta, sig) == CACHE_SIGNATURE_MISMATCH
00099 | 
00100 | 
00101 | def test_corrupt_npz_rejected(tmp_path):
00102 |     sig = input_signature_sha256(BASE_PAYLOAD)
00103 |     npz, meta = _write_cache(tmp_path, sig)
00104 |     npz.write_bytes(b"not-an-npz")
00105 |     assert cached_result_status(npz, meta, sig) == CACHE_UNREADABLE
```
