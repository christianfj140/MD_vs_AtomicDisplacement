#!/usr/bin/env python3
"""Build line-numbered, blind scientific-audit contexts from the repository."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

EXCLUDED = ("audit_corrections", "scratchpad", "prompt_goal", "prompt_goal_simetria")

DOSSIERS: dict[str, dict[str, Any]] = {
    "03A1_generation_md_fc.md": {
        "title": "Dossier 1A — Generación MD, FC y cartesiana genérica",
        "questions": (
            "Auditar cómo se construyen MD, FC y desplazamientos aleatorios; "
            "comprobar unidades, condiciones de contorno, independencia de splits, "
            "geometrías, validaciones y procedencia SIESTA."
        ),
        "full": [
            "README.md",
            "MD/pipeline_config.yaml",
            "AtomDisplacement/pipeline_config.yaml",
            "Comparison/config/shared_siesta_settings.yaml",
            "AtomDisplacement/scripts/generate_atom_displacement_dataset.py",
            "AtomDisplacement/scripts/generate_generic_cartesian_displacement_dataset.py",
            "shared/material_bundle.py",
            "shared/fdf_materialization.py",
            "shared/siesta_run_fdf.py",
            "tests/test_generic_cartesian_displacement.py",
            "tests/test_material_bundle.py",
            "tests/test_fdf_materialization.py",
            "tests/test_siesta_material_provenance.py",
        ],
        "symbols": {
            "MD/scripts/generate_md_dataset.py": [
                "execution_environment_provenance",
                "probe_siesta_version",
                "prepare_material_inputs",
                "write_run_fdf",
                "_split_counts",
                "_select_spread",
                "_split_spread",
                "_split_block",
                "_split_blocked_with_gap",
                "parse_xv_geometry",
                "rewrite_run_fdf_from_xv",
                "write_joint_snapshot_metadata",
                "effective_fdf_geometry_signature",
                "xv_geometry_signature",
                "run_temperature_block",
                "combine_temperature_blocks",
                "run_temperature_block_dataset",
                "write_excluded_gap_manifest",
                "write_split_manifests",
                "write_split_summary",
                "prepare_dataset_splits",
            ],
        },
    },
    "03A2_expanded_recipes.md": {
        "title": "Dossier 1B — Recetas expandidas de datasets",
        "questions": (
            "Auditar composición y escalado de las recetas MD, FC y random Cartesian, "
            "incluyendo temperaturas, amplitudes, seeds, tamaños y repetición de bloques."
        ),
        "full": [
            "MD/pipeline_config.yaml",
            "AtomDisplacement/pipeline_config.yaml",
        ],
        "compact_json": [
            "Comparison/dataset_recipes/h2o_efficiency_reliable_3seed_train_200epochs.json",
            "Comparison/dataset_recipes/h2o_recommended_190_570_1140.json",
            "Comparison/dataset_recipes/h2o_scientific_285_750_1500_improved.json",
            "Comparison/dataset_recipes/scientific_large_3seed_equalN.json",
        ],
    },
    "03A3_random_sampling.md": {
        "title": "Dossier 1C — Muestreo cartesiano aleatorio",
        "questions": (
            "Auditar distribuciones y amplitudes, transformaciones geométricas, "
            "rechazo de estructuras, reproducibilidad, agrupación familiar y aislamiento de splits."
        ),
        "full": [
            "AtomDisplacement/pipeline_config.yaml",
            "tests/test_generic_random_cartesian.py",
        ],
        "symbols": {
            "AtomDisplacement/scripts/generate_random_cartesian_dataset.py": [
                "normalize_validation_config",
                "normalize_component_config",
                "normalize_generic_random_cartesian_config",
                "random_cartesian_config",
                "moving_atom_indices",
                "sample_displacement_vector",
                "displacement_field",
                "positions_with_displacements",
                "minimum_pair_distance",
                "apply_bond_displacement",
                "apply_angle_displacement",
                "apply_atom_displacement",
                "remove_mean_translation_from_reference",
                "build_geometry_metrics",
                "generate_candidate",
                "validate_random_structure",
                "random_cartesian_family_payload",
                "deterministic_split_group_id",
                "random_cartesian_split_group",
                "grouped_split_assignment",
                "assert_group_isolation",
                "write_split_manifests",
                "generate_generic_random_cartesian_dataset",
                "generate_dataset",
            ],
        },
    },
    "03B_material_inputs.md": {
        "title": "Dossier 1D — Materiales e inputs SIESTA",
        "questions": (
            "Auditar especies, geometrías, celdas, k-points, malla, funcional, "
            "bases, pseudopotenciales, espín y consistencia entre familias de materiales."
        ),
        "globs": ["materials/*/material.yaml", "materials/*/RUN*.fdf"],
        "full": [
            "AtomDisplacement/base/RUN.fdf",
            "Comparison/config/ml_vs_siesta_example_structure.fdf",
            "configs/config_fc.yaml",
            "configs/config_md.yaml",
        ],
        "compact_json": [
            "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/"
            "benchmark_dataset_manifest.json",
            "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/"
            "frozen_split_manifest.json",
            "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/"
            "md_temperature_blocks_manifest.json",
            "Comparison/datasets/graphene_5x5_vacancy/benchmark_dataset_manifest.json",
            "Comparison/datasets/graphene_hBN_bilayer_train/benchmark_dataset_manifest.json",
            "Comparison/datasets/graphene_hBN_moire_22deg/benchmark_dataset_manifest.json",
        ],
    },
    "03C1_protocol_provenance.md": {
        "title": "Dossier 2A — Protocolo, splits y procedencia",
        "questions": (
            "Auditar si ambos modelos resuelven el mismo problema: base y orden orbital, "
            "vectores R, espín, overlap, referencia energética, k-points, splits y gates."
        ),
        "full": [
            "docs/graph2mat_deeph_benchmark.md",
            "docs/cross_structure_evaluation.md",
            "Comparison/config/g2m_deeph_paper_protocol_v1_example.json",
            "Comparison/config/graphene_w90_g2m_deeph_weekend_iid1000_paper_ready_v1.json",
            "Comparison/scripts/g2m_deeph_protocol.py",
            "tests/test_g2m_deeph_protocol.py",
        ],
    },
    "03C2_splits_provenance.md": {
        "title": "Dossier 2B — Splits, referencias y procedencia",
        "questions": (
            "Auditar aislamiento train/validation/test, referencias prohibidas, "
            "identidad material, contratos de artefactos y estado reproducible."
        ),
        "full": [
            "Comparison/scripts/deeph_fair_utils.py",
            "Comparison/scripts/deeph_split_audit.py",
            "Comparison/scripts/g2m_deeph_test_blindness.py",
            "Comparison/scripts/reference_selection.py",
            "Comparison/scripts/material_provenance.py",
            "shared/joint_artifact_contract.py",
            "shared/benchmark_manifest.py",
            "shared/artifact_signature.py",
            "shared/run_inventory.py",
            "tests/test_g2m_deeph_test_blindness.py",
            "tests/test_deeph_split_audit.py",
            "tests/test_joint_artifact_contract.py",
            "tests/test_artifact_signature.py",
        ],
    },
    "03C3_basis_equivalence.md": {
        "title": "Dossier 2C — Equivalencia de base Graph2Mat–DeepH",
        "questions": (
            "Auditar mapeo y orden orbital, bloques R, construcción H(k), overlap, "
            "shift energético, hermiticidad y evidencia necesaria para declarar equivalencia."
        ),
        "full": [
            "Comparison/scripts/reference_selection.py",
            "tests/test_deeph_raw_global_equivalence_preflight.py",
            "tests/test_method_provenance_fairness.py",
        ],
        "symbols": {
            "Comparison/scripts/deeph_prediction_adapter.py": [
                "equivalence_status_from_adapter_status",
                "equivalence_scope_from_adapter_status",
                "DeepHPredictionAdapterResult",
                "find_raw_global_equivalence_evidence",
                "validate_raw_global_equivalence_evidence",
                "parse_block_key",
                "expected_block_shape",
                "h5_block_shapes",
                "assemble_hk",
                "hermiticity_defect",
                "adapt_deeph_prediction_sample",
                "write_adapter_manifest",
            ],
            "Comparison/scripts/deeph_raw_global_equivalence_preflight.py": [
                "normalize_orbital_label",
                "siesta_orbital_labels_from_orb_indx",
                "deeph_orbital_labels_from_orbital_types",
                "derive_deeph_to_siesta_basis_transform",
                "apply_basis_transform",
                "fit_energy_reference_shift",
                "select_frozen_rows",
                "raw_reference_matrices",
                "kpoints_from_fdf",
                "numeric_evidence_for_sample",
                "build_preflight_manifest",
            ],
        },
        "compact_json": [
            "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/"
            "artifact_validation.json",
            "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/"
            "benchmark_dataset_manifest.json",
            "Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1000/"
            "frozen_split_manifest.json",
            "Comparison/datasets/graphene_5x5_vacancy/artifact_validation.json",
            "Comparison/datasets/graphene_5x5_vacancy/benchmark_dataset_manifest.json",
            "Comparison/datasets/graphene_5x5_vacancy/frozen_split_manifest.json",
        ],
    },
    "03D1_hamiltonian_spectra.md": {
        "title": "Dossier 3A — Hamiltonianos, overlap y espectros",
        "questions": (
            "Auditar H y S, compatibilidad de matrices, hermiticidad, problema "
            "generalizado, métricas sparse/espectrales, stencils, unidades, delta y autograd."
        ),
        "full": [
            "docs/derivadas_simetria.md",
            "Comparison/scripts/g2m_deeph_metrics.py",
            "tests/test_graphene_band_comparison.py",
        ],
        "symbols": {
            "Comparison/scripts/evaluate_hamiltonian_metrics.py": [
                "MatrixData",
                "MonkhorstPackKGrid",
                "_monkhorst_pack_axis_points",
                "_monkhorst_pack_points",
                "_monkhorst_pack_grid",
                "parse_monkhorst_pack_kgrid",
                "evaluate_kpoint_sample",
                "evaluate_sample",
                "read_matrix",
                "matrix_compatibility_errors",
                "matrix_compatibility_warnings",
                "overlap_diagnostics",
                "matrix_semantics_fields",
                "component_channel_metrics",
                "hermiticity_defect",
                "sparse_metrics",
                "complex_matrix_error_metrics",
                "kpoint_hamiltonian_matrix",
                "kpoint_overlap_matrix",
                "complex_generalized_eigenvalues",
                "kpoint_eigenvalues_with_reference_overlap",
                "generalized_eigenvalues",
                "low_energy_metrics",
                "eigen_error_metrics",
                "dos_for_sample",
                "dos_fermi_window_metrics",
                "kpoint_weighted_dos_metrics",
                "prediction_artifact_safety_summary",
                "matrix_spectrum_rows",
            ],
        },
    },
    "03D2_derivatives.md": {
        "title": "Dossier 3B — Derivadas FD y autograd",
        "questions": (
            "Auditar definición de dH/dR, stencils, signo, unidades, delta, ruido, "
            "soporte sparse, geometrías, sum rule traslacional y gates de comparabilidad."
        ),
        "full": [
            "Comparison/config/derivative_metrics_only_existing_artifacts.json",
            "Comparison/config/derivative_stencils_only_minimal.json",
            "Comparison/config/adaptive_derivative_selection_smoke.json",
            "Comparison/scripts/graph2mat_autograd_derivatives.py",
            "tests/test_graph2mat_autograd_derivatives.py",
            "tests/test_deeph_autograd_derivatives.py",
        ],
        "symbols": {
            "Comparison/scripts/hamiltonian_derivative_stencil.py": [
                "DerivativeMetadata",
                "DerivativeMatrixInput",
                "DerivativeStencil",
                "DerivativeMatrixResult",
                "DerivativeComparisonResult",
                "DerivativeSparseMetrics",
                "validate_derivative_geometry",
                "finite_difference_derivative",
                "derivative_signal_to_noise_metrics",
                "finite_difference_derivative_pair",
                "direct_predicted_derivative_pair",
                "derivative_sparse_metrics",
                "derivative_ref_abs_quantile_metrics",
                "discover_derivative_stencils",
                "sparse_hermiticity_defect",
                "sparse_blockwise_hermiticity_defect",
                "validate_derivative_stencil",
                "_validate_metadata",
                "_validate_units",
                "_validate_unit_metadata_explicit",
                "_validate_operands",
                "_validate_comparability_hashes",
            ],
            "Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py": [
                "evaluate_derivative_metrics",
                "_evaluate_discovery",
                "_scientific_status",
                "_micro_macro_domain",
                "_summary",
                "_derivative_group_metrics",
                "_aggregate_derivative_group",
                "_delta_stability_summary",
                "_delta_stability_convergence_summary",
                "_reference_noise_summary",
            ],
            "Comparison/scripts/g2m_deeph_derivative_gate_check.py": [
                "central_stencil_rows",
                "central_metric_rows",
                "max_hermiticity_defect",
                "support_discontinuity_detected",
                "has_consistent_required_hashes",
                "geometry_validation_passed",
                "unit_metadata_explicit",
                "delta_sensitivity_has_two_deltas",
                "split_consistency_proven",
                "deeph_autograd_equivalence_proven",
                "dataset_paper_evidence",
                "evaluate_dataset",
                "overall_status",
                "allowed_claims_for_status",
                "blocked_claims_for_status",
                "build_derivative_gate_report",
            ],
        },
    },
    "03E1_claims_gates.md": {
        "title": "Dossier 4A — Claims, rankings y gates",
        "questions": (
            "Auditar qué claims sobreviven a los datos: seeds, intervalos, checkpoint "
            "selection, test blindness, leakage temporal, agregación, umbrales y gates."
        ),
        "full": [
            "docs/phase6_hamiltonian_architecture_benchmark.md",
            "docs/derivative_smoke_validation_note.md",
            "Comparison/config/h2o_phase6_hamiltonian_architecture_benchmark_payload.json",
            "Comparison/scripts/g2m_deeph_rank_runs.py",
            "Comparison/scripts/g2m_deeph_final_stats.py",
            "Comparison/scripts/g2m_deeph_gate_check.py",
            "Comparison/scripts/g2m_deeph_early_stopping.py",
            "tests/test_g2m_deeph_rank_runs.py",
            "tests/test_g2m_deeph_final_stats.py",
            "tests/test_g2m_deeph_gate_check.py",
        ],
        "symbols": {
            "Comparison/scripts/g2m_deeph_paper_diagnostics.py": [
                "representative_rows",
                "best_median_worst",
                "aggregate_rows",
                "linear_regression_summary",
                "gate_release_rows",
                "build_diagnostics",
            ],
        },
    },
    "03E2_dataset_size_statistics.md": {
        "title": "Dossier 4B — Dataset-size minimum y estadística",
        "questions": (
            "Auditar agregación de réplicas y seeds, definición de N mínimo, "
            "umbrales, sensibilidad, fits, extrapolación y estabilidad predictiva."
        ),
        "full": [
            "Comparison/config/graphene_dataset_size_minimum_paper_threshold_protocol.json",
            "Comparison/config/graphene_dataset_size_minimum_paper_audit_payload.json",
        ],
        "symbols": {
            "Comparison/scripts/g2m_deeph_dataset_size_minimum.py": [
                "mean",
                "std",
                "sem",
                "resolve_threshold_protocol",
                "row_dataset_sizes",
                "normalize_rows",
                "aggregate_rows_mean_replicates",
                "aggregate_rows_mean_seeds_per_config",
                "aggregate_rows_best_config_mean",
                "analysis_rows_for_aggregation_mode",
                "best_by_method_size",
                "mean_by_method_size",
                "n_min_abs",
                "n_min_rel_tol",
                "n_min_rel95",
                "n_min_plateau",
                "n_min_cost_eff",
                "thresholds_by_method",
                "threshold_sensitivity_summary",
                "fit_predictive_stability_by_left_out_N",
                "fit_linear_model",
                "fit_power_law_floor",
                "fit_summary",
            ],
        },
        "compact_json": [
            "Comparison/results/dataset_size_minimum_ui_20260722_122042/"
            "threshold_10meV/dataset_size_minimum_summary.json",
            "Comparison/results/dataset_size_minimum_ui_20260722_122042/"
            "threshold_25meV/dataset_size_minimum_summary.json",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numbered(text: str, start: int = 1) -> str:
    return "\n".join(f"{line_no:05d} | {line}" for line_no, line in enumerate(text.splitlines(), start))


def full_section(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    text = path.read_text(encoding="utf-8", errors="replace")
    language = path.suffix.lstrip(".") or "text"
    return (
        f"## `{relative}`\n\nSHA-256: `{sha256(path)}`\n\n"
        f"```{language}\n{numbered(text)}\n```\n"
    )


def symbol_section(relative: str, names: list[str]) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=relative)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = sorted(set(names) - nodes.keys())
    if missing:
        raise ValueError(f"{relative}: missing symbols {missing}")

    first_node_line = min((node.lineno for node in nodes.values()), default=len(lines) + 1)
    parts = [
        f"## `{relative}` — extractos seleccionados\n\nSHA-256 del archivo completo: `{sha256(path)}`\n",
        "### Cabecera, imports y constantes iniciales\n\n"
        f"```py\n{numbered(chr(10).join(lines[: first_node_line - 1]))}\n```\n",
    ]
    for name in names:
        node = nodes[name]
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        end = node.end_lineno or node.lineno
        parts.append(
            f"### `{name}` — líneas {start}–{end}\n\n"
            f"```py\n{numbered(chr(10).join(lines[start - 1 : end]), start)}\n```\n"
        )
    return "\n".join(parts)


def compact(value: Any, depth: int = 0) -> Any:
    if depth >= 8:
        if isinstance(value, list):
            return {"_omitted_list_length": len(value)}
        if isinstance(value, dict):
            return {"_omitted_object_keys": sorted(value)}
        return value
    if isinstance(value, dict):
        return {key: compact(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) <= 8:
            return [compact(item, depth + 1) for item in value]
        return {
            "_list_length": len(value),
            "_first_two": [compact(item, depth + 1) for item in value[:2]],
            "_last_two": [compact(item, depth + 1) for item in value[-2:]],
        }
    if isinstance(value, str) and len(value) > 2000:
        return {
            "_string_length": len(value),
            "_prefix": value[:500],
            "_sha256": hashlib.sha256(value.encode()).hexdigest(),
        }
    return value


def compact_json_section(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rendered = json.dumps(compact(payload), indent=2, ensure_ascii=False)
    return (
        f"## `{relative}` — vista compacta\n\n"
        f"SHA-256 del JSON completo: `{sha256(path)}`. Las listas de más de ocho "
        "elementos conservan longitud, dos primeros y dos últimos elementos; "
        "esta vista no sustituye al artefacto para verificaciones cuantitativas.\n\n"
        f"```json\n{numbered(rendered)}\n```\n"
    )


def render_dossier(spec: dict[str, Any]) -> str:
    paths = list(spec.get("full", []))
    for pattern in spec.get("globs", []):
        paths.extend(str(path.relative_to(ROOT)) for path in sorted(ROOT.glob(pattern)))
    all_paths = paths + list(spec.get("symbols", {})) + list(spec.get("compact_json", []))
    forbidden = [path for path in all_paths if any(term in path for term in EXCLUDED)]
    if forbidden:
        raise ValueError(f"blind-audit exclusions violated: {forbidden}")

    parts = [
        f"# {spec['title']}\n",
        f"## Objeto de revisión\n\n{spec['questions']}\n",
        "## Condiciones del contexto\n\n"
        "- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.\n"
        "- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.\n"
        "- Los extractos Python omiten funciones operativas sin semántica científica directa.\n"
        "- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.\n",
    ]
    parts.extend(full_section(path) for path in paths)
    parts.extend(symbol_section(path, names) for path, names in spec.get("symbols", {}).items())
    parts.extend(compact_json_section(path) for path in spec.get("compact_json", []))
    text = "\n".join(parts).rstrip() + "\n"
    return text


def build(check: bool) -> int:
    failures: list[str] = []
    rows: list[tuple[str, int, int, str]] = []
    for filename, spec in DOSSIERS.items():
        text = render_dossier(spec)
        path = OUT / filename
        digest = hashlib.sha256(text.encode()).hexdigest()
        rows.append((filename, len(text), max(1, len(text) // 4), digest))
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                failures.append(filename)
        else:
            path.write_text(text, encoding="utf-8")

    index = [
        "# Índice de contextos ciegos",
        "",
        "| Archivo | Caracteres | Tokens estimados | SHA-256 |",
        "| --- | ---: | ---: | --- |",
        *[f"| `{name}` | {chars} | {tokens} | `{digest}` |" for name, chars, tokens, digest in rows],
        "",
        "La estimación usa cuatro caracteres por token y solo sirve para elegir el límite de subida.",
        "",
    ]
    index_text = "\n".join(index)
    index_path = OUT / "03_CONTEXT_INDEX.md"
    if check:
        if not index_path.is_file() or index_path.read_text(encoding="utf-8") != index_text:
            failures.append(index_path.name)
    else:
        index_path.write_text(index_text, encoding="utf-8")

    if failures:
        raise SystemExit("outdated generated contexts: " + ", ".join(failures))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated contexts are stale")
    args = parser.parse_args()
    return build(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
