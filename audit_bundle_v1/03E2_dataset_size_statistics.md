# Dossier 4B — Dataset-size minimum y estadística

## Objeto de revisión

Auditar agregación de réplicas y seeds, definición de N mínimo, umbrales, sensibilidad, fits, extrapolación y estabilidad predictiva.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `Comparison/config/graphene_dataset_size_minimum_paper_threshold_protocol.json`

SHA-256: `25799cb79d5befcdb8193ecabdbaa962d90d490921d1c8aa6180f7ed9ae0ef51`

```json
00001 | {
00002 |   "metric": "h_mae_eV_mean",
00003 |   "threshold_mev": 10.0,
00004 |   "physical_rationale": "Paper-audit Hamiltonian MAE threshold for the locked Graphene W90 Graph2Mat-vs-DeepH scaling protocol. This threshold is documented explicitly for publication-facing N_min analysis and is not an exploratory UI preset.",
00005 |   "reference": "graphene_dataset_size_minimum_paper_protocol_v1",
00006 |   "reference_type": "internal_protocol",
00007 |   "validation_scope": "documented_internal_paper_audit_protocol",
00008 |   "material_scope": [
00009 |     "graphene_w90"
00010 |   ],
00011 |   "metric_scope": [
00012 |     "h_mae_eV_mean"
00013 |   ],
00014 |   "applies_to_metrics": [
00015 |     "h_mae_eV_mean"
00016 |   ],
00017 |   "recommended_sensitivity_thresholds_mev": [
00018 |     8.0,
00019 |     10.0,
00020 |     12.0
00021 |   ],
00022 |   "sensitivity_recommendation": "Audit the lower, main, and upper Hamiltonian MAE thresholds before treating nominal N_min as a paper-level claim."
00023 | }
```

## `Comparison/config/graphene_dataset_size_minimum_paper_audit_payload.json`

SHA-256: `ade3414088c3f01619f9d6a342b7da557779da42be4df9c72cd2da7176894381`

```json
00001 | {
00002 |   "primary_metric": "h_mae_eV_mean",
00003 |   "threshold_mev": 10.0,
00004 |   "threshold_preset_key": "h_mae_relaxed_10",
00005 |   "threshold_is_user_defined": false,
00006 |   "threshold_protocol_file": "Comparison/config/graphene_dataset_size_minimum_paper_threshold_protocol.json",
00007 |   "relative_tolerance": 0.05,
00008 |   "plateau_gain": 0.05,
00009 |   "x_axis": "n_train",
00010 |   "aggregation_mode": "mean_seeds_per_config",
00011 |   "cost_basis": "protocol_total",
00012 |   "claim_mode": "paper_candidate",
00013 |   "fit_models": "linear,quadratic,inverse,inverse_square,power_law_floor",
00014 |   "n_min_source": "fit",
00015 |   "n_min_fit_model": "power_law_floor",
00016 |   "moving_average_window": 3,
00017 |   "bootstrap_replicates": 2000,
00018 |   "bootstrap_seed": 12345,
00019 |   "ci_level": 0.95,
00020 |   "run_roots": [
00021 |     "Comparison/results/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_20260610_122311",
00022 |     "Comparison/results/graphene_w90_snapshot_scaling_1100_1300_4seeds_followup/graphene_w90_snapshot_scaling_1100_1300_4seeds_followup_20260612_104920"
00023 |   ]
00024 | }
```

## `Comparison/scripts/g2m_deeph_dataset_size_minimum.py` — extractos seleccionados

SHA-256 del archivo completo: `236979bdad073b8ffabeeefcf225bfd163f961d7d269a731a30652358f5ba0c0`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Estimate minimum dataset size from existing Graph2Mat-vs-DeepH metrics.
00003 | 
00004 | This is a read-only post-processing script. It does not train, predict, run
00005 | SIESTA, run Graph2Mat, run DeepH, or materialize new Hamiltonians. It consumes
00006 | already-written metric tables/JSON files and writes derived CSV/JSON/plots.
00007 | 
00008 | Nominal N vs effective N (N_eff):
00009 |     N_min is computed on nominal train counts (N_train) from manifests or metric
00010 |     rows. MD trajectory snapshots can be temporally autocorrelated, so those
00011 |     counts overstate independent samples. When temporal metadata and a cheap
00012 |     per-snapshot scalar series are available, this script also reports a
00013 |     diagnostic N_eff ≈ N / statistical_inefficiency with
00014 |     statistical_inefficiency = 1 + 2 Σ ρ(k) (positive lags only), without changing
00015 |     the main N_min fits.
00016 |     Interpret N_min cautiously when N_eff ≪ N or when autocorrelation diagnostics
00017 |     are unavailable. See Comparison/scripts/DATASET_SIZE_MINIMUM.md.
00018 | """
00019 | 
00020 | from __future__ import annotations
00021 | 
00022 | import argparse
00023 | import csv
00024 | import functools
00025 | import json
00026 | import math
00027 | import random
00028 | import re
00029 | import statistics
00030 | import sys
00031 | import time
00032 | from collections import defaultdict
00033 | from pathlib import Path
00034 | from typing import Any, Iterable
00035 | 
00036 | try:
00037 |     import numpy as np
00038 | except ImportError:  # pragma: no cover - optional dependency path
00039 |     np = None  # type: ignore[assignment]
00040 | 
00041 | 
00042 | REPO_ROOT = Path(__file__).resolve().parents[2]
00043 | 
00044 | DEFAULT_PRIMARY_METRIC = "h_mae_eV_mean"
00045 | DEFAULT_FIT_MODELS = "linear,quadratic,inverse,inverse_square,power_law_floor"
00046 | CANONICAL_POWER_LAW_MODEL = "power_law_floor"
00047 | MIN_FIT_POINTS_FOR_PAPER_CANDIDATE = 5
00048 | POWER_LAW_LEGACY_ALIASES = frozenset({"power_law"})
00049 | UNCONSTRAINED_FIT_MODELS = {"linear", "quadratic", "inverse", "inverse_square"}
00050 | DIAGNOSTIC_ONLY_FIT_MODELS = {
00051 |     *UNCONSTRAINED_FIT_MODELS,
00052 |     "moving_average",
00053 |     "lowess_logx",
00054 |     "lowess_logx_robust",
00055 |     "monotone_lowess_logx",
00056 |     "cumulative_best",
00057 |     "none",
00058 | }
00059 | CURVE_POINT_FIT_MODELS = {
00060 |     "lowess_logx",
00061 |     "lowess_logx_robust",
00062 |     "monotone_lowess_logx",
00063 |     "moving_average",
00064 |     "cumulative_best",
00065 | }
00066 | NONNEG_PREDICTION_TOL = 1e-9
00067 | DIAGNOSTIC_FIT_CONDITION_WARN = 1e8
00068 | DIAGNOSTIC_FIT_CONDITION_UNSTABLE = 1e12
00069 | POWER_LAW_ALPHA_MIN = 0.05
00070 | POWER_LAW_ALPHA_MAX = 4.0
00071 | POWER_LAW_ALPHA_GRID_POINTS = 160
00072 | POWER_LAW_ALPHA_REFINE_MAX_ITER = 48
00073 | POWER_LAW_ALPHA_REFINE_TOL = 1e-4
00074 | DEFAULT_AGGREGATION_MODE = "mean_replicates"
00075 | AGGREGATION_MODES = (
00076 |     "mean_replicates",
00077 |     "best_config",
00078 |     "mean_seeds_per_config",
00079 |     "best_config_mean",
00080 | )
00081 | COST_BASES = (
00082 |     "per_seed_mean",
00083 |     "protocol_total",
00084 | )
00085 | CLAIM_MODES = (
00086 |     "diagnostic",
00087 |     "paper_candidate",
00088 | )
00089 | PAPER_READY_AGGREGATION_MODE = "mean_seeds_per_config"
00090 | BASE_CONFIG_SEED_SUFFIX = re.compile(r"-seed\d+$", re.IGNORECASE)
00091 | EXPLICIT_BASE_CONFIG_ID_FIELDS = (
00092 |     "base_config_id",
00093 |     "config_family_id",
00094 |     "parent_config_id",
00095 | )
00096 | BASE_CONFIG_ID_FIELDS = (*EXPLICIT_BASE_CONFIG_ID_FIELDS, "selected_config_id")
00097 | DEFAULT_BOOTSTRAP_SEED = 12345
00098 | DEFAULT_CI_LEVEL = 0.95
00099 | N_MIN_REL_TOL_KEY = "N_min_rel_tol"
00100 | LEGACY_N_MIN_REL95_KEY = "N_min_rel95"
00101 | LEGACY_THRESHOLD_ALIASES = {LEGACY_N_MIN_REL95_KEY: N_MIN_REL_TOL_KEY}
00102 | BOOTSTRAP_N_MIN_CRITERIA = ("N_min_abs", N_MIN_REL_TOL_KEY, "N_min_plateau")
00103 | MIN_BOOTSTRAP_SUCCESS_FOR_CI = 2
00104 | REPLICATE_BOOTSTRAP_LABEL = "replicate-resampling CI"
00105 | N_MIN_COST_EFF_BOOTSTRAP_POLICY = "excluded_no_joint_metric_cost_resampling"
00106 | N_MIN_COST_EFF_BOOTSTRAP_REASON = (
00107 |     "N_min_cost_eff is excluded from replicate-resampling CI because this diagnostic "
00108 |     "does not jointly resample cost and metric under the selected cost_basis."
00109 | )
00110 | N_MIN_COST_EFF_DIAGNOSTIC_LABEL = "observed_cost_error_behavior_only"
00111 | N_MIN_COST_EFF_DIAGNOSTIC_NOTE = (
00112 |     "N_min_cost_eff is a diagnostic based on observed cost-error behavior under the selected cost_basis; "
00113 |     "it is not a joint cost-error uncertainty estimate."
00114 | )
00115 | HIERARCHICAL_UNCERTAINTY_LABEL = "hierarchical uncertainty (paper-readiness audit)"
00116 | HIERARCHICAL_UNCERTAINTY_REPLICATES = 200
00117 | DEFAULT_THRESHOLD_REFERENCE = "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets"
00118 | THRESHOLD_BASIS_EXPLORATORY_PRESET = "metric_specific_exploratory_preset"
00119 | THRESHOLD_BASIS_USER_DEFINED = "user_defined_exploratory"
00120 | THRESHOLD_BASIS_EXPLICIT_PROTOCOL = "explicit_threshold_publication_protocol"
00121 | THRESHOLD_MANUAL_PRESET_KEY = "manual"
00122 | THRESHOLD_PROTOCOL_REFERENCE_TYPES = {
00123 |     "internal_protocol",
00124 |     "external_publication",
00125 |     "experimental_validation",
00126 |     "other_documented_protocol",
00127 |     "unspecified_documented_protocol",
00128 | }
00129 | THRESHOLD_SENSITIVITY_MAX_STEP_MULTIPLIER = 1.0
00130 | DATASET_MINIMUM_THRESHOLD_PRESETS: dict[str, list[dict[str, Any]]] = {
00131 |     "h_mae_eV_mean": [
00132 |         {
00133 |             "key": "h_mae_relaxed_10",
00134 |             "threshold_mev": 10.0,
00135 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00136 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00137 |             "interpretation": "Exploratory absolute H-MAE target in meV; not universal or paper-justified by itself.",
00138 |             "metric_family": "hamiltonian_element_error_mev",
00139 |             "paper_justified": False,
00140 |         },
00141 |         {
00142 |             "key": "h_mae_relaxed_20",
00143 |             "threshold_mev": 20.0,
00144 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00145 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00146 |             "interpretation": "Looser exploratory H-MAE threshold for internal scans; not a universal physical criterion.",
00147 |             "metric_family": "hamiltonian_element_error_mev",
00148 |             "paper_justified": False,
00149 |         },
00150 |     ],
00151 |     "h_rmse_eV": [
00152 |         {
00153 |             "key": "h_rmse_relaxed_15",
00154 |             "threshold_mev": 15.0,
00155 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00156 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00157 |             "interpretation": "Exploratory H-RMSE target in meV; chosen as a metric-specific internal protocol, not a universal claim threshold.",
00158 |             "metric_family": "hamiltonian_element_error_mev",
00159 |             "paper_justified": False,
00160 |         },
00161 |         {
00162 |             "key": "h_rmse_relaxed_25",
00163 |             "threshold_mev": 25.0,
00164 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00165 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00166 |             "interpretation": "Looser exploratory H-RMSE threshold for sweep triage; not paper-ready on its own.",
00167 |             "metric_family": "hamiltonian_element_error_mev",
00168 |             "paper_justified": False,
00169 |         },
00170 |     ],
00171 |     "low_energy_rmse_eV": [
00172 |         {
00173 |             "key": "low_energy_rmse_exploratory_20",
00174 |             "threshold_mev": 20.0,
00175 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00176 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00177 |             "interpretation": "Exploratory low-energy spectral RMSE target; metric-specific default, not a universal meV rule.",
00178 |             "metric_family": "spectral_error_mev",
00179 |             "paper_justified": False,
00180 |         },
00181 |         {
00182 |             "key": "low_energy_rmse_exploratory_40",
00183 |             "threshold_mev": 40.0,
00184 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00185 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00186 |             "interpretation": "Looser exploratory low-energy spectral RMSE threshold for internal scans.",
00187 |             "metric_family": "spectral_error_mev",
00188 |             "paper_justified": False,
00189 |         },
00190 |     ],
00191 |     "fermi_window_rmse_eV": [
00192 |         {
00193 |             "key": "fermi_window_rmse_exploratory_15",
00194 |             "threshold_mev": 15.0,
00195 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00196 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00197 |             "interpretation": "Exploratory Fermi-window spectral RMSE target; metric-specific and not universally transferable.",
00198 |             "metric_family": "spectral_error_mev",
00199 |             "paper_justified": False,
00200 |         },
00201 |         {
00202 |             "key": "fermi_window_rmse_exploratory_30",
00203 |             "threshold_mev": 30.0,
00204 |             "basis": THRESHOLD_BASIS_EXPLORATORY_PRESET,
00205 |             "reference": DEFAULT_THRESHOLD_REFERENCE,
00206 |             "interpretation": "Looser exploratory Fermi-window RMSE threshold for internal comparisons.",
00207 |             "metric_family": "spectral_error_mev",
00208 |             "paper_justified": False,
00209 |         },
00210 |     ],
00211 | }
00212 | ENERGY_METRICS_WITHOUT_EV = {"dos_mae_500_fermi_window"}
00213 | _REFERENCED_METRIC_CACHE: dict[tuple[str, str], float | None] = {}
00214 | 
00215 | FORBIDDEN_COMPUTE_COMMANDS = (
00216 |     "deeph-train",
00217 |     "deeph-preprocess",
00218 |     "deeph-inference",
00219 |     "graph2mat fit",
00220 |     "graph2mat test",
00221 |     "graph2mat predict",
00222 |     "siesta",
00223 |     "gnubands",
00224 | )
00225 | 
00226 | METHOD_COLORS = {
00227 |     "deeph": "#d62728",
00228 |     "graph2mat": "#1f77b4",
00229 | }
00230 | 
```

### `mean` — líneas 443–445

```py
00443 | def mean(values: Iterable[float]) -> float | None:
00444 |     clean = [value for value in values if math.isfinite(value)]
00445 |     return sum(clean) / len(clean) if clean else None
```

### `std` — líneas 448–452

```py
00448 | def std(values: Iterable[float]) -> float | None:
00449 |     clean = [value for value in values if math.isfinite(value)]
00450 |     if not clean:
00451 |         return None
00452 |     return statistics.stdev(clean) if len(clean) > 1 else 0.0
```

### `sem` — líneas 455–461

```py
00455 | def sem(values: Iterable[float]) -> float | None:
00456 |     clean = [value for value in values if math.isfinite(value)]
00457 |     if not clean:
00458 |         return None
00459 |     if len(clean) == 1:
00460 |         return 0.0
00461 |     return statistics.stdev(clean) / math.sqrt(len(clean))
```

### `resolve_threshold_protocol` — líneas 556–653

```py
00556 | def resolve_threshold_protocol(
00557 |     *,
00558 |     primary_metric: str,
00559 |     threshold_mev: float,
00560 |     threshold_protocol_file: str | None = None,
00561 | ) -> dict[str, Any] | None:
00562 |     file_value = str(threshold_protocol_file or "").strip()
00563 |     if not file_value:
00564 |         return None
00565 |     path = Path(file_value).expanduser().resolve()
00566 |     payload = load_threshold_protocol(path)
00567 |     protocol_metric = str(payload.get("metric") or "").strip()
00568 |     if protocol_metric and protocol_metric != primary_metric:
00569 |         raise ValueError(
00570 |             f"threshold protocol metric mismatch: expected {primary_metric}, got {protocol_metric}"
00571 |         )
00572 |     protocol_threshold = finite_number(payload.get("threshold_mev"))
00573 |     if protocol_threshold is None:
00574 |         raise ValueError(f"threshold protocol missing numeric threshold_mev: {path}")
00575 |     if abs(protocol_threshold - float(threshold_mev)) >= 1e-9:
00576 |         raise ValueError(
00577 |             f"threshold protocol threshold mismatch: expected {threshold_mev:g}, got {protocol_threshold:g}"
00578 |         )
00579 |     rationale = str(payload.get("physical_rationale") or payload.get("rationale") or "").strip()
00580 |     if not rationale:
00581 |         raise ValueError(f"threshold protocol missing physical_rationale: {path}")
00582 |     reference = str(payload.get("reference") or "").strip()
00583 |     if not reference:
00584 |         raise ValueError(f"threshold protocol missing reference: {path}")
00585 |     reference_type = str(payload.get("reference_type") or "unspecified_documented_protocol").strip()
00586 |     if not reference_type:
00587 |         reference_type = "unspecified_documented_protocol"
00588 |     if reference_type not in THRESHOLD_PROTOCOL_REFERENCE_TYPES:
00589 |         raise ValueError(
00590 |             "threshold protocol reference_type must be one of: "
00591 |             + ", ".join(sorted(THRESHOLD_PROTOCOL_REFERENCE_TYPES))
00592 |             + f": {path}"
00593 |         )
00594 |     applicability = payload.get("applies_to_metrics")
00595 |     if applicability is None:
00596 |         applicability = [primary_metric]
00597 |     elif isinstance(applicability, str):
00598 |         applicability = [applicability]
00599 |     elif isinstance(applicability, list):
00600 |         applicability = [str(item).strip() for item in applicability if str(item).strip()]
00601 |     else:
00602 |         raise ValueError(f"threshold protocol applies_to_metrics must be a string or list: {path}")
00603 |     if primary_metric not in applicability:
00604 |         raise ValueError(
00605 |             f"threshold protocol does not declare applicability to metric {primary_metric}: {path}"
00606 |         )
00607 |     sensitivity_values = payload.get("recommended_sensitivity_thresholds_mev")
00608 |     thresholds_mev: list[float] = []
00609 |     if sensitivity_values is None:
00610 |         thresholds_mev = [float(protocol_threshold)]
00611 |     elif isinstance(sensitivity_values, list):
00612 |         for item in sensitivity_values:
00613 |             value = finite_number(item)
00614 |             if value is None:
00615 |                 raise ValueError(f"invalid sensitivity threshold in protocol: {item!r}")
00616 |             thresholds_mev.append(float(value))
00617 |     else:
00618 |         raise ValueError(f"recommended_sensitivity_thresholds_mev must be a list: {path}")
00619 |     thresholds_mev.append(float(protocol_threshold))
00620 |     thresholds_mev = sorted({round(value, 12) for value in thresholds_mev})
00621 |     sensitivity_recommendation = str(
00622 |         payload.get("sensitivity_recommendation")
00623 |         or "Check N_min sensitivity across the documented threshold range before paper-level use."
00624 |     ).strip()
00625 |     validation_scope = normalize_threshold_protocol_scope(
00626 |         payload.get("validation_scope"),
00627 |         field_name="validation_scope",
00628 |         path=path,
00629 |     )
00630 |     material_scope = normalize_threshold_protocol_scope(
00631 |         payload.get("material_scope"),
00632 |         field_name="material_scope",
00633 |         path=path,
00634 |     )
00635 |     metric_scope = normalize_threshold_protocol_scope(
00636 |         payload.get("metric_scope"),
00637 |         field_name="metric_scope",
00638 |         path=path,
00639 |     ) or list(applicability)
00640 |     return {
00641 |         "threshold_protocol_file": str(path),
00642 |         "threshold_protocol_metric": primary_metric,
00643 |         "threshold_protocol_threshold_mev": float(protocol_threshold),
00644 |         "threshold_protocol_physical_rationale": rationale,
00645 |         "threshold_protocol_reference": reference,
00646 |         "threshold_protocol_reference_type": reference_type,
00647 |         "threshold_protocol_validation_scope": validation_scope,
00648 |         "threshold_protocol_material_scope": material_scope,
00649 |         "threshold_protocol_metric_scope": metric_scope,
00650 |         "threshold_protocol_applies_to_metrics": applicability,
00651 |         "threshold_protocol_sensitivity_recommendation": sensitivity_recommendation,
00652 |         "threshold_protocol_sensitivity_thresholds_mev": thresholds_mev,
00653 |     }
```

### `row_dataset_sizes` — líneas 933–959

```py
00933 | def row_dataset_sizes(row: dict[str, Any]) -> tuple[int | None, int | None, bool, list[str]]:
00934 |     warnings: list[str] = []
00935 |     n_total = (
00936 |         int_number(row.get("n_total"))
00937 |         or int_number(row.get("dataset_size"))
00938 |         or int_number(row.get("total_snapshots"))
00939 |         or int_number(row.get("valid_snapshots"))
00940 |     )
00941 |     n_train = int_number(row.get("n_train")) or int_number(row.get("train_dataset_size")) or int_number(row.get("train_size"))
00942 |     n_train_explicit = n_train is not None
00943 | 
00944 |     dataset_root = row.get("dataset_root") or row.get("reference_dir") or row.get("reference_root")
00945 |     if n_total is None:
00946 |         n_total = dataset_size_from_root(dataset_root)
00947 |     if n_train is None:
00948 |         counts = split_counts_from_dataset_root(dataset_root)
00949 |         n_train = counts.get("train") or counts.get("training")
00950 |         n_train_explicit = n_train is not None
00951 |     if n_total is None:
00952 |         for key in ("dataset_root", "reference_dir", "metrics_root", "run_dir", "prediction_dir", "dataset_id"):
00953 |             n_total = infer_size_from_text(row.get(key))
00954 |             if n_total is not None:
00955 |                 break
00956 |     if n_train is None and n_total is not None:
00957 |         warnings.append(f"missing_train_split_count_for_size_{n_total}; using n_total as n_train fallback")
00958 |         n_train = n_total
00959 |     return n_total, n_train, n_train_explicit, warnings
```

### `normalize_rows` — líneas 1126–1180

```py
01126 | def normalize_rows(
01127 |     raw_rows: list[dict[str, Any]],
01128 |     *,
01129 |     primary_metric: str,
01130 |     x_axis: str,
01131 | ) -> tuple[list[dict[str, Any]], list[str]]:
01132 |     scale, unit = metric_scale(primary_metric)
01133 |     rows: list[dict[str, Any]] = []
01134 |     warnings: list[str] = []
01135 |     for row in raw_rows:
01136 |         value_e = metric_value(row, primary_metric)
01137 |         n_total, n_train, n_train_explicit, row_warnings = row_dataset_sizes(row)
01138 |         if x_axis == "n_train":
01139 |             warnings.extend(row_warnings)
01140 |         x_value = n_train if x_axis == "n_train" else n_total
01141 |         method = model_key(row.get("method") or row.get("model"))
01142 |         config_id = str(row.get("selected_config_id") or row.get("config_id") or row.get("config_hash") or "unknown")
01143 |         epoch_label = str(row.get("epoch_label") or (f"{row.get('epochs')} epochs" if row.get("epochs") not in (None, "") else "unknown"))
01144 |         if value_e is None:
01145 |             warnings.append(f"missing_primary_metric:{primary_metric}:{method}:{config_id}")
01146 |             continue
01147 |         if x_value is None:
01148 |             warnings.append(f"missing_dataset_size:{method}:{config_id}")
01149 |             continue
01150 |         rows.append(
01151 |             {
01152 |                 "source_run_root": row.get("source_run_root"),
01153 |                 "source_metric_file": row.get("source_metric_file"),
01154 |                 "method": method,
01155 |                 "dataset_id": row.get("dataset_id") or "",
01156 |                 "dataset_root": row.get("dataset_root") or row.get("reference_dir") or "",
01157 |                 "dataset_size_total": n_total,
01158 |                 "dataset_size_train": n_train,
01159 |                 "dataset_size_train_explicit": n_train_explicit,
01160 |                 "dataset_size_train_source": "explicit" if n_train_explicit else "n_total_fallback",
01161 |                 "dataset_size_x": int(x_value),
01162 |                 "x_axis": x_axis,
01163 |                 "config_id": config_id,
01164 |                 "base_config_id": row.get("base_config_id") or "",
01165 |                 "config_family_id": row.get("config_family_id") or "",
01166 |                 "parent_config_id": row.get("parent_config_id") or "",
01167 |                 "selected_config_id": row.get("selected_config_id") or "",
01168 |                 "config_hash": row.get("config_hash") or "",
01169 |                 "seed": row.get("seed") or "unknown",
01170 |                 "epochs": row.get("epochs") or "",
01171 |                 "epoch_label": epoch_label,
01172 |                 "primary_metric": primary_metric,
01173 |                 "primary_metric_raw": value_e,
01174 |                 "primary_metric_mev": value_e * scale,
01175 |                 "primary_metric_unit": unit,
01176 |                 "gpu_hours_total": cost_value(row),
01177 |                 "elapsed_seconds": elapsed_value(row),
01178 |             }
01179 |         )
01180 |     return rows, warnings
```

### `aggregate_rows_mean_replicates` — líneas 1305–1366

```py
01305 | def aggregate_rows_mean_replicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
01306 |     """Average replicate rows for each (method, dataset_size_x)."""
01307 |     grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
01308 |     for row in rows:
01309 |         metric = row_primary_metric_mev(row)
01310 |         size = int_number(row.get("dataset_size_x"))
01311 |         if metric is None or size is None:
01312 |             continue
01313 |         grouped[(str(row["method"]), int(size))].append(row)
01314 | 
01315 |     out: list[dict[str, Any]] = []
01316 |     for (method, size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
01317 |         metric_values = [
01318 |             float(metric)
01319 |             for item in items
01320 |             if (metric := row_primary_metric_mev(item)) is not None
01321 |         ]
01322 |         costs = [
01323 |             value
01324 |             for item in items
01325 |             if (value := finite_number(item.get("gpu_hours_total_mean") or item.get("gpu_hours_total"))) is not None
01326 |         ]
01327 |         elapsed = [
01328 |             value
01329 |             for item in items
01330 |             if (value := finite_number(item.get("elapsed_seconds_mean") or item.get("elapsed_seconds"))) is not None
01331 |         ]
01332 |         first = items[0]
01333 |         config_ids = sorted({str(item.get("config_id") or "") for item in items if item.get("config_id")})
01334 |         seeds = sorted({str(item.get("seed") or "") for item in items if item.get("seed") not in (None, "", "unknown")})
01335 |         source_roots = sorted(
01336 |             {str(item.get("source_run_root") or "") for item in items if item.get("source_run_root")}
01337 |         )
01338 |         out.append(
01339 |             {
01340 |                 "method": method,
01341 |                 "dataset_size_x": size,
01342 |                 "dataset_size_total": first.get("dataset_size_total"),
01343 |                 "dataset_size_train": first.get("dataset_size_train"),
01344 |                 "config_id": "aggregated_mean",
01345 |                 "epoch_label": f"mean of {len(items)} replicate(s)",
01346 |                 "primary_metric": first.get("primary_metric"),
01347 |                 "primary_metric_unit": first.get("primary_metric_unit"),
01348 |                 "primary_metric_mev_mean": mean(metric_values),
01349 |                 "primary_metric_mev_std": std(metric_values),
01350 |                 "primary_metric_mev_sem": sem(metric_values),
01351 |                 "replicate_count": len(items),
01352 |                 "y_min": min(metric_values) if metric_values else None,
01353 |                 "y_max": max(metric_values) if metric_values else None,
01354 |                 "config_ids": config_ids,
01355 |                 "seeds": seeds,
01356 |                 "source_run_roots": source_roots,
01357 |                 "gpu_hours_total_mean": mean(costs),
01358 |                 "gpu_hours_per_seed_mean": mean(costs),
01359 |                 "gpu_hours_protocol_total": sum(costs) if costs else None,
01360 |                 "gpu_hours_protocol_sem": sem(costs),
01361 |                 "elapsed_seconds_mean": mean(elapsed),
01362 |                 "is_aggregated_mean": True,
01363 |                 "aggregation_mode": "mean_replicates",
01364 |             }
01365 |         )
01366 |     return out
```

### `aggregate_rows_mean_seeds_per_config` — líneas 1369–1437

```py
01369 | def aggregate_rows_mean_seeds_per_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
01370 |     """Average seed replicates for each (method, dataset_size_x, base_config_id)."""
01371 |     grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
01372 |     for row in rows:
01373 |         metric = row_primary_metric_mev(row)
01374 |         size = int_number(row.get("dataset_size_x"))
01375 |         if metric is None or size is None:
01376 |             continue
01377 |         base_config_id = extract_base_config_id(row)
01378 |         grouped[(str(row["method"]), int(size), base_config_id)].append(row)
01379 | 
01380 |     out: list[dict[str, Any]] = []
01381 |     for (method, size, base_config_id), items in sorted(
01382 |         grouped.items(),
01383 |         key=lambda item: (item[0][0], item[0][1], item[0][2]),
01384 |     ):
01385 |         metric_values = [
01386 |             float(metric)
01387 |             for item in items
01388 |             if (metric := row_primary_metric_mev(item)) is not None
01389 |         ]
01390 |         costs = [
01391 |             value
01392 |             for item in items
01393 |             if (value := finite_number(item.get("gpu_hours_total_mean") or item.get("gpu_hours_total"))) is not None
01394 |         ]
01395 |         elapsed = [
01396 |             value
01397 |             for item in items
01398 |             if (value := finite_number(item.get("elapsed_seconds_mean") or item.get("elapsed_seconds"))) is not None
01399 |         ]
01400 |         first = items[0]
01401 |         config_ids = sorted({str(item.get("config_id") or "") for item in items if item.get("config_id")})
01402 |         seeds = sorted({str(item.get("seed") or "") for item in items if item.get("seed") not in (None, "", "unknown")})
01403 |         source_roots = sorted(
01404 |             {str(item.get("source_run_root") or "") for item in items if item.get("source_run_root")}
01405 |         )
01406 |         out.append(
01407 |             {
01408 |                 "method": method,
01409 |                 "dataset_size_x": size,
01410 |                 "dataset_size_total": first.get("dataset_size_total"),
01411 |                 "dataset_size_train": first.get("dataset_size_train"),
01412 |                 "base_config_id": base_config_id,
01413 |                 "config_id": base_config_id,
01414 |                 "epoch_label": first.get("epoch_label") or f"mean of {len(items)} seed(s)",
01415 |                 "primary_metric": first.get("primary_metric"),
01416 |                 "primary_metric_unit": first.get("primary_metric_unit"),
01417 |                 "primary_metric_mev_mean": mean(metric_values),
01418 |                 "primary_metric_mev_std": std(metric_values),
01419 |                 "primary_metric_mev_sem": sem(metric_values),
01420 |                 "seed_count": len(seeds) if seeds else len(items),
01421 |                 "seeds": seeds,
01422 |                 "config_ids": config_ids,
01423 |                 "replicate_count": len(items),
01424 |                 "y_min": min(metric_values) if metric_values else None,
01425 |                 "y_max": max(metric_values) if metric_values else None,
01426 |                 "source_run_roots": source_roots,
01427 |                 "gpu_hours_total_mean": mean(costs),
01428 |                 "gpu_hours_per_seed_mean": mean(costs),
01429 |                 "gpu_hours_protocol_total": sum(costs) if costs else None,
01430 |                 "gpu_hours_protocol_sem": sem(costs),
01431 |                 "elapsed_seconds_mean": mean(elapsed),
01432 |                 "is_aggregated_mean": True,
01433 |                 "aggregation_mode": "mean_seeds_per_config",
01434 |                 "selection_basis": "mean_over_seeds",
01435 |             }
01436 |         )
01437 |     return out
```

### `aggregate_rows_best_config_mean` — líneas 1440–1463

```py
01440 | def aggregate_rows_best_config_mean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
01441 |     """Pick the lowest seed-mean config for each (method, dataset_size_x)."""
01442 |     seed_means = aggregate_rows_mean_seeds_per_config(rows)
01443 |     grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
01444 |     for row in seed_means:
01445 |         grouped[(str(row["method"]), int(row["dataset_size_x"]))].append(row)
01446 | 
01447 |     best: list[dict[str, Any]] = []
01448 |     for (_method, _size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
01449 |         chosen = min(
01450 |             items,
01451 |             key=lambda row: (
01452 |                 finite_number(row.get("primary_metric_mev_mean")) or math.inf,
01453 |                 finite_number(row.get("gpu_hours_total_mean")) or math.inf,
01454 |                 str(row.get("base_config_id") or row.get("config_id")),
01455 |             ),
01456 |         )
01457 |         out = dict(chosen)
01458 |         out["is_best_for_method_size"] = True
01459 |         out["aggregation_mode"] = "best_config_mean"
01460 |         out["selection_basis"] = "mean_over_seeds"
01461 |         out["config_id"] = out.get("base_config_id") or out.get("config_id")
01462 |         best.append(out)
01463 |     return best
```

### `analysis_rows_for_aggregation_mode` — líneas 1466–1480

```py
01466 | def analysis_rows_for_aggregation_mode(
01467 |     normalized_rows: list[dict[str, Any]],
01468 |     grouped_rows: list[dict[str, Any]],
01469 |     *,
01470 |     aggregation_mode: str,
01471 | ) -> list[dict[str, Any]]:
01472 |     if aggregation_mode == "best_config":
01473 |         return best_by_method_size(grouped_rows)
01474 |     if aggregation_mode == "mean_replicates":
01475 |         return aggregate_rows_mean_replicates(normalized_rows)
01476 |     if aggregation_mode == "mean_seeds_per_config":
01477 |         return aggregate_rows_mean_seeds_per_config(normalized_rows)
01478 |     if aggregation_mode == "best_config_mean":
01479 |         return aggregate_rows_best_config_mean(normalized_rows)
01480 |     raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")
```

### `best_by_method_size` — líneas 1483–1504

```py
01483 | def best_by_method_size(grouped_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
01484 |     grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
01485 |     for row in grouped_rows:
01486 |         value = finite_number(row.get("primary_metric_mev_mean"))
01487 |         if value is None:
01488 |             continue
01489 |         grouped[(str(row["method"]), int(row["dataset_size_x"]))].append(row)
01490 |     best: list[dict[str, Any]] = []
01491 |     for (_method, _size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
01492 |         chosen = min(
01493 |             items,
01494 |             key=lambda row: (
01495 |                 finite_number(row.get("primary_metric_mev_mean")) or math.inf,
01496 |                 finite_number(row.get("gpu_hours_total_mean")) or math.inf,
01497 |                 str(row.get("config_id")),
01498 |             ),
01499 |         )
01500 |         out = dict(chosen)
01501 |         out["is_best_for_method_size"] = True
01502 |         out["aggregation_mode"] = "best_config"
01503 |         best.append(out)
01504 |     return best
```

### `mean_by_method_size` — líneas 1507–1567

```py
01507 | def mean_by_method_size(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
01508 |     """Average rows across sources for each (method, dataset_size_x)."""
01509 |     grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
01510 |     for row in rows:
01511 |         value = finite_number(row.get("primary_metric_mev_mean"))
01512 |         if value is None:
01513 |             continue
01514 |         grouped[(str(row["method"]), int(row["dataset_size_x"]))].append(row)
01515 |     out: list[dict[str, Any]] = []
01516 |     for (method, size), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
01517 |         metric_values = [
01518 |             value
01519 |             for item in items
01520 |             if (value := finite_number(item.get("primary_metric_mev_mean"))) is not None
01521 |         ]
01522 |         costs = [
01523 |             value
01524 |             for item in items
01525 |             if (value := finite_number(item.get("gpu_hours_total_mean"))) is not None
01526 |         ]
01527 |         first = items[0]
01528 |         source_roots = sorted(
01529 |             {
01530 |                 str(item.get("source_run_root") or "")
01531 |                 for item in items
01532 |                 if item.get("source_run_root")
01533 |             }
01534 |         )
01535 |         sweep_labels = sorted(
01536 |             {
01537 |                 str(item.get("sweep_label") or "")
01538 |                 for item in items
01539 |                 if item.get("sweep_label")
01540 |             }
01541 |         )
01542 |         out.append(
01543 |             {
01544 |                 "method": method,
01545 |                 "dataset_size_x": size,
01546 |                 "dataset_size_total": first.get("dataset_size_total"),
01547 |                 "dataset_size_train": first.get("dataset_size_train"),
01548 |                 "config_id": "aggregated_mean",
01549 |                 "epoch_label": f"mean of {len(items)} source(s)",
01550 |                 "primary_metric": first.get("primary_metric"),
01551 |                 "primary_metric_unit": first.get("primary_metric_unit"),
01552 |                 "primary_metric_mev_mean": mean(metric_values),
01553 |                 "primary_metric_mev_std": std(metric_values),
01554 |                 "gpu_hours_total_mean": mean(costs),
01555 |                 "source_count": len(items),
01556 |                 "source_run_roots": source_roots,
01557 |                 "sweep_labels": sweep_labels,
01558 |                 "sweep_label": (
01559 |                     f"mean ({len(sweep_labels)} sweeps)"
01560 |                     if sweep_labels
01561 |                     else f"mean ({len(items)} sources)"
01562 |                 ),
01563 |                 "source_run_root": source_roots[0] if len(source_roots) == 1 else None,
01564 |                 "is_aggregated_mean": True,
01565 |             }
01566 |         )
01567 |     return out
```

### `n_min_abs` — líneas 1570–1576

```py
01570 | def n_min_abs(best_rows: list[dict[str, Any]], threshold_mev: float) -> int | None:
01571 |     candidates = [
01572 |         int(row["dataset_size_x"])
01573 |         for row in best_rows
01574 |         if (value := finite_number(row.get("primary_metric_mev_mean"))) is not None and value <= threshold_mev
01575 |     ]
01576 |     return min(candidates) if candidates else None
```

### `n_min_rel_tol` — líneas 1579–1591

```py
01579 | def n_min_rel_tol(best_rows: list[dict[str, Any]], relative_tolerance: float) -> int | None:
01580 |     """First N within relative tolerance of the best observed/fitted value."""
01581 |     values = [
01582 |         (int(row["dataset_size_x"]), float(row["primary_metric_mev_mean"]))
01583 |         for row in best_rows
01584 |         if finite_number(row.get("primary_metric_mev_mean")) is not None
01585 |     ]
01586 |     if not values:
01587 |         return None
01588 |     best_observed = min(value for _size, value in values)
01589 |     cutoff = best_observed * (1.0 + relative_tolerance)
01590 |     candidates = [size for size, value in values if value <= cutoff]
01591 |     return min(candidates) if candidates else None
```

### `n_min_rel95` — líneas 1594–1596

```py
01594 | def n_min_rel95(best_rows: list[dict[str, Any]], relative_tolerance: float) -> int | None:
01595 |     """Deprecated alias for n_min_rel_tol; not a 95% confidence quantity."""
01596 |     return n_min_rel_tol(best_rows, relative_tolerance)
```

### `n_min_plateau` — líneas 1607–1626

```py
01607 | def n_min_plateau(best_rows: list[dict[str, Any]], plateau_gain: float) -> int | None:
01608 |     values = sorted(
01609 |         [
01610 |             (int(row["dataset_size_x"]), float(row["primary_metric_mev_mean"]))
01611 |             for row in best_rows
01612 |             if finite_number(row.get("primary_metric_mev_mean")) is not None
01613 |         ]
01614 |     )
01615 |     if not values:
01616 |         return None
01617 |     final_best = min(value for _size, value in values)
01618 |     best_so_far = math.inf
01619 |     for size, value in values:
01620 |         best_so_far = min(best_so_far, value)
01621 |         if best_so_far <= 0.0:
01622 |             return size
01623 |         future_gain_fraction = max(0.0, (best_so_far - final_best) / best_so_far)
01624 |         if future_gain_fraction <= plateau_gain:
01625 |             return size
01626 |     return values[-1][0]
```

### `n_min_cost_eff` — líneas 1629–1658

```py
01629 | def n_min_cost_eff(
01630 |     best_rows: list[dict[str, Any]],
01631 |     relative_tolerance: float,
01632 |     *,
01633 |     cost_basis: str = "per_seed_mean",
01634 | ) -> int | None:
01635 |     values = [
01636 |         row
01637 |         for row in best_rows
01638 |         if finite_number(row.get("primary_metric_mev_mean")) is not None
01639 |         and row_cost_for_basis(row, cost_basis) is not None
01640 |     ]
01641 |     if not values:
01642 |         return None
01643 |     best_observed = min(float(row["primary_metric_mev_mean"]) for row in values)
01644 |     cutoff = best_observed * (1.0 + relative_tolerance)
01645 |     passing = [row for row in values if float(row["primary_metric_mev_mean"]) <= cutoff]
01646 |     if not passing:
01647 |         return None
01648 |     chosen = min(
01649 |         passing,
01650 |         key=lambda row: (
01651 |             row_cost_for_basis(row, cost_basis)
01652 |             if row_cost_for_basis(row, cost_basis) is not None
01653 |             else math.inf,
01654 |             int(row["dataset_size_x"]),
01655 |             finite_number(row.get("primary_metric_mev_mean")) or math.inf,
01656 |         ),
01657 |     )
01658 |     return int(chosen["dataset_size_x"])
```

### `thresholds_by_method` — líneas 1661–1684

```py
01661 | def thresholds_by_method(
01662 |     best_rows: list[dict[str, Any]],
01663 |     *,
01664 |     threshold_mev: float,
01665 |     relative_tolerance: float,
01666 |     plateau_gain: float,
01667 |     cost_basis: str = "per_seed_mean",
01668 | ) -> dict[str, dict[str, Any]]:
01669 |     out: dict[str, dict[str, Any]] = {}
01670 |     for method in sorted({str(row["method"]) for row in best_rows}):
01671 |         rows = sorted([row for row in best_rows if row["method"] == method], key=lambda row: int(row["dataset_size_x"]))
01672 |         values = [finite_number(row.get("primary_metric_mev_mean")) for row in rows]
01673 |         clean_values = [value for value in values if value is not None]
01674 |         out[method] = with_legacy_threshold_aliases({
01675 |             "available_sizes": [int(row["dataset_size_x"]) for row in rows],
01676 |             "best_observed_mev": min(clean_values) if clean_values else None,
01677 |             "N_min_abs": n_min_abs(rows, threshold_mev),
01678 |             N_MIN_REL_TOL_KEY: n_min_rel_tol(rows, relative_tolerance),
01679 |             "N_min_plateau": n_min_plateau(rows, plateau_gain),
01680 |             "N_min_cost_eff": n_min_cost_eff(rows, relative_tolerance, cost_basis=cost_basis),
01681 |             "N_min_cost_eff_basis": cost_basis,
01682 |             "N_min_cost_eff_basis_label": cost_basis_label(cost_basis),
01683 |         })
01684 |     return out
```

### `threshold_sensitivity_summary` — líneas 1813–1937

```py
01813 | def threshold_sensitivity_summary(
01814 |     best_rows: list[dict[str, Any]],
01815 |     *,
01816 |     threshold_values_mev: list[float],
01817 |     main_threshold_mev: float,
01818 |     relative_tolerance: float,
01819 |     plateau_gain: float,
01820 |     cost_basis: str,
01821 |     n_min_source: str,
01822 |     fit_model: str,
01823 |     moving_average_window: int,
01824 |     claim_mode: str = "diagnostic",
01825 | ) -> dict[str, Any]:
01826 |     unique_thresholds = sorted({round(float(value), 12) for value in threshold_values_mev if finite_number(value) is not None})
01827 |     if not unique_thresholds:
01828 |         return {
01829 |             "enabled": False,
01830 |             "status": "not_requested",
01831 |             "thresholds_mev": [],
01832 |             "by_method": {},
01833 |             "paper_level_blockers": [],
01834 |             "warnings": [],
01835 |         }
01836 |     claim_mode = parse_claim_mode(claim_mode)
01837 |     by_method: dict[str, dict[str, Any]] = {}
01838 |     warnings: list[str] = []
01839 |     methods = sorted({str(row["method"]) for row in best_rows})
01840 |     per_threshold_results: dict[float, dict[str, dict[str, Any]]] = {}
01841 |     for threshold_mev in unique_thresholds:
01842 |         if n_min_source == "fit":
01843 |             threshold_map, _fit_details, fit_warnings = thresholds_by_method_from_fit(
01844 |                 best_rows,
01845 |                 threshold_mev=threshold_mev,
01846 |                 relative_tolerance=relative_tolerance,
01847 |                 plateau_gain=plateau_gain,
01848 |                 fit_model=fit_model,
01849 |                 moving_average_window=moving_average_window,
01850 |                 cost_basis=cost_basis,
01851 |             )
01852 |             warnings.extend(fit_warnings)
01853 |         else:
01854 |             threshold_map = thresholds_by_method(
01855 |                 best_rows,
01856 |                 threshold_mev=threshold_mev,
01857 |                 relative_tolerance=relative_tolerance,
01858 |                 plateau_gain=plateau_gain,
01859 |                 cost_basis=cost_basis,
01860 |             )
01861 |         per_threshold_results[threshold_mev] = threshold_map
01862 |     blockers: list[str] = []
01863 |     has_below_main = any(value < float(main_threshold_mev) for value in unique_thresholds)
01864 |     has_main = any(abs(value - float(main_threshold_mev)) < 1e-9 for value in unique_thresholds)
01865 |     has_above_main = any(value > float(main_threshold_mev) for value in unique_thresholds)
01866 |     sufficient_range = (
01867 |         len(unique_thresholds) >= 3
01868 |         and has_below_main
01869 |         and has_main
01870 |         and has_above_main
01871 |     )
01872 |     if claim_mode == "paper_candidate" and not sufficient_range:
01873 |         blockers.append("paper_blocked_if_threshold_sensitivity_insufficient_range")
01874 |     for method in methods:
01875 |         series: list[dict[str, Any]] = []
01876 |         n_values: list[int] = []
01877 |         available_sizes: list[int] = []
01878 |         missing_threshold_crossings: list[float] = []
01879 |         for threshold_mev in unique_thresholds:
01880 |             method_thresholds = per_threshold_results.get(threshold_mev, {}).get(method) or {}
01881 |             n_min_abs_value = int_number(method_thresholds.get("N_min_abs"))
01882 |             if not available_sizes:
01883 |                 available_sizes = [
01884 |                     int(size) for size in (method_thresholds.get("available_sizes") or []) if int_number(size) is not None
01885 |                 ]
01886 |             n_min_rel_tol_value = int_number(
01887 |                 method_thresholds.get(N_MIN_REL_TOL_KEY)
01888 |                 or method_thresholds.get(LEGACY_N_MIN_REL95_KEY)
01889 |             )
01890 |             n_min_plateau_value = int_number(method_thresholds.get("N_min_plateau"))
01891 |             series.append({
01892 |                 "threshold_mev": threshold_mev,
01893 |                 "N_min_abs": n_min_abs_value,
01894 |                 N_MIN_REL_TOL_KEY: n_min_rel_tol_value,
01895 |                 "N_min_plateau": n_min_plateau_value,
01896 |                 "source": method_thresholds.get("N_min_abs_source"),
01897 |             })
01898 |             if n_min_abs_value is not None:
01899 |                 n_values.append(n_min_abs_value)
01900 |             else:
01901 |                 missing_threshold_crossings.append(float(threshold_mev))
01902 |         observed_steps = sorted({
01903 |             later - earlier
01904 |             for earlier, later in zip(available_sizes, available_sizes[1:])
01905 |             if later > earlier
01906 |         })
01907 |         allowed_delta = observed_steps[0] if observed_steps else None
01908 |         span = (max(n_values) - min(n_values)) if n_values else None
01909 |         unstable = (
01910 |             allowed_delta is not None
01911 |             and span is not None
01912 |             and span > allowed_delta * THRESHOLD_SENSITIVITY_MAX_STEP_MULTIPLIER
01913 |         )
01914 |         method_blockers: list[str] = []
01915 |         if missing_threshold_crossings:
01916 |             method_blockers.append(f"paper_blocked_if_threshold_sensitivity_missing_n_min_abs:{method}")
01917 |         if unstable:
01918 |             method_blockers.append(f"paper_blocked_if_threshold_sensitivity_unstable:{method}")
01919 |         blockers.extend(method_blockers)
01920 |         by_method[method] = {
01921 |             "threshold_series": series,
01922 |             "missing_threshold_crossings": missing_threshold_crossings,
01923 |             "n_min_abs_span": span,
01924 |             "allowed_n_min_abs_delta": allowed_delta,
01925 |             "unstable": unstable,
01926 |             "paper_level_blockers": method_blockers,
01927 |         }
01928 |     return {
01929 |         "enabled": True,
01930 |         "status": "ok",
01931 |         "thresholds_mev": unique_thresholds,
01932 |         "main_threshold_mev": float(main_threshold_mev),
01933 |         "sufficient_range_for_paper_candidate": sufficient_range,
01934 |         "by_method": by_method,
01935 |         "paper_level_blockers": sorted(set(blockers)),
01936 |         "warnings": sorted(set(warnings)),
01937 |     }
```

### `fit_predictive_stability_by_left_out_N` — líneas 1940–2117

```py
01940 | def fit_predictive_stability_by_left_out_N(
01941 |     best_rows: list[dict[str, Any]],
01942 |     *,
01943 |     threshold_mev: float,
01944 |     relative_tolerance: float,
01945 |     plateau_gain: float,
01946 |     fit_model: str,
01947 |     moving_average_window: int = 3,
01948 |     cost_basis: str = "per_seed_mean",
01949 |     n_min_source: str = "fit",
01950 |     baseline_thresholds: dict[str, dict[str, Any]] | None = None,
01951 |     baseline_fit_details: dict[str, dict[str, Any]] | None = None,
01952 | ) -> dict[str, Any]:
01953 |     canonical_model = canonical_fit_model(fit_model)
01954 |     if n_min_source != "fit":
01955 |         return {
01956 |             "status": "not_applicable",
01957 |             "reason": "observed_only_mode",
01958 |             "fit_model": canonical_model,
01959 |             "methods": {},
01960 |         }
01961 |     if canonical_model == "none":
01962 |         return {
01963 |             "status": "not_applicable",
01964 |             "reason": "no_curve_fit_requested",
01965 |             "fit_model": canonical_model,
01966 |             "methods": {},
01967 |         }
01968 | 
01969 |     if baseline_thresholds is None or baseline_fit_details is None:
01970 |         baseline_thresholds, baseline_fit_details, _ = thresholds_by_method_from_fit(
01971 |             best_rows,
01972 |             threshold_mev=threshold_mev,
01973 |             relative_tolerance=relative_tolerance,
01974 |             plateau_gain=plateau_gain,
01975 |             fit_model=fit_model,
01976 |             moving_average_window=moving_average_window,
01977 |             cost_basis=cost_basis,
01978 |         )
01979 | 
01980 |     methods_out: dict[str, dict[str, Any]] = {}
01981 |     global_blockers: list[str] = []
01982 | 
01983 |     for method in sorted({str(row["method"]) for row in best_rows}):
01984 |         observed_rows = sorted(
01985 |             [row for row in best_rows if str(row["method"]) == method],
01986 |             key=lambda row: int(row["dataset_size_x"]),
01987 |         )
01988 |         observed_sizes = [int(row["dataset_size_x"]) for row in observed_rows]
01989 |         unique_sizes = sorted(dict.fromkeys(observed_sizes))
01990 |         min_observed_size = unique_sizes[0] if unique_sizes else None
01991 |         max_observed_size = unique_sizes[-1] if unique_sizes else None
01992 |         step_sizes = [
01993 |             right - left
01994 |             for left, right in zip(unique_sizes, unique_sizes[1:])
01995 |             if right > left
01996 |         ]
01997 |         one_size_step = min(step_sizes) if step_sizes else None
01998 |         baseline_method_thresholds = dict(baseline_thresholds.get(method) or {})
01999 |         baseline_fit = dict(baseline_fit_details.get(method) or {})
02000 | 
02001 |         trials: list[dict[str, Any]] = []
02002 |         for omitted_size in unique_sizes:
02003 |             reduced_rows = [row for row in observed_rows if int(row["dataset_size_x"]) != omitted_size]
02004 |             trial_thresholds, trial_details, trial_warnings = thresholds_by_method_from_fit(
02005 |                 reduced_rows,
02006 |                 threshold_mev=threshold_mev,
02007 |                 relative_tolerance=relative_tolerance,
02008 |                 plateau_gain=plateau_gain,
02009 |                 fit_model=fit_model,
02010 |                 moving_average_window=moving_average_window,
02011 |                 cost_basis=cost_basis,
02012 |             )
02013 |             trial_fit = dict(trial_details.get(method) or {})
02014 |             trial_threshold = dict(trial_thresholds.get(method) or {})
02015 |             fit_status = str(trial_fit.get("status") or "missing_fit_status")
02016 |             successful = fit_status == "ok" and bool(trial_threshold)
02017 |             trials.append(
02018 |                 {
02019 |                     "omitted_N": omitted_size,
02020 |                     "fit_status": fit_status,
02021 |                     "successful": successful,
02022 |                     "thresholds": {
02023 |                         criterion: trial_threshold.get(criterion)
02024 |                         for criterion in PAPER_RELEVANT_STABILITY_CRITERIA
02025 |                     },
02026 |                     "failure_reason": None if successful else (trial_fit.get("error") or fit_status),
02027 |                     "warnings": trial_warnings,
02028 |                 }
02029 |             )
02030 | 
02031 |         max_abs_delta: dict[str, float | None] = {}
02032 |         max_relative_delta: dict[str, float | None] = {}
02033 |         unstable_criteria: list[str] = []
02034 |         for criterion in PAPER_RELEVANT_STABILITY_CRITERIA:
02035 |             baseline_value = finite_number(baseline_method_thresholds.get(criterion))
02036 |             deltas: list[float] = []
02037 |             rel_deltas: list[float] = []
02038 |             step_deltas: list[int] = []
02039 |             for trial in trials:
02040 |                 trial_value = finite_number((trial.get("thresholds") or {}).get(criterion))
02041 |                 if baseline_value is None or trial_value is None:
02042 |                     continue
02043 |                 if (
02044 |                     min_observed_size is not None
02045 |                     and max_observed_size is not None
02046 |                     and (
02047 |                         baseline_value < min_observed_size
02048 |                         or baseline_value > max_observed_size
02049 |                         or trial_value < min_observed_size
02050 |                         or trial_value > max_observed_size
02051 |                     )
02052 |                 ):
02053 |                     continue
02054 |                 delta = abs(float(trial_value) - float(baseline_value))
02055 |                 deltas.append(delta)
02056 |                 if baseline_value != 0.0:
02057 |                     rel_deltas.append(delta / abs(float(baseline_value)))
02058 |                 if unique_sizes:
02059 |                     baseline_index = min(
02060 |                         range(len(unique_sizes)),
02061 |                         key=lambda idx: abs(float(unique_sizes[idx]) - float(baseline_value)),
02062 |                     )
02063 |                     trial_index = min(
02064 |                         range(len(unique_sizes)),
02065 |                         key=lambda idx: abs(float(unique_sizes[idx]) - float(trial_value)),
02066 |                     )
02067 |                     step_deltas.append(abs(trial_index - baseline_index))
02068 |             max_abs_delta[criterion] = max(deltas) if deltas else None
02069 |             max_relative_delta[criterion] = max(rel_deltas) if rel_deltas else None
02070 |             if step_deltas and max(step_deltas) > 1:
02071 |                 unstable_criteria.append(criterion)
02072 | 
02073 |         n_trials = len(trials)
02074 |         n_successful = sum(1 for trial in trials if trial["successful"])
02075 |         n_failed = n_trials - n_successful
02076 |         failure_threshold = max(1, n_trials // 4) if n_trials else 0
02077 |         unstable_due_to_failures = n_failed > failure_threshold if n_trials else False
02078 |         method_blockers: list[str] = []
02079 |         if unstable_criteria:
02080 |             method_blockers.append(
02081 |                 f"paper_blocked_if_fit_predictive_stability_unstable:{method}:{','.join(sorted(unstable_criteria))}"
02082 |             )
02083 |         if unstable_due_to_failures:
02084 |             method_blockers.append(
02085 |                 f"paper_blocked_if_fit_predictive_stability_leave_one_out_failures:{method}"
02086 |             )
02087 |         global_blockers.extend(method_blockers)
02088 | 
02089 |         methods_out[method] = {
02090 |             "status": "ok",
02091 |             "fit_model": canonical_model,
02092 |             "baseline_fit_status": baseline_fit.get("status"),
02093 |             "baseline_thresholds": {
02094 |                 criterion: baseline_method_thresholds.get(criterion)
02095 |                 for criterion in PAPER_RELEVANT_STABILITY_CRITERIA
02096 |             },
02097 |             "observed_sizes": unique_sizes,
02098 |             "one_observed_size_step": one_size_step,
02099 |             "n_leave_one_out_trials": n_trials,
02100 |             "n_successful": n_successful,
02101 |             "n_failed": n_failed,
02102 |             "max_abs_delta_N_min": max_abs_delta,
02103 |             "max_relative_delta_N_min": max_relative_delta,
02104 |             "unstable_criteria": sorted(set(unstable_criteria)),
02105 |             "unstable_due_to_failures": unstable_due_to_failures,
02106 |             "paper_level_blockers": method_blockers,
02107 |             "failure_threshold": failure_threshold,
02108 |             "trials": trials,
02109 |         }
02110 | 
02111 |     return {
02112 |         "status": "ok",
02113 |         "fit_model": canonical_model,
02114 |         "n_min_source": n_min_source,
02115 |         "paper_level_blockers": sorted(set(global_blockers)),
02116 |         "methods": methods_out,
02117 |     }
```

### `fit_linear_model` — líneas 2324–2350

```py
02324 | def fit_linear_model(model: str, n_values: list[float], y_values: list[float]) -> dict[str, Any]:
02325 |     design = fit_design(model, n_values)
02326 |     coefficients, numerical_meta = least_squares_coefficients_stable(design, y_values)
02327 |     predicted = [sum(coef * item for coef, item in zip(coefficients, row)) for row in design]
02328 |     summary = fit_summary(model, n_values, y_values, predicted, coefficients)
02329 |     summary.update(numerical_meta)
02330 |     summary["scaled_fit_domain"].setdefault(
02331 |         "n_values",
02332 |         {
02333 |             "min": min(float(value) for value in n_values) if n_values else None,
02334 |             "max": max(float(value) for value in n_values) if n_values else None,
02335 |         },
02336 |     )
02337 |     condition_estimate = numerical_meta.get("fit_condition_estimate")
02338 |     effective_rank = numerical_meta.get("effective_rank")
02339 |     if (
02340 |         (effective_rank is not None and effective_rank < len(design[0]))
02341 |         or (
02342 |             condition_estimate is not None
02343 |             and math.isfinite(float(condition_estimate))
02344 |             and float(condition_estimate) >= DIAGNOSTIC_FIT_CONDITION_UNSTABLE
02345 |         )
02346 |     ):
02347 |         summary["status"] = "diagnostic_unstable"
02348 |         summary["error"] = "diagnostic_fit_numerically_unstable"
02349 |         summary["diagnostic_only"] = True
02350 |     return summary
```

### `fit_power_law_floor` — líneas 2538–2638

```py
02538 | def fit_power_law_floor(n_values: list[float], y_values: list[float]) -> dict[str, Any]:
02539 |     """Constrained canonical model: y(N) = E_inf + A * N^(-alpha)."""
02540 |     if len(n_values) < required_fit_points(CANONICAL_POWER_LAW_MODEL):
02541 |         return {
02542 |             "model": CANONICAL_POWER_LAW_MODEL,
02543 |             "status": "skipped_insufficient_points",
02544 |             "n_points": len(n_values),
02545 |             "formula": "y = E_inf + A N^-alpha",
02546 |             **fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="skipped_insufficient_points", n_points=len(n_values)),
02547 |         }
02548 | 
02549 |     n_min = min(n_values)
02550 |     n_max = max(n_values)
02551 |     if n_min <= 0:
02552 |         return {
02553 |             "model": CANONICAL_POWER_LAW_MODEL,
02554 |             "status": "failed_invalid_domain",
02555 |             "n_points": len(n_values),
02556 |             "formula": "y = E_inf + A N^-alpha",
02557 |             **fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="failed_invalid_domain", n_points=len(n_values)),
02558 |         }
02559 | 
02560 |     best: dict[str, Any] | None = None
02561 |     coarse_grid = [
02562 |         POWER_LAW_ALPHA_MIN
02563 |         + (POWER_LAW_ALPHA_MAX - POWER_LAW_ALPHA_MIN) * idx / (POWER_LAW_ALPHA_GRID_POINTS - 1)
02564 |         for idx in range(POWER_LAW_ALPHA_GRID_POINTS)
02565 |     ]
02566 |     best_idx: int | None = None
02567 |     objective_evaluations = 0
02568 |     for idx, alpha in enumerate(coarse_grid):
02569 |         candidate = evaluate_power_law_floor_alpha(alpha, n_values, y_values)
02570 |         objective_evaluations += 1
02571 |         if candidate is None:
02572 |             continue
02573 |         if best is None or candidate["sse"] < best["sse"]:
02574 |             best = candidate
02575 |             best_idx = idx
02576 | 
02577 |     if best is None:
02578 |         return {
02579 |             "model": CANONICAL_POWER_LAW_MODEL,
02580 |             "status": "failed_constraint_violation",
02581 |             "n_points": len(n_values),
02582 |             "formula": "y = E_inf + A N^-alpha",
02583 |             **fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="failed_constraint_violation", n_points=len(n_values)),
02584 |         }
02585 | 
02586 |     alpha_refinement_interval = [best["alpha"], best["alpha"]]
02587 |     if best_idx is not None and len(coarse_grid) >= 2:
02588 |         left_index = max(0, best_idx - 1)
02589 |         right_index = min(len(coarse_grid) - 1, best_idx + 1)
02590 |         refine_left = coarse_grid[left_index]
02591 |         refine_right = coarse_grid[right_index]
02592 |         alpha_refinement_interval = [float(refine_left), float(refine_right)]
02593 |         refined_best, refinement_evaluations = golden_section_refine_power_law_alpha(
02594 |             n_values,
02595 |             y_values,
02596 |             left=refine_left,
02597 |             right=refine_right,
02598 |         )
02599 |         objective_evaluations += refinement_evaluations
02600 |         if refined_best is not None and refined_best["sse"] < best["sse"]:
02601 |             best = refined_best
02602 | 
02603 |     summary = fit_summary(
02604 |         CANONICAL_POWER_LAW_MODEL,
02605 |         n_values,
02606 |         y_values,
02607 |         best["predicted"],
02608 |         best["coefficients"],
02609 |     )
02610 |     summary["status"] = "ok"
02611 |     summary["formula"] = "y = E_inf + A N^-alpha"
02612 |     summary.update(fit_policy_metadata(CANONICAL_POWER_LAW_MODEL, status="ok", n_points=len(n_values)))
02613 |     summary["constraints"] = {
02614 |         "e_inf_nonnegative": best["coefficients"][0] >= 0,
02615 |         "amplitude_nonnegative": best["coefficients"][1] >= 0,
02616 |         "alpha_positive": best["coefficients"][2] > 0,
02617 |         "predictions_nonnegative_on_observed_domain": True,
02618 |     }
02619 |     summary["fit_domain"] = {"min_n": float(n_min), "max_n": float(n_max)}
02620 |     summary["coefficients_named"] = {
02621 |         "e_inf": best["coefficients"][0],
02622 |         "amplitude": best["coefficients"][1],
02623 |         "alpha": best["coefficients"][2],
02624 |     }
02625 |     summary["alpha"] = best["coefficients"][2]
02626 |     summary["sse"] = best["sse"]
02627 |     summary["alpha_search_method"] = "coarse_grid_plus_golden_section"
02628 |     summary["alpha_bounds"] = {
02629 |         "min": POWER_LAW_ALPHA_MIN,
02630 |         "max": POWER_LAW_ALPHA_MAX,
02631 |     }
02632 |     summary["alpha_refinement_interval"] = {
02633 |         "min": alpha_refinement_interval[0],
02634 |         "max": alpha_refinement_interval[1],
02635 |     }
02636 |     summary["objective_evaluations"] = objective_evaluations
02637 |     summary["nonnegative_constraints_active"] = True
02638 |     return summary
```

### `fit_summary` — líneas 3115–3145

```py
03115 | def fit_summary(
03116 |     model: str,
03117 |     n_values: list[float],
03118 |     y_values: list[float],
03119 |     predicted: list[float],
03120 |     coefficients: list[float],
03121 | ) -> dict[str, Any]:
03122 |     residuals = [y - yhat for y, yhat in zip(y_values, predicted)]
03123 |     mae = mean([abs(value) for value in residuals])
03124 |     rmse = math.sqrt(mean([value * value for value in residuals]) or 0.0)
03125 |     y_mean = mean(y_values) or 0.0
03126 |     ss_tot = sum((value - y_mean) ** 2 for value in y_values)
03127 |     ss_res = sum(value * value for value in residuals)
03128 |     r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
03129 |     return {
03130 |         "model": model,
03131 |         "fit_model": model,
03132 |         "status": "ok",
03133 |         "n_points": len(n_values),
03134 |         "coefficients": coefficients,
03135 |         "mae_mev": mae,
03136 |         "rmse_mev": rmse,
03137 |         "r2": r2,
03138 |         "formula": {
03139 |             "linear": "y = a + bN",
03140 |             "quadratic": "y = a + bN + cN^2",
03141 |             "inverse": "y = a + b/N",
03142 |             "inverse_square": "y = a + b/N^2",
03143 |         }.get(model, "y = E_inf + A N^-alpha"),
03144 |         **fit_policy_metadata(model, status="ok", n_points=len(n_values)),
03145 |     }
```

## `Comparison/results/dataset_size_minimum_ui_20260722_122042/threshold_10meV/dataset_size_minimum_summary.json` — vista compacta

SHA-256 del JSON completo: `e21844c8aa67e38ba11486372a61d99d26c593fc40ea66ebbec479e137fdebde`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "status": "ok",
00003 |   "outputs": [
00004 |     "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/results/dataset_size_minimum_ui_20260722_122042/threshold_10meV/dataset_size_minimum_summary.json"
00005 |   ]
00006 | }
```

## `Comparison/results/dataset_size_minimum_ui_20260722_122042/threshold_25meV/dataset_size_minimum_summary.json` — vista compacta

SHA-256 del JSON completo: `08e2c358ce13cb67f94ebb35b0f67c8763190a857c0db68da6eb196dfe9da46a`. Las listas de más de ocho elementos conservan longitud, dos primeros y dos últimos elementos; esta vista no sustituye al artefacto para verificaciones cuantitativas.

```json
00001 | {
00002 |   "status": "ok"
00003 | }
```
