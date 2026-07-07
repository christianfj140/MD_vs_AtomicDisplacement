"""ML vs SIESTA benchmark toolkit.

Small, dependency-light infrastructure for a benchmark flow that compares
Graph2Mat / DeepH matrix predictions (and their finite-difference derivatives)
against SIESTA references. This package is deliberately *infrastructure only*:
it never launches SIESTA, never trains a model, and never needs a GPU.

Everything here reuses the existing ``shared/`` FDF helpers where possible and
exposes clean, testable APIs. Heavy model inference and real SIESTA parsing are
delegated to the already-existing scripts in ``Comparison/scripts`` through thin
adapters (or explicit, well-signposted stubs).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from .config import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkConfigError,
    load_benchmark_config,
    parse_benchmark_config,
)
from .structure import (  # noqa: E402
    BenchmarkStructure,
    StructureError,
    find_central_atom,
    make_displaced_structures,
    make_supercell,
    structure_from_fdf,
)
from .matrices import (  # noqa: E402
    MatrixCompatibilityError,
    MatrixData,
    MatrixErrorSummary,
    compute_error_by_species_pair,
    compute_matrix_error,
    load_siesta_matrices,
    validate_matrix_compatible,
)
from .predictors import (  # noqa: E402
    DeepHPredictor,
    FunctionMatrixPredictor,
    Graph2MatPredictor,
    MatrixPredictor,
)
from .compare import (  # noqa: E402
    compare_derivatives_to_siesta,
    compare_model_to_siesta,
    finite_difference_matrix_derivative,
    torch_finite_difference_matrix_derivative,
)
from .dataset_mixing import (  # noqa: E402
    classify_dataset_by_size,
    generate_mixed_dataset_configs,
    make_mixed_dataset_manifest,
)
from .species_transfer import (  # noqa: E402
    SpeciesSupportReport,
    inspect_species_support,
    load_species_transfer_config,
    prepare_species_expansion,
)
from .viewer import (  # noqa: E402
    build_derivative_viewer_payload,
    build_matrix_viewer_payload,
    prepare_matrix_plot_payload,
)
from .fdf_io import generate_siesta_displacement_inputs  # noqa: E402
from .pipeline import benchmark_dry_run  # noqa: E402
from .mixed_dataset_materialize import (  # noqa: E402
    DEFAULT_SPLIT_FRACTIONS,
    DatasetCompatibilityError,
    DatasetMaterializeError,
    dataset_atom_count,
    fixed_common_test_ids,
    materialize_mixed_dataset,
    read_dataset_samples,
    validate_datasets_compatible,
)
from .mixing_sweep import (  # noqa: E402
    discover_dataset_sizes,
    plan_mixing_sweep,
    plan_mixing_sweep_from_roots,
    reserved_small_ids_by_size_for_fixed_common_test,
    run_mixing_sweep,
)
from .plot_mixing_mae_vs_size import (  # noqa: E402
    aggregate_mae_vs_size,
    build_mae_vs_size_from_sweep,
    plot_mae_vs_size,
    write_mae_vs_size_outputs,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkConfigError",
    "load_benchmark_config",
    "parse_benchmark_config",
    "BenchmarkStructure",
    "StructureError",
    "find_central_atom",
    "make_displaced_structures",
    "make_supercell",
    "structure_from_fdf",
    "MatrixCompatibilityError",
    "MatrixData",
    "MatrixErrorSummary",
    "compute_error_by_species_pair",
    "compute_matrix_error",
    "load_siesta_matrices",
    "validate_matrix_compatible",
    "DeepHPredictor",
    "FunctionMatrixPredictor",
    "Graph2MatPredictor",
    "MatrixPredictor",
    "compare_derivatives_to_siesta",
    "compare_model_to_siesta",
    "finite_difference_matrix_derivative",
    "torch_finite_difference_matrix_derivative",
    "classify_dataset_by_size",
    "generate_mixed_dataset_configs",
    "make_mixed_dataset_manifest",
    "SpeciesSupportReport",
    "inspect_species_support",
    "load_species_transfer_config",
    "prepare_species_expansion",
    "build_derivative_viewer_payload",
    "build_matrix_viewer_payload",
    "prepare_matrix_plot_payload",
    "generate_siesta_displacement_inputs",
    "benchmark_dry_run",
    "DEFAULT_SPLIT_FRACTIONS",
    "DatasetCompatibilityError",
    "DatasetMaterializeError",
    "dataset_atom_count",
    "fixed_common_test_ids",
    "materialize_mixed_dataset",
    "read_dataset_samples",
    "reserved_small_ids_by_size_for_fixed_common_test",
    "validate_datasets_compatible",
    "discover_dataset_sizes",
    "plan_mixing_sweep",
    "plan_mixing_sweep_from_roots",
    "run_mixing_sweep",
    "aggregate_mae_vs_size",
    "build_mae_vs_size_from_sweep",
    "plot_mae_vs_size",
    "write_mae_vs_size_outputs",
]
