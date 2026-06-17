from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for directory in (SCRIPTS_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_hamiltonian_derivative_stencils import build_derivative_stencils  # noqa: E402
from fdf_materialization import extract_fdf_structure, materialize_sample_fdf  # noqa: E402
from hamiltonian_derivative_stencil import discover_derivative_stencils, validate_derivative_geometry  # noqa: E402
from validate_hamiltonian_derivative_geometry import validate_derivative_geometry_outputs  # noqa: E402


def synthetic_base_fdf() -> str:
    return "\n".join(
        [
            "SystemName synthetic base",
            "SystemLabel shared_label",
            "NumberOfSpecies 1",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 6 C",
            "%endblock ChemicalSpeciesLabel",
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 8.0 0.0 0.0",
            " 0.0 8.0 0.0",
            " 0.0 0.0 8.0",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 1.0 0.0 0.0 1",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "",
        ]
    )


class HamiltonianDerivativeGeometryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "source_dataset"
        self.sample_dir = self.dataset_root / "splits" / "test" / "base_0"
        self.sample_dir.mkdir(parents=True)
        (self.sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(), encoding="utf-8")
        (self.sample_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": "base_0",
                    "material_label": "synthetic",
                    "system_label": "shared_label",
                    "material_compatibility_hash": "material-hash",
                    "orbital_ordering_hash": "orbital-hash",
                    "basis_hash": "basis-hash",
                    "pseudopotential_hash": "pseudo-hash",
                }
            ),
            encoding="utf-8",
        )
        (self.dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps({"rows": [{"sample_id": "base_0", "split": "test", "sample_dir": str(self.sample_dir)}]}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build_stencil(self, *, include_base: bool = True) -> tuple[Path, dict]:
        output_root = self.root / "stencils"
        manifest = build_derivative_stencils(
            source_dataset_root=self.dataset_root,
            output_stencil_root=output_root,
            split="test",
            method="central",
            delta_ang_values=[0.1],
            atom_indices_zero_based=[1],
            axes=["y"],
            include_base=include_base,
        )
        return output_root, manifest

    def discovery(self, output_root: Path):
        discoveries = discover_derivative_stencils(
            output_root,
            method="graph2mat",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )
        self.assertEqual(len(discoveries), 1)
        return discoveries[0]

    def issue_codes(self, output_root: Path) -> set[str]:
        return {issue.code for issue in validate_derivative_geometry(self.discovery(output_root))}

    def rewrite_positions(self, run_fdf: Path, positions: list[tuple[float, float, float]]) -> None:
        structure = extract_fdf_structure(run_fdf)
        materialize_sample_fdf(
            run_fdf,
            run_fdf,
            positions_ang=positions,
            atom_species=structure.atom_species,
            lattice_vectors_ang=structure.lattice_vectors_ang,
            system_label=run_fdf.parent.name,
            system_name=run_fdf.parent.name,
            structure_type=structure.structure_type,
        )

    def sample_record(self, manifest: dict, sign_label: str) -> dict:
        return next(record for record in manifest["samples"] if record.get("sign_label") == sign_label)

    def test_valid_central_geometry_passes_and_writes_outputs(self) -> None:
        output_root, _manifest = self.build_stencil()

        issues = validate_derivative_geometry(self.discovery(output_root))
        summary = validate_derivative_geometry_outputs(output_root, output_dir=output_root / "validation", split="test")

        self.assertEqual(issues, [])
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["errors"], 0)
        self.assertTrue((output_root / "validation" / "derivative_geometry_validation.csv").exists())
        self.assertTrue((output_root / "validation" / "derivative_geometry_validation.json").exists())

    def test_wrong_atom_displaced_fails(self) -> None:
        output_root, manifest = self.build_stencil()
        plus = self.sample_record(manifest, "+")
        self.rewrite_positions(Path(plus["run_fdf"]), [(0.0, 0.1, 0.0), (1.0, 0.0, 0.0)])

        codes = self.issue_codes(output_root)

        self.assertIn("displacement_component_mismatch", codes)
        self.assertIn("unexpected_coordinate_drift", codes)

    def test_wrong_axis_fails(self) -> None:
        output_root, manifest = self.build_stencil()
        plus = self.sample_record(manifest, "+")
        self.rewrite_positions(Path(plus["run_fdf"]), [(0.0, 0.0, 0.0), (1.1, 0.0, 0.0)])

        codes = self.issue_codes(output_root)

        self.assertIn("displacement_component_mismatch", codes)
        self.assertIn("unexpected_coordinate_drift", codes)

    def test_extra_coordinate_drift_fails(self) -> None:
        output_root, manifest = self.build_stencil()
        plus = self.sample_record(manifest, "+")
        self.rewrite_positions(Path(plus["run_fdf"]), [(0.001, 0.0, 0.0), (1.0, 0.1, 0.0)])

        self.assertIn("unexpected_coordinate_drift", self.issue_codes(output_root))

    def test_split_leakage_fails(self) -> None:
        output_root, manifest = self.build_stencil()
        minus = self.sample_record(manifest, "-")
        metadata_path = Path(minus["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["split"] = "train"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        discoveries = discover_derivative_stencils(
            output_root,
            method="graph2mat",
            split="all",
            finite_difference_method="central",
            require_central=True,
        )
        self.assertEqual(len(discoveries), 1)
        codes = {issue.code for issue in validate_derivative_geometry(discoveries[0])}

        self.assertIn("split_mismatch", codes)

    def test_missing_base_structure_fails_clearly(self) -> None:
        output_root, _manifest = self.build_stencil(include_base=False)

        self.assertIn("missing_base_structure", self.issue_codes(output_root))


if __name__ == "__main__":
    unittest.main()
