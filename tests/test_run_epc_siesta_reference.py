"""C09 / E-F_001-S11: the SIESTA reference campaign of graphene.

What is exercised here is everything that decides *what SIESTA is asked to do*
and *what may be believed afterwards* -- the plan, the identity of the physics
across branches, the FC layer, and the fail-closed certification -- without
running SIESTA. The SCF itself is covered by the campaign's own manifest, and
the last test reads that manifest when it exists.

The plan is checked against the repository's real graphene material and against
``build_hamiltonian_derivative_stencils.displaced_positions``: the one-hot
branch must still be the historical single-coordinate stencil, not a parallel
one.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import fd_perturbation_space as fdp  # noqa: E402
from build_hamiltonian_derivative_stencils import displaced_positions  # noqa: E402
from run_epc_siesta_reference import (  # noqa: E402
    BRANCH_EQUILIBRIUM,
    BRANCH_FC,
    BRANCH_FD,
    DEFAULT_MATERIAL_FDF,
    DEFAULT_SIESTA_COMMAND,
    _required_artifacts,
    backend_preflight,
    canonical_ang_base_fdf,
    certification_set,
    physics_fingerprint,
    plan_runs,
    read_system_label,
    write_fc_layer,
)

CAMPAIGN_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"

BASE_FDF = """\
SystemName   graphene
SystemLabel  graphene
NumberOfAtoms 2
AtomicCoordinatesFormat Ang
MeshCutoff             600.0 Ry
DM.Tolerance           1.d-4
TS.HS.Save                       T
%block AtomicCoordinatesAndAtomicSpecies
 0.000000  0.000000  0.000000  1
 1.467000  0.000000  0.000000  1
%endblock AtomicCoordinatesAndAtomicSpecies
%block PAO.Basis
C           1
 n=2   0   1
%endblock PAO.Basis
MD.TypeOfRun CG
MD.NumCGsteps 0
"""


def _positions() -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [1.467, 0.0, 0.0]]


# --------------------------------------------------------------------------- #
# The physics has to be the same in every branch
# --------------------------------------------------------------------------- #


class TestPhysicsFingerprint:
    def test_geometry_label_and_md_layer_do_not_change_it(self) -> None:
        moved = BASE_FDF.replace("1.467000", "1.477000").replace("SystemLabel  graphene", "SystemLabel  fd_x_plus")
        moved = moved.replace("MD.TypeOfRun CG", "MD.TypeOfRun FC").replace("MD.NumCGsteps 0", "FC.Displacement 0.01 Ang")
        assert physics_fingerprint(moved) == physics_fingerprint(BASE_FDF)

    def test_repeating_a_directive_with_the_same_value_does_not_change_it(self) -> None:
        # Appending the FC layer restates TS.HS.Save T; fdf resolves that to the
        # value it already had.
        assert physics_fingerprint(BASE_FDF + "TS.HS.Save T\n") == physics_fingerprint(BASE_FDF)

    def test_repeating_a_directive_with_a_different_value_does_change_it(self) -> None:
        assert physics_fingerprint(BASE_FDF + "TS.HS.Save F\n") != physics_fingerprint(BASE_FDF)

    @pytest.mark.parametrize(
        "replacement",
        [
            ("MeshCutoff             600.0 Ry", "MeshCutoff             300.0 Ry"),
            ("DM.Tolerance           1.d-4", "DM.Tolerance           1.d-6"),
            ("C           1", "C           2"),
        ],
    )
    def test_real_physics_changes_it(self, replacement: tuple[str, str]) -> None:
        changed = BASE_FDF.replace(*replacement)
        assert changed != BASE_FDF
        assert physics_fingerprint(changed) != physics_fingerprint(BASE_FDF)


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


class TestPlan:
    def test_covers_equilibrium_one_fc_per_delta_and_both_signs(self) -> None:
        directions = fdp.default_direction_set(2)
        deltas = [0.005, 0.01, 0.02]
        runs = plan_runs(_positions(), directions=directions, delta_ang_values=deltas)

        assert len({run.run_id for run in runs}) == len(runs)
        assert [run.run_id for run in runs if run.branch == BRANCH_EQUILIBRIUM] == [BRANCH_EQUILIBRIUM]
        # The FC half-step *is* the amplitude, so a swept delta needs a swept FC.
        assert sorted(run.delta_ang for run in runs if run.branch == BRANCH_FC) == deltas
        fd_runs = [run for run in runs if run.branch == BRANCH_FD]
        assert len(fd_runs) == len(directions) * len(deltas) * 2
        assert {run.sign for run in fd_runs} == {1, -1}

    def test_one_hot_branch_is_the_legacy_single_coordinate_stencil(self) -> None:
        direction = fdp.one_hot(2, 0, "x")
        runs = plan_runs(_positions(), directions=[direction], delta_ang_values=[0.01])
        plus = next(run for run in runs if run.sign == 1)
        assert plus.positions_ang == displaced_positions(
            _positions(), atom_index_zero_based=0, axis_index=0, signed_delta=0.01
        )

    def test_uniform_translation_moves_every_atom_by_the_same_vector(self) -> None:
        direction = fdp.uniform_translation(2, "x")
        runs = plan_runs(_positions(), directions=[direction], delta_ang_values=[0.02])
        plus = next(run for run in runs if run.sign == 1)
        steps = [
            [moved - original for moved, original in zip(row, base)]
            for row, base in zip(plus.positions_ang, _positions())
        ]
        assert steps[0] == pytest.approx(steps[1])


# --------------------------------------------------------------------------- #
# The fdf that is actually written
# --------------------------------------------------------------------------- #


class TestGeneratedInputs:
    def test_fractional_material_becomes_an_ang_base_the_stencil_layer_accepts(self, tmp_path: Path) -> None:
        sisl = pytest.importorskip("sisl")
        from fdf_materialization import extract_fdf_structure

        base = canonical_ang_base_fdf(DEFAULT_MATERIAL_FDF, tmp_path / "base_ang.fdf")
        structure = extract_fdf_structure(base)  # refuses anything but Ang
        reference = sisl.get_sile(str(DEFAULT_MATERIAL_FDF)).read_geometry()
        assert structure.atom_count == len(reference)
        assert [list(position) for position in structure.positions_ang] == pytest.approx(reference.xyz)
        assert [list(vector) for vector in structure.lattice_vectors_ang] == pytest.approx(reference.lattice.cell)
        assert structure.atom_species == [1, 1]

    def test_fc_layer_leaves_exactly_one_md_type_of_run(self, tmp_path: Path) -> None:
        run_fdf = tmp_path / "RUN.fdf"
        run_fdf.write_text(BASE_FDF, encoding="utf-8")
        write_fc_layer(
            run_fdf,
            displacement_ang=0.01,
            atom_count=2,
            tolerances={"dhdr": "0.0 Ry/Bohr", "dsdr": "0.0 1/Bohr"},
        )
        text = run_fdf.read_text(encoding="utf-8")
        directives = [line.split()[0].lower() for line in text.splitlines() if line.strip()]
        # fdf honours the first occurrence: a leftover MD.TypeOfRun CG from the
        # single-point materialisation would silently cancel the FC run.
        assert directives.count("md.typeofrun") == 1
        assert "MD.TypeOfRun                     FC" in text
        assert "FC.Save.dHS                      T" in text
        assert "FC.Displacement                  0.01 Ang" in text
        assert "FC.First                         1" in text
        assert "FC.Last                          2" in text
        assert read_system_label(run_fdf) == "graphene"

    def test_fc_displacement_is_the_swept_delta(self, tmp_path: Path) -> None:
        for delta in (0.005, 0.01, 0.02):
            run_fdf = tmp_path / f"RUN_{delta}.fdf"
            run_fdf.write_text(BASE_FDF, encoding="utf-8")
            write_fc_layer(
                run_fdf,
                displacement_ang=delta,
                atom_count=2,
                tolerances={"dhdr": "0.0 Ry/Bohr", "dsdr": "0.0 1/Bohr"},
            )
            assert f"FC.Displacement                  {delta:g} Ang" in run_fdf.read_text(encoding="utf-8")


class TestRequiredArtifacts:
    def test_single_point_branches_require_the_double_precision_tshs(self, tmp_path: Path) -> None:
        (tmp_path / "run.HSX").write_bytes(b"")
        # HSX is single precision; the C01 follow-up list forbids it as the sole
        # elementwise FD reference, so its presence certifies nothing.
        assert _required_artifacts(BRANCH_FD, tmp_path, "run") == ["run.TSHS", "run.ORB_INDX"]
        (tmp_path / "run.TSHS").write_bytes(b"")
        (tmp_path / "run.ORB_INDX").write_bytes(b"")
        assert _required_artifacts(BRANCH_FD, tmp_path, "run") == []

    def test_fc_branch_requires_the_derivative_file(self, tmp_path: Path) -> None:
        (tmp_path / "run.FC").write_bytes(b"")
        (tmp_path / "run.ORB_INDX").write_bytes(b"")
        assert _required_artifacts(BRANCH_FC, tmp_path, "run") == ["*dHSdR.nc"]
        (tmp_path / "run.dHSdR.nc").write_bytes(b"")
        assert _required_artifacts(BRANCH_FC, tmp_path, "run") == []


# --------------------------------------------------------------------------- #
# Certification is fail-closed
# --------------------------------------------------------------------------- #


def _rows(*, failing: str = "") -> list[dict]:
    directions = fdp.default_direction_set(2)
    rows = []
    for spec in plan_runs(_positions(), directions=directions, delta_ang_values=[0.01, 0.02]):
        rows.append({**spec.to_metadata(), "certified": spec.run_id != failing})
    return rows


class TestCertificationSet:
    def test_every_delta_is_comparable_when_everything_converged(self) -> None:
        result = certification_set(_rows(), delta_ang_values=[0.01, 0.02])
        assert result["all_deltas_comparable"]
        assert result["uncertified_run_ids"] == []
        assert all(pair["ready_for_central_difference"] for pair in result["fd_pairs"])

    def test_one_unconverged_member_removes_its_pair(self) -> None:
        failing = "fd_atom0000_x_d0p01_minus"
        result = certification_set(_rows(failing=failing), delta_ang_values=[0.01, 0.02])
        broken = next(pair for pair in result["fd_pairs"] if pair["minus_run_id"] == failing)
        assert broken["plus_certified"] and not broken["minus_certified"]
        assert not broken["ready_for_central_difference"]
        assert failing in result["uncertified_run_ids"]
        # The other directions at the same delta survive; only the pair is lost.
        at_delta = next(entry for entry in result["per_delta"] if entry["delta_ang"] == 0.01)
        assert at_delta["ready_fd_pair_count"] == 2
        assert "atom0000_x" not in at_delta["ready_fd_directions"]

    def test_a_delta_without_its_fc_run_is_not_comparable(self) -> None:
        result = certification_set(_rows(failing="fc_dhs_d0p02"), delta_ang_values=[0.01, 0.02])
        entry = next(item for item in result["per_delta"] if item["delta_ang"] == 0.02)
        assert not entry["fc_dhs_available"]
        assert not entry["fc_vs_fd_comparable"]
        assert not result["all_deltas_comparable"]


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    shutil.which(DEFAULT_SIESTA_COMMAND) is None, reason="the pinned SIESTA runtime is not installed here"
)
def test_backend_preflight_records_why_the_scf_runs_on_cpu() -> None:
    preflight = backend_preflight(DEFAULT_SIESTA_COMMAND)
    assert preflight["probe_returncode"] == 0
    assert preflight["requested_backend"] == "gpu_if_available"
    assert preflight["effective_backend"] == "cpu"
    # The evidence is the binary's own banner, not an assumption about SIESTA.
    assert preflight["gpu_build_tokens_found"] == []
    assert any("Version" in line for line in preflight["version_banner"])


# --------------------------------------------------------------------------- #
# The campaign that was actually run, if its artifacts are still here
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (CAMPAIGN_ROOT / "epc_siesta_reference_manifest.json").is_file(),
    reason="no graphene SIESTA reference campaign in this checkout",
)
def test_recorded_campaign_certified_only_converged_runs() -> None:
    manifest = json.loads((CAMPAIGN_ROOT / "epc_siesta_reference_manifest.json").read_text(encoding="utf-8"))
    assert manifest["physics_identical_across_branches"]
    assert manifest["topology_preserved"]
    assert manifest["compute_policy"]["effective_backend"] == "cpu"
    assert manifest["siesta_runtime_ref"]["classification"] in {"VERIFIED_SOURCE", "VERIFIED_BINARY_ONLY"}
    for row in manifest["rows"]:
        if not row.get("certified"):
            continue
        status = row["siesta_output_status"]
        assert status["scf_converged"] and status["job_completed"], row["run_id"]
        assert not row["missing_artifacts"], row["run_id"]
    for entry in manifest["certification_set"]["per_delta"]:
        assert entry["fc_vs_fd_comparable"], entry
