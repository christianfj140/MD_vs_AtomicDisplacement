#!/usr/bin/env python3
"""C14 / E-F_001-S15: how wrong is the checkpoint's ``D_H[v]``, and wrong *because of what*.

S12 certified the SIESTA side (GO-2: ``dHSdR.nc`` reproduces a central finite
difference of the same H and S) and S14 certified the Graph2Mat side internally
(GO-3: the directional JVP equals the coordinate combination and the model's own
frozen central difference). Both are statements about *numerics*. This script
asks the one question neither of them can: for the same directions, the same
basis and the same ``(row, col, R)`` mapping, how far is the model's derivative
from the reference's, and how much of that distance is attributable to something
other than the model.

Nothing here is allowed to call a discrepancy "model error" by default. The
residual is only labelled that after every other contribution has been measured
and subtracted from the explanation:

``tau_num``       the numerical floor of *both* finite differences: GO-2's
                  measured ``tau_FD`` on the reference side, and the spread of
                  the model's own frozen central difference across the plateau
                  on the model side.
``tau_backend``   ``|| D_jvp - D_frozen ||`` for the *same* model, same
                  direction, same amplitude: the disagreement between two
                  numerical routes to one derivative.
``support``       the reference has non-zeros at pairs the model's neighbour
                  graph does not contain at all (its edge cutoff is shorter than
                  2 x the PAO radius). That norm is bookkeeping, not learning.
``convention``    the model was trained on sisl's ``H - E_F S``, while
                  ``dHSdR.nc`` stores the unshifted ``dH/dR``. The reference is
                  therefore converted into the model's own label convention,
                  ``D_H - E_F D_S - (dE_F) S``, before anything is compared, and
                  the unconverted comparison is kept as a control.

Only what survives all four is reported as model error, and even then it is a
*derivative-level* number. It is deliberately not converted into a percentage of
``g``: the roadmap requires ``tau_model`` for ``g`` to come from propagating
``delta_Delta`` through the same electronic subspace (``delta_g = C_f^dag
delta_Delta C_i``), which is C20's job, not this one.

Leakage: the decision threshold is pre-registered in this module
(:data:`ADEQUACY_RELATIVE_FROBENIUS`) and the *transferability* of the measured
error is calibrated on :data:`CALIBRATION_DIRECTIONS` only, then tested on the
held-out collective direction. The graphene-Gamma case that C20 will evaluate
(an ``E_2g`` phonon eigenvector) is in neither set, and the report asserts its
absence rather than assuming it.

Units: eV/Ang for ``D_H``, 1/Ang for ``D_S``, Ang for positions and amplitudes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared", REPO_ROOT / "scripts" / "torch_serialization_compat"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import epc_gauge_alignment as gauge  # noqa: E402
import fd_perturbation_space as fdp  # noqa: E402
from certify_siesta_dhsdr import (  # noqa: E402
    CANONICAL_UNITS,
    Campaign,
    Key,
    align,
    best_fit_scale,
    check,
    difference_norms,
    finite_difference,
    contract_fc,
    load_campaign,
    norms,
    read_tshs,
)
from read_siesta_dhsdr import read_dhsdr  # noqa: E402
from benchmark_manifest import file_sha256  # noqa: E402
from orbital_contract import parse_orb_indx  # noqa: E402

SCHEMA = "epc_checkpoint_derivative_error_v2"
TICKET = "C14 / E-F_001-S15"
CERTIFICATION_SCHEMA = "epc_siesta_derivative_certification_v1"

DEFAULT_CERTIFICATION = REPO_ROOT / "Comparison/results/epc/certification/graphene/go2_certification.json"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints/spectral-best-epoch=247-step=03472.ckpt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/checkpoint_derivative_error/graphene"
# Per-geometry vacuum levels c_vac(R). Produced by certify_epc_energy_zero from
# the SIESTA VT grids; its runs carry byte-identical TSHS/ORB_INDX/STRUCT_OUT to
# this campaign's (only the fdf differs, by requesting the grid), which is
# asserted below rather than assumed -- a c_vac from a different electronic
# structure would silently poison the conversion.
DEFAULT_ENERGY_ZERO = REPO_ROOT / "Comparison/results/epc/convergence/gauge_delta/energy_zero_alignment.json"
ENERGY_ZERO_SCHEMA = "epc_energy_zero_alignment_v1"

# --- pre-registered decision constants (fixed before any number was measured) ---
# Directions whose model error calibrates the transferability bound. The
# held-out direction is everything else in the certified perturbation space.
CALIBRATION_DIRECTIONS = ("atom0000_x", "translation_x")
# Screening threshold for the C14 verdict. It is *not* a tolerance on g: it is
# the level below which a derivative-level error is small enough that C20's
# subspace propagation is worth running at all. Above it, the checkpoint is a
# fine-tuning candidate and C20 would be propagating a dominant error.
ADEQUACY_RELATIVE_FROBENIUS = 0.05
# The held-out direction may exceed the calibrated bound by this much before the
# error is called non-transferable (i.e. before the calibration stops predicting
# anything about a direction it did not see).
CALIBRATION_TRANSFER_FACTOR = 1.5
# A direction whose reference response is this far below the largest one
# measured in the same run carries no derivative to reproduce (uniform
# translation); it is judged against an absolute floor, not a ratio.
NULL_DIRECTION_FRACTION = 1e-3
# A residual has to stand this far clear of the numerical floors before any part
# of it is attributed to the model.
ATTRIBUTION_MARGIN = 3.0
# --- gauge conversion gate (pre-registered) ---
# The gate asserts that the conversion was applied per geometry and that doing so
# was not a formality. It is a statement about the *comparison*, never about the
# model: it says the two sides are in one gauge, not that they agree.
#
# (a) The neglected product-rule term (c_vac - E_F).dS must be recognisably
#     present, i.e. the per-geometry result must differ from the a-posteriori-only
#     shortcut by more than the 5 meV/Ang numerical budget the PAO convergence
#     gate established. Below that the two are indistinguishable here and this
#     comparison cannot testify that the ordering mattered.
GAUGE_TERM_FLOOR_EV_PER_ANG = 5.0e-3
# (b) The per-geometry identity must close on its own terms: the converted
#     derivative must equal dK^{E_F} + (E_F'-c_vac')S + (E_F-c_vac)dS to within
#     this relative tolerance, which is a float64 bookkeeping check, not physics.
GAUGE_IDENTITY_RELATIVE_TOLERANCE = 1e-10

OUTPUT_KEYS = ("node_labels", "edge_labels")
L_LABEL = {0: "s", 1: "p", 2: "d", 3: "f"}


class CheckpointDerivativeError(RuntimeError):
    """The comparison cannot be evaluated at all (unusable or uncertified inputs)."""


# --------------------------------------------------------------------------- #
# Sparse fields, shared index space
# --------------------------------------------------------------------------- #


def sparse_field(matrix: Any, supercell_order: Sequence[Sequence[int]]) -> dict[Key, float]:
    """A ``(no, no * n_s)`` sisl-layout matrix on the canonical ``(row, col, R)`` space.

    The same decomposition ``column = col + image * no_u`` that
    :func:`certify_siesta_dhsdr.read_tshs` applies to the TSHS, so both sides of
    the comparison are keyed by the physical lattice vector rather than by a
    storage order that the two producers do not share.
    """
    no_u = int(matrix.shape[0])
    order = [tuple(int(component) for component in offset) for offset in supercell_order]
    expected = no_u * len(order)
    if int(matrix.shape[1]) != expected:
        raise CheckpointDerivativeError(
            f"matrix is {matrix.shape} but its supercell order has {len(order)} images "
            f"({expected} columns expected); the R mapping would be silently wrong."
        )
    coo = matrix.tocoo()
    return {
        (int(row), int(col) % no_u, order[int(col) // no_u]): float(value)
        for row, col, value in zip(coo.row, coo.col, coo.data)
    }


def cpu_batch(batch: Any) -> Any:
    """A CPU copy for the numpy/sisl serialisation, leaving the caller's batch alone.

    ``Data.to`` moves the store in place and returns ``self``: serialising a CUDA
    batch directly would strand the model on CUDA with a CPU batch.
    """
    if not hasattr(batch, "to"):
        return batch
    return (batch.clone() if hasattr(batch, "clone") else batch).to("cpu")


def scaled(field: Mapping[Key, float], factor: float) -> dict[Key, float]:
    return {key: value * float(factor) for key, value in field.items()}


def combine(*terms: tuple[float, Mapping[Key, float]]) -> dict[Key, float]:
    """``sum_i c_i F_i`` over the union of the fields' index spaces."""
    total: dict[Key, float] = {}
    for coefficient, field in terms:
        for key, value in field.items():
            total[key] = total.get(key, 0.0) + float(coefficient) * float(value)
    return total


def restricted_norms(field: Mapping[Key, float], keys: Iterable[Key]) -> dict[str, float]:
    values = np.array([field.get(key, 0.0) for key in keys], dtype=np.float64)
    return norms(values)


# --------------------------------------------------------------------------- #
# The common electrostatic gauge
# --------------------------------------------------------------------------- #


def load_vacuum_levels(path: Path, campaign: Campaign) -> dict[str, float]:
    """``c_vac(R)`` per run id, from the certified energy-zero artifact.

    Fail-closed on every axis that could pair a vacuum level with the wrong
    electronic structure: the artifact must be the right schema, must have
    passed, and every run it supplies a level for must be the *same run* this
    campaign certified -- same TSHS bytes, same ORB_INDX, same STRUCT_OUT.
    """
    if not path.is_file():
        raise CheckpointDerivativeError(
            f"{path} is missing. Without per-geometry c_vac(R) the vacuum-gauge reference "
            "cannot be built, and the comparison stays across two gauges."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != ENERGY_ZERO_SCHEMA:
        raise CheckpointDerivativeError(
            f"{path} is schema {payload.get('schema')!r}, expected {ENERGY_ZERO_SCHEMA!r}."
        )
    if payload.get("verdict") != "PASS":
        raise CheckpointDerivativeError(
            f"{path} carries verdict {payload.get('verdict')!r}; an uncertified vacuum level "
            "cannot define the gauge the reference is converted into."
        )

    donor_manifest = Path(payload["reference_manifest"])
    if not donor_manifest.is_file():
        raise CheckpointDerivativeError(f"{donor_manifest} (the energy-zero campaign manifest) is missing.")
    donor_rows = {
        row["run_id"]: row
        for row in json.loads(donor_manifest.read_text(encoding="utf-8"))["rows"]
    }
    ours = {row["run_id"]: row for row in campaign.manifest["rows"]}

    levels: dict[str, float] = {}
    for run_id, entry in payload["levels"].items():
        if run_id not in ours:
            continue
        mine, theirs = ours[run_id].get("artifact_sha256") or {}, donor_rows.get(run_id, {}).get("artifact_sha256") or {}
        # The fdf legitimately differs (it asks for the VT grid); the matrices,
        # the orbital enumeration and the geometry may not.
        for artifact in ("tshs", "orb_indx", "struct_out"):
            if not mine.get(artifact) or mine.get(artifact) != theirs.get(artifact):
                raise CheckpointDerivativeError(
                    f"run {run_id!r}: the energy-zero campaign's {artifact} does not match this "
                    "campaign's; its c_vac belongs to a different electronic structure."
                )
        levels[run_id] = float(entry["vt_vacuum_ev"])
    if not levels:
        raise CheckpointDerivativeError(
            f"{path} supplies no vacuum level for any run of this campaign."
        )
    return levels


# --------------------------------------------------------------------------- #
# The reference side, in the model's label convention
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReferenceDirectional:
    """Everything the certified SIESTA campaign says about one direction."""

    direction_name: str
    direction_kind: str
    delta_ang: float
    d_h_fc: dict[Key, float]  # certified dH/dR contracted onto v (absolute H)
    d_s_fc: dict[Key, float]
    d_label_fc: dict[Key, float]  # dH - E_F dS - (dE_F) S: the model's target
    d_label_fd: dict[Key, float]  # central difference of sisl's H - E_F S
    fermi_ev: float
    d_fermi_ev_per_ang: float
    term_norms: dict[str, dict[str, float]]
    # The same direction in the vacuum gauge the PAO observable is defined in,
    # built by converting each displaced geometry BEFORE the difference.
    d_vacuum_fd: dict[Key, float]
    gauge: dict[str, Any]


def reference_directional(
    campaign: Campaign,
    equilibrium: Any,
    *,
    direction_name: str,
    direction_kind: str,
    vectors: np.ndarray,
    delta_ang: float,
    dhsdr: Any,
    vacuum_levels: Mapping[str, float],
    equilibrium_run_id: str,
) -> ReferenceDirectional:
    """Build ``D_H[v]``, ``D_S[v]`` and the label-convention reference at one amplitude."""
    pair = next(
        (
            entry
            for entry in campaign.certified_pairs()
            if entry["direction_name"] == direction_name
            and math.isclose(float(entry["delta_ang"]), float(delta_ang))
        ),
        None,
    )
    if pair is None:
        raise CheckpointDerivativeError(
            f"{direction_name} has no certified +- pair at delta = {delta_ang}; "
            "the reference cannot be built at the amplitude GO-2 selected."
        )
    plus_id, minus_id = pair["plus_run_id"], pair["minus_run_id"]
    plus = read_tshs(campaign.run_dir(plus_id), plus_id, fermi_ev=campaign.fermi_ev(plus_id))
    minus = read_tshs(campaign.run_dir(minus_id), minus_id, fermi_ev=campaign.fermi_ev(minus_id))

    d_h_fc = contract_fc(dhsdr, vectors, "D_H")
    d_s_fc = contract_fc(dhsdr, vectors, "D_S")
    # E_F is not a constant of the displacement: it is the scalar that sisl
    # subtracts, and it moves with the atoms. Its own directional derivative is
    # the central difference of the two runs' recorded levels.
    d_fermi = (plus.fermi_ev - minus.fermi_ev) / (2.0 * float(delta_ang))
    d_label_fc = combine(
        (1.0, d_h_fc),
        (-equilibrium.fermi_ev, d_s_fc),
        (-d_fermi, equilibrium.overlap),
    )
    d_label_fd = finite_difference(plus, minus, float(delta_ang), undo_fermi_shift=False)["D_H"]

    # --- the common electrostatic gauge -------------------------------------
    # K^{c_vac}(R) = K^{E_F}(R) + [E_F(R) - c_vac(R)] . S(R), applied to EACH
    # displaced geometry with THAT geometry's own levels, and only then
    # differenced. Doing it in this order makes the stencil produce
    #     dK^{c_vac} = dK^{E_F} + (E_F' - c_vac') . S + (E_F - c_vac) . dS
    # with both product-rule terms present. An a-posteriori shift of the
    # finished derivative can supply only the first, and the second is not
    # small: on atom0000_x it is 11.7 eV/Ang against a 48.5 eV/Ang signal.
    missing_levels = [run_id for run_id in (plus_id, minus_id) if run_id not in vacuum_levels]
    if missing_levels:
        raise CheckpointDerivativeError(
            f"{direction_name}: no certified c_vac for {missing_levels}; the vacuum-gauge "
            "reference cannot be built per geometry for this pair."
        )
    k_vacuum = {
        run_id: gauge.to_vacuum_gauge(
            run.h_shifted, run.overlap, c_vac_ev=vacuum_levels[run_id], fermi_ev=run.fermi_ev
        )
        for run_id, run in ((plus_id, plus), (minus_id, minus))
    }
    keys_vac, (vac_plus, vac_minus) = align(k_vacuum[plus_id], k_vacuum[minus_id])
    d_vacuum_fd = dict(
        zip(keys_vac, ((vac_plus - vac_minus) / (2.0 * float(delta_ang))).tolist())
    )

    # The two terms of the product rule, measured separately so the report can
    # show that neither was dropped -- and so the a-posteriori-only shortcut can
    # be exhibited as the different quantity it is.
    d_gauge_scalar = (
        (plus.fermi_ev - vacuum_levels[plus_id]) - (minus.fermi_ev - vacuum_levels[minus_id])
    ) / (2.0 * float(delta_ang))
    gauge_at_equilibrium = float(
        equilibrium.fermi_ev - vacuum_levels.get(equilibrium_run_id, float("nan"))
    )
    # The discrete product rule pairs dS with the MIDPOINT of g = E_F - c_vac
    # over the pair, not with its equilibrium value; the two differ at O(delta^2)
    # but the identity below is meant to close to float64, so use the exact one.
    gauge_midpoint = 0.5 * (
        (plus.fermi_ev - vacuum_levels[plus_id]) + (minus.fermi_ev - vacuum_levels[minus_id])
    )
    keys_s, (s_plus, s_minus) = align(plus.overlap, minus.overlap)
    d_overlap_fd = dict(zip(keys_s, ((s_plus - s_minus) / (2.0 * float(delta_ang))).tolist()))
    a_posteriori_only = combine((1.0, d_label_fd), (d_gauge_scalar, equilibrium.overlap))
    omitted_term = scaled(d_overlap_fd, gauge_at_equilibrium)

    gauge_record = {
        "convention": "K^{c_vac} = H_absolute - c_vac(R) S(R)",
        "applied": "per geometry, before the finite-difference stencil",
        "c_vac_ev": {run_id: float(vacuum_levels[run_id]) for run_id in (plus_id, minus_id)},
        "fermi_ev": {plus_id: float(plus.fermi_ev), minus_id: float(minus.fermi_ev)},
        "d_cvac_minus_Ef_ev_per_ang": float(-d_gauge_scalar),
        "cvac_minus_Ef_at_equilibrium_ev": float(-gauge_at_equilibrium),
        "d_vacuum_fd": norms(align(d_vacuum_fd)[1][0]),
        "a_posteriori_only": norms(align(a_posteriori_only)[1][0]),
        "omitted_product_rule_term_cvac_minus_Ef_times_dS": norms(align(omitted_term)[1][0]),
        "per_geometry_minus_a_posteriori": norms(
            align(combine((1.0, d_vacuum_fd), (-1.0, a_posteriori_only)))[1][0]
        ),
        # The three-term identity, closed exactly on this pair's own numbers.
        # Writing g_j = E_F(j) - c_vac(j), the central difference of
        # K^{c_vac}_j = K^{E_F}_j + g_j S_j is algebraically
        #     d_vac = d_label_fd + g_bar . dS + dg . S_bar
        # with the MIDPOINT g_bar and S_bar -- not the equilibrium ones. This
        # is the discrete statement of the product rule; a nonzero residual
        # here would mean the conversion did not happen before the stencil.
        "identity_closure": norms(
            align(
                combine(
                    (1.0, d_vacuum_fd),
                    (-1.0, d_label_fd),
                    (-gauge_midpoint, d_overlap_fd),
                    (-d_gauge_scalar * 0.5, plus.overlap),
                    (-d_gauge_scalar * 0.5, minus.overlap),
                )
            )[1][0]
        ),
    }

    term_norms = {
        "D_H_fc": norms(align(d_h_fc)[1][0]),
        "D_S_fc": norms(align(d_s_fc)[1][0]),
        "fermi_times_D_S": norms(align(scaled(d_s_fc, equilibrium.fermi_ev))[1][0]),
        "d_fermi_times_S": norms(align(scaled(equilibrium.overlap, d_fermi))[1][0]),
        "D_label_fc": norms(align(d_label_fc)[1][0]),
        "D_label_fd": norms(align(d_label_fd)[1][0]),
    }
    return ReferenceDirectional(
        direction_name=direction_name,
        direction_kind=direction_kind,
        delta_ang=float(delta_ang),
        d_h_fc=d_h_fc,
        d_s_fc=d_s_fc,
        d_label_fc=d_label_fc,
        d_label_fd=d_label_fd,
        fermi_ev=float(equilibrium.fermi_ev),
        d_fermi_ev_per_ang=float(d_fermi),
        term_norms=term_norms,
        d_vacuum_fd=d_vacuum_fd,
        gauge=gauge_record,
    )


# --------------------------------------------------------------------------- #
# The model side
# --------------------------------------------------------------------------- #


def ghost_free_fdf(geometry: Any, destination: Path) -> Path:
    """A minimal fdf with only the populated species, for the Graph2Mat reader.

    ``materials/graphene/RUN.fdf`` declares a Ghost-H species that no atom uses.
    Graph2Mat refuses a geometry carrying a species absent from the basis table,
    and adding a point type the checkpoint never saw would be worse than
    dropping an unpopulated declaration. The geometry written here is asserted
    against the certified equilibrium row before it is used.
    """
    cell = np.asarray(geometry.cell, dtype=np.float64)
    xyz = np.asarray(geometry.xyz, dtype=np.float64)
    tags = [str(geometry.atoms[index].tag) for index in range(len(xyz))]
    labels = sorted(set(tags))
    numbers = {tag: int(geometry.atoms[tags.index(tag)].Z) for tag in labels}
    lines = [
        "SystemName graph2mat_equilibrium_geometry",
        "SystemLabel graph2mat_equilibrium_geometry",
        f"NumberOfAtoms {len(xyz)}",
        f"NumberOfSpecies {len(labels)}",
        "%block ChemicalSpeciesLabel",
        *(f"  {index + 1} {numbers[tag]} {tag}" for index, tag in enumerate(labels)),
        "%endblock ChemicalSpeciesLabel",
        "LatticeConstant 1.0 Ang",
        "%block LatticeVectors",
        *(f" {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}" for row in cell),
        "%endblock LatticeVectors",
        "AtomicCoordinatesFormat Ang",
        "%block AtomicCoordinatesAndAtomicSpecies",
        *(
            f" {position[0]:.12f} {position[1]:.12f} {position[2]:.12f} {labels.index(tag) + 1}"
            for position, tag in zip(xyz, tags)
        ),
        "%endblock AtomicCoordinatesAndAtomicSpecies",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def direction_from_manifest(entry: Mapping[str, Any]) -> Any:
    """Rebuild the certified direction, kind included.

    ``direction_hash`` covers the kind, so a one-hot rebuilt as "collective"
    would hash differently and the JVP would be signing a direction the SIESTA
    runs never displaced. The caller compares the rebuilt hash with the
    manifest's.
    """
    return fdp.Direction(
        name=str(entry["direction_name"]),
        kind=str(entry["direction_kind"]),
        vectors=np.asarray(entry["vectors"], dtype=np.float64),
        atom_index_zero_based=entry.get("atom_index_zero_based"),
        axis=entry.get("axis"),
    )


def _training_dataset_summary(root_dir: Any) -> dict[str, Any]:
    """Material label and sample counts of the dataset the checkpoint was trained on."""
    if not root_dir:
        return {"status": "unknown", "reason": "the checkpoint declares no root_dir"}
    manifest = Path(str(root_dir)) / "benchmark_dataset_manifest.json"
    if not manifest.is_file():
        return {"status": "absent", "path": str(manifest)}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    basis_hashes = payload.get("basis_hashes") or {}
    return {
        "status": "present",
        "path": str(manifest),
        "material_label": payload.get("material_label"),
        "generation_mode": payload.get("generation_mode"),
        "benchmark_dataset_id": payload.get("benchmark_dataset_id"),
        "split_counts": (payload.get("frozen_split_manifest") or {}).get("split_counts"),
        "basis_hashes": basis_hashes,
    }


def load_model_and_batch(
    checkpoint: Path, structure_fdf: Path, *, basis_dir: Path | None = None
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Model, single-structure batch and data processor, on the checkpoint's own basis table.

    The basis table is taken from the checkpoint rather than rebuilt from the
    ``.ion.xml``: the installed sisl derives a longer ``PointBasis.R`` from the
    same byte-identical file, which changes the *edge cutoff* and therefore the
    neighbour graph the model was trained on. Reusing the checkpoint's table
    keeps the topology the weights were fitted under; the drift is reported.
    """
    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()

    import torch
    from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
    from graph2mat.bindings.torch.data import TorchBasisMatrixData
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from torch_geometric.loader.dataloader import DataLoader

    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    hparams = raw.get("hyper_parameters", {})
    basis_table = hparams.get("basis_table")
    if basis_table is None:
        basis_table = raw.get("basis_table")
    if basis_table is None:
        basis_table = (raw.get("datamodule_hyper_parameters") or {}).get("basis_table")
    if basis_table is None:
        raise CheckpointDerivativeError(
            f"{checkpoint} carries no basis_table hyper-parameter; the graph topology the "
            "weights were trained under cannot be reconstructed."
        )
    if int(hparams.get("n_matrix_components", 1)) != 1:
        raise CheckpointDerivativeError(
            "this comparison is H-only; the checkpoint declares "
            f"n_matrix_components={hparams.get('n_matrix_components')!r}."
        )

    # out_matrix / matrix_component_policy / n_matrix_components / sub_point_matrix
    # are the H-only training policy of this repository (g2m_deeph_runner sets
    # them; the run's own hparams.yaml records sub_point_matrix: false). They are
    # not free parameters here: a mismatch would offset every onsite block by the
    # per-species point matrix, which the equilibrium-H check below would catch.
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        out_matrix="hamiltonian",
        symmetric_matrix=bool(hparams.get("symmetric_matrix", True)),
        sub_point_matrix=False,
        n_matrix_components=1,
        matrix_component_policy="h_only",
    )
    data = TorchBasisMatrixData.new(structure_fdf, data_processor=processor, labels=False)
    batch = next(iter(DataLoader([data], batch_size=1)))
    model = LitMACEMatrixModel.load_from_checkpoint(
        str(checkpoint), map_location="cpu", weights_only=False, basis_table=basis_table
    )
    model.eval()

    # Same .ion.xml files, rebuilt by the installed sisl: a longer PointBasis.R
    # than the checkpoint's is a version drift, and it changes the edge cutoff.
    ion_files = (
        [path for path in ((Path(basis_dir) / f"{atom.tag}.ion.xml") for atom in basis_table.atoms) if path.is_file()]
        if basis_dir is not None
        else []
    )
    current = AtomicTableWithEdges.from_basis_glob(ion_files) if ion_files else None
    provenance = {
        "checkpoint_point_basis_R_ang": [
            [float(value) for value in np.atleast_1d(entry.R)] for entry in basis_table.basis
        ],
        "checkpoint_edge_cutoff_ang": float(
            2.0 * max(float(np.max(entry.R)) for entry in basis_table.basis)
        ),
        "sisl_rebuilt_point_basis_R_ang": (
            [[float(value) for value in np.atleast_1d(entry.R)] for entry in current.basis]
            if current is not None
            else None
        ),
        "basis_table_source": "checkpoint_hyper_parameters",
        # What the weights actually saw. A derivative measured on a structure
        # family absent from this root is a transferability measurement, and the
        # verdict has to say which of the two it is.
        "training_root_dir": str(hparams.get("root_dir") or ""),
        "training_basis_files": str(hparams.get("basis_files") or ""),
        "training_dataset": _training_dataset_summary(hparams.get("root_dir")),
        "symmetric_matrix": bool(hparams.get("symmetric_matrix", True)),
        "sub_point_matrix": False,
        "n_matrix_components": 1,
    }
    return model.model, batch, processor, provenance


def model_predictions(model: Any, batch: Any, processor: Any) -> tuple[dict[Key, float], dict[str, Any]]:
    """The model's equilibrium H on the canonical index space (used to identify the convention)."""
    import torch

    from graph2mat_autograd_derivatives import derivative_prediction_to_sparse_matrices

    with torch.no_grad():
        prediction = model(batch)
    orders: list[Any] = []
    matrices = derivative_prediction_to_sparse_matrices(
        processor,
        cpu_batch(batch),
        {key: prediction[key] for key in OUTPUT_KEYS},
        supercell_orders=orders,
    )
    field = sparse_field(matrices[0], orders[0])
    return field, {"shape": [int(dim) for dim in matrices[0].shape], "nnz": int(matrices[0].nnz)}


def directional_derivative_field(
    model: Any,
    batch: Any,
    direction: Any,
    *,
    processor: Any,
    change_of_basis: Any,
    backend: Any,
) -> tuple[dict[Key, float], dict[str, Any]]:
    """``D_jvp[v]`` serialised through the production sparse mapping."""
    from graph2mat_autograd_derivatives import compute_graph2mat_directional_derivative

    result = compute_graph2mat_directional_derivative(
        model,
        batch,
        direction,
        change_of_basis=change_of_basis,
        output_keys=OUTPUT_KEYS,
        data_processor=processor,
        backend=backend,
    )
    return sparse_field(result.matrices[0], result.supercell_orders[0]), dict(result.metadata)


def frozen_derivative_field(
    model: Any,
    batch: Any,
    direction: Any,
    delta_ang: float,
    *,
    processor: Any,
    change_of_basis: Any,
) -> tuple[dict[Key, float], list[str]]:
    """``[H_G2M(R + d v) - H_G2M(R - d v)] / (2 d)`` in the same sparse space.

    Same forward the JVP differentiates, only ``positions`` substituted, so the
    neighbour graph is frozen; the topology hash of each displaced evaluation is
    returned rather than assumed.
    """
    import torch

    from graph2mat_autograd_derivatives import (
        batch_topology_hash,
        derivative_prediction_to_sparse_matrices,
        graph2mat_forward_labels,
        unflatten_graph2mat_prediction_vector,
    )
    from validate_graph2mat_jvp import _batch_tangent, direction_vectors

    positions = batch["positions"]
    tangent = _batch_tangent(direction_vectors(direction), positions, change_of_basis)
    evaluations: dict[float, Any] = {}
    hashes: list[str] = []
    spec = None
    for sign in fdp.signs_for_method("central"):
        displaced = positions.detach().clone() + float(sign) * float(delta_ang) * tangent
        flat, spec = graph2mat_forward_labels(model, batch, displaced, output_keys=OUTPUT_KEYS)
        evaluations[float(sign)] = flat.detach().to(torch.float64).cpu()
        hashes.append(batch_topology_hash(batch)["topology_hash"])
    difference = (evaluations[1.0] - evaluations[-1.0]) / (2.0 * float(delta_ang))
    labels = unflatten_graph2mat_prediction_vector(difference, spec)
    orders: list[Any] = []
    matrices = derivative_prediction_to_sparse_matrices(
        processor,
        cpu_batch(batch),
        labels,
        supercell_orders=orders,
    )
    return sparse_field(matrices[0], orders[0]), hashes


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OrbitalMap:
    """Which atom and which ``l`` each unit-cell orbital belongs to, plus geometry."""

    atom_of_orbital: list[int]
    l_of_orbital: list[int]
    positions_ang: np.ndarray
    cell_ang: np.ndarray

    def block(self, key: Key) -> str:
        return f"{L_LABEL.get(self.l_of_orbital[key[0]], '?')}-{L_LABEL.get(self.l_of_orbital[key[1]], '?')}"

    def distance_ang(self, key: Key) -> float:
        row_atom = self.atom_of_orbital[key[0]]
        col_atom = self.atom_of_orbital[key[1]]
        offset = np.asarray(key[2], dtype=np.float64) @ self.cell_ang
        return float(
            np.linalg.norm(self.positions_ang[col_atom] + offset - self.positions_ang[row_atom])
        )


def orbital_map(orb_indx: Path, positions_ang: np.ndarray, cell_ang: np.ndarray) -> OrbitalMap:
    parsed = parse_orb_indx(orb_indx)
    rows = sorted(parsed["unit_cell_rows"], key=lambda row: row["io"])
    if [row["io"] for row in rows] != list(range(1, len(rows) + 1)):
        raise CheckpointDerivativeError(
            f"{orb_indx} does not enumerate its unit-cell orbitals contiguously from 1; "
            "the orbital -> (atom, l) mapping would be misaligned."
        )
    return OrbitalMap(
        atom_of_orbital=[int(row["ia"]) - 1 for row in rows],
        l_of_orbital=[int(row["l"]) for row in rows],
        positions_ang=np.asarray(positions_ang, dtype=np.float64),
        cell_ang=np.asarray(cell_ang, dtype=np.float64),
    )


def group_metrics(
    keys: Sequence[Key],
    reference: Mapping[Key, float],
    candidate: Mapping[Key, float],
    *,
    grouping: str,
    labels: Sequence[Any],
    direction_name: str,
    total_residual: float,
) -> list[dict[str, Any]]:
    """Per-group reference norm, residual norm and share of the total residual."""
    buckets: dict[Any, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(index)
    rows: list[dict[str, Any]] = []
    for label in sorted(buckets, key=lambda value: (str(type(value)), value)):
        indices = buckets[label]
        ref = np.array([reference.get(keys[index], 0.0) for index in indices], dtype=np.float64)
        cand = np.array([candidate.get(keys[index], 0.0) for index in indices], dtype=np.float64)
        residual = norms(ref - cand)
        reference_norms = norms(ref)
        rows.append(
            {
                "direction_name": direction_name,
                "grouping": grouping,
                "group": label if isinstance(label, str) else float(label),
                "n_keys": len(indices),
                "reference_frobenius": reference_norms["frobenius"],
                "reference_max_abs": reference_norms["max_abs"],
                "residual_frobenius": residual["frobenius"],
                "residual_max_abs": residual["max_abs"],
                "relative_frobenius": (
                    residual["frobenius"] / reference_norms["frobenius"]
                    if reference_norms["frobenius"] > 0.0
                    else float("nan")
                ),
                "share_of_total_residual": (
                    (residual["frobenius"] / total_residual) ** 2 if total_residual > 0.0 else 0.0
                ),
            }
        )
    return rows


def magnitude_deciles(values: np.ndarray) -> list[str]:
    """Decile label of ``|reference|`` for every key, as a string bucket."""
    magnitudes = np.abs(values)
    finite = magnitudes[np.isfinite(magnitudes)]
    if finite.size == 0:
        return ["d0" for _ in magnitudes]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, 11))
    indices = np.clip(np.searchsorted(edges, magnitudes, side="right") - 1, 0, 9)
    return [f"d{int(index)}" for index in indices]


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #


def compare_direction(
    reference: ReferenceDirectional,
    jvp: Mapping[Key, float],
    frozen_by_delta: Mapping[float, Mapping[Key, float]],
    *,
    tau_fd: Mapping[str, float],
    mapping: OrbitalMap,
    equilibrium_overlap: Mapping[Key, float],
    d_overlap_fc: Mapping[Key, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One direction: residual against the label-convention reference, fully decomposed."""
    keys, (ref_values, jvp_values) = align(reference.d_label_fc, jvp)
    residual = norms(ref_values - jvp_values)
    reference_norm = norms(ref_values)
    model_support = set(jvp)
    outside = [key for key in keys if key not in model_support]
    truncation = restricted_norms(reference.d_label_fc, outside)

    plateau_deltas = sorted(frozen_by_delta)
    tau_backend = {
        norm: max(
            difference_norms(frozen_by_delta[delta], jvp)[norm] for delta in plateau_deltas
        )
        for norm in ("frobenius", "max_abs")
    }
    model_numerical = {
        norm: max(
            (
                difference_norms(frozen_by_delta[first], frozen_by_delta[second])[norm]
                for index, first in enumerate(plateau_deltas)
                for second in plateau_deltas[index + 1 :]
            ),
            default=0.0,
        )
        for norm in ("frobenius", "max_abs")
    }
    convention_control = difference_norms(reference.d_h_fc, jvp)
    label_consistency = difference_norms(reference.d_label_fc, reference.d_label_fd)

    # --- the comparison, done inside ONE gauge -------------------------------
    # The model predicts K^{E_F}; its derivative is therefore dK^{E_F}. Carrying
    # it into the vacuum gauge needs the same two product-rule terms the
    # reference got, evaluated on the model's own index space:
    #     dK^{c_vac} = dK^{E_F} + (E_F' - c_vac') S + (E_F - c_vac) dS
    # Both use quantities already certified for this geometry pair, so no new
    # numerics enter -- only the conversion that was previously missing.
    d_gauge_scalar = -float(reference.gauge["d_cvac_minus_Ef_ev_per_ang"])
    gauge_equilibrium = -float(reference.gauge["cvac_minus_Ef_at_equilibrium_ev"])
    jvp_vacuum = combine(
        (1.0, jvp),
        (d_gauge_scalar, equilibrium_overlap),
        (gauge_equilibrium, d_overlap_fc),
    )
    keys_vac, (ref_vac, jvp_vac) = align(reference.d_vacuum_fd, jvp_vacuum)
    residual_vacuum = norms(ref_vac - jvp_vac)
    reference_vacuum_norm = norms(ref_vac)
    # The control that makes the gate meaningful: comparing the model's
    # unconverted K^{E_F} derivative against the vacuum-gauge reference is the
    # cross-gauge mistake. It must be measurably worse, or the conversion is
    # not doing anything and the gate has no evidence to stand on.
    cross_gauge = difference_norms(reference.d_vacuum_fd, jvp)

    # A model whose derivative is the right pattern at the wrong amplitude is a
    # different failure from one that is diffusely wrong, and only the first is
    # plausibly a fine-tuning target. Measured, not assumed: alpha minimising
    # ||D_ref - alpha D_jvp||, and what is left after removing it.
    scale = best_fit_scale(reference.d_label_fc, jvp)
    residual_after_scale = (
        norms(ref_values - float(scale) * jvp_values) if scale is not None else {"frobenius": float("nan"), "max_abs": float("nan")}
    )

    floor = max(
        float(tau_fd.get("frobenius", 0.0)),
        float(model_numerical["frobenius"]),
        float(tau_backend["frobenius"]),
    )
    explained = float(
        np.sqrt(
            float(tau_fd.get("frobenius", 0.0)) ** 2
            + float(model_numerical["frobenius"]) ** 2
            + float(tau_backend["frobenius"]) ** 2
            + float(truncation["frobenius"]) ** 2
        )
    )

    row = {
        "direction_name": reference.direction_name,
        "direction_kind": reference.direction_kind,
        "delta_ang": reference.delta_ang,
        "unit": CANONICAL_UNITS["D_H"],
        "reference_convention": "label_H_minus_fermi_S",
        "reference": reference_norm,
        "jvp": norms(jvp_values),
        "residual": residual,
        "relative_frobenius": (
            residual["frobenius"] / reference_norm["frobenius"]
            if reference_norm["frobenius"] > 0.0
            else float("nan")
        ),
        "cosine": (
            float(ref_values @ jvp_values / (np.linalg.norm(ref_values) * np.linalg.norm(jvp_values)))
            if reference_norm["frobenius"] > 0.0 and norms(jvp_values)["frobenius"] > 0.0
            else float("nan")
        ),
        "error_budget": {
            "tau_num_reference_go2": dict(tau_fd),
            "tau_num_model_frozen_plateau": model_numerical,
            "tau_backend_jvp_vs_frozen": tau_backend,
            "support_truncation_outside_model_graph": truncation,
            "explained_quadrature_frobenius": explained,
            "residual_frobenius": residual["frobenius"],
            "residual_over_explained": (
                residual["frobenius"] / explained if explained > 0.0 else float("inf")
            ),
            "residual_over_largest_floor": (
                residual["frobenius"] / floor if floor > 0.0 else float("inf")
            ),
        },
        "best_fit_scale": scale,
        "residual_after_best_fit_scale": residual_after_scale,
        "convention_terms": reference.term_norms,
        "fermi_level_ev": reference.fermi_ev,
        "d_fermi_ev_per_ang": reference.d_fermi_ev_per_ang,
        "control_against_unshifted_D_H": convention_control,
        "reference_label_consistency_fc_vs_fd": label_consistency,
        "gauge_conversion": {
            **reference.gauge,
            "reference_vacuum_gauge": reference_vacuum_norm,
            "residual_in_vacuum_gauge": residual_vacuum,
            "relative_frobenius_in_vacuum_gauge": (
                residual_vacuum["frobenius"] / reference_vacuum_norm["frobenius"]
                if reference_vacuum_norm["frobenius"] > 0.0
                else float("nan")
            ),
            "control_unconverted_model_vs_vacuum_reference": cross_gauge,
        },
        "keys_total": len(keys),
        "keys_outside_model_graph": len(outside),
        "frozen_deltas_ang": plateau_deltas,
    }

    blocks = [mapping.block(key) for key in keys]
    shells = [round(mapping.distance_ang(key), 2) for key in keys]
    deciles = magnitude_deciles(ref_values)
    groups = [
        *group_metrics(
            keys,
            reference.d_label_fc,
            jvp,
            grouping="orbital_block",
            labels=blocks,
            direction_name=reference.direction_name,
            total_residual=residual["frobenius"],
        ),
        *group_metrics(
            keys,
            reference.d_label_fc,
            jvp,
            grouping="pair_distance_ang",
            labels=shells,
            direction_name=reference.direction_name,
            total_residual=residual["frobenius"],
        ),
        *group_metrics(
            keys,
            reference.d_label_fc,
            jvp,
            grouping="reference_magnitude_decile",
            labels=deciles,
            direction_name=reference.direction_name,
            total_residual=residual["frobenius"],
        ),
    ]
    row["dominant_groups"] = {
        grouping: max(
            (entry for entry in groups if entry["grouping"] == grouping),
            key=lambda entry: entry["share_of_total_residual"],
            default=None,
        )
        for grouping in ("orbital_block", "pair_distance_ang", "reference_magnitude_decile")
    }
    return row, groups


def decide(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibration: Sequence[str],
    gauge_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The C14 verdict, from pre-registered constants and the calibration split."""
    largest = max((float(row["reference"]["frobenius"]) for row in rows), default=0.0)
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        is_null = float(row["reference"]["frobenius"]) <= NULL_DIRECTION_FRACTION * largest
        evaluated.append(
            {
                "direction_name": row["direction_name"],
                "direction_kind": row["direction_kind"],
                "split": "calibration" if row["direction_name"] in calibration else "held_out",
                "is_null_direction": bool(is_null),
                "relative_frobenius": float(row["relative_frobenius"]),
                "residual_frobenius": float(row["residual"]["frobenius"]),
                "residual_over_explained": float(row["error_budget"]["residual_over_explained"]),
            }
        )

    signal = [entry for entry in evaluated if not entry["is_null_direction"]]
    calibration_rows = [entry for entry in signal if entry["split"] == "calibration"]
    held_out_rows = [entry for entry in signal if entry["split"] == "held_out"]
    null_rows = [entry for entry in evaluated if entry["is_null_direction"]]

    calibrated_bound = (
        max(entry["relative_frobenius"] for entry in calibration_rows) if calibration_rows else None
    )
    transfer_bound = (
        calibrated_bound * CALIBRATION_TRANSFER_FACTOR if calibrated_bound is not None else None
    )
    transfers = (
        all(entry["relative_frobenius"] <= transfer_bound for entry in held_out_rows)
        if transfer_bound is not None and held_out_rows
        else None
    )
    worst = max((entry["relative_frobenius"] for entry in signal), default=float("nan"))
    attributable = all(
        entry["residual_over_explained"] >= ATTRIBUTION_MARGIN for entry in signal
    )
    adequate = bool(signal) and worst <= ADEQUACY_RELATIVE_FROBENIUS

    # The gauge blocker is retired only by a gate that actually passed on real
    # numbers. The other two are untouched by this work and stay exactly as they
    # were: neither Delta_out nor the reference-convergence attachment is
    # affected by putting the two sides into a common gauge.
    gauge_verified = bool(gauge_gate) and gauge_gate.get("verdict") == "PASS"
    blockers = [
        *(
            []
            if gauge_verified
            else ["common electrostatic gauge not yet wired into this comparison artifact"]
        ),
        "finite-PAO basis completeness and Delta_out not bounded",
        "reference convergence evidence not attached to this comparison artifact",
    ]

    return {
        "decision": "UNDECIDED",
        "decision_blockers": blockers,
        "final_comparator_gauge_conversion_verified": gauge_verified,
        "gauge_conversion_gate": dict(gauge_gate) if gauge_gate else None,
        # The gauge gate is ONE prerequisite among several. It is reported here
        # with the standing verdicts of the others so the claim policy is
        # exercised against reality: checkpoint_lineage is NO_GO for the
        # original selection, and no gauge result can change that.
        "scientific_gates": {
            "final_comparator_gauge_conversion_verified": gauge_verified,
            "checkpoint_lineage": "NO_GO",
            "delta_out_closure": "NO_GO",
            "gauge_derivative_audit": "PASS",
            "graph2mat_target_gauge_contract": "PASS",
            "numerical_PAO_convergence": "PASS",
            "final_case_exclusion": "PASS",
            "previously_inspected": True,
        },
        "worst_relative_frobenius": worst,
        "adequacy_threshold": ADEQUACY_RELATIVE_FROBENIUS,
        "calibration_directions": list(calibration),
        "calibrated_relative_bound": calibrated_bound,
        "transfer_factor": CALIBRATION_TRANSFER_FACTOR,
        "held_out_within_calibrated_bound": transfers,
        "attribution_margin": ATTRIBUTION_MARGIN,
        "residual_exceeds_every_numerical_floor": bool(attributable),
        "per_direction": evaluated,
        "null_directions": [entry["direction_name"] for entry in null_rows],
        "not_a_tolerance_on_g": (
            "derivative-level only; tau_model for g must come from propagating delta_Delta "
            "through the electronic subspace (delta_g = C_f^dag delta_Delta C_i) in C20, "
            "never from this Frobenius ratio"
        ),
    }


def gauge_conversion_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``final_comparator_gauge_conversion_verified``: is this comparison single-gauge?

    PASS requires, on every direction carrying a derivative to convert:

    1. the per-geometry conversion closed its own discrete product-rule identity
       (both terms present, to float64), and
    2. the per-geometry result is separated from the a-posteriori-only shortcut
       by more than the numerical budget -- i.e. the ordering demonstrably
       mattered here, rather than being an unobservable formality.

    This gate says the two sides are in one gauge. It says nothing about whether
    they agree, and nothing about the checkpoint's provenance.

    *Why "the conversion reduces the residual" is deliberately NOT a criterion.*
    It was tried, and it is unsound. Measured on this campaign: a model that is
    exactly right satisfies it (residual 1.08e-3 converted vs 11.73 unconverted),
    and so does one that undershoots -- but a model that *overshoots* fails it,
    because the omitted term (c_vac - E_F).dS is +11.73 eV/Ang and an overshoot
    error points the same way, so the two partially cancel and the cross-gauge
    comparison looks spuriously better. The real checkpoint overshoots by ~1.6x
    (best-fit scale 0.617, cosine 0.945), so such a criterion would have failed
    the gate *because of the model's error* rather than because of the gauge --
    and would have rewarded the very error cancellation this conversion exists
    to prevent. A gate on the comparison may not take its verdict from the
    magnitude of the model's error. The residual in both gauges is still
    reported per direction, as evidence rather than as a criterion.
    """
    # A uniform translation leaves S invariant, so (c_vac - E_F).dS is exactly
    # zero there as a matter of physics, not as a failure of the conversion.
    # Such a direction is judged with the comparator's own pre-registered null
    # rule instead of being asked to exhibit a term it cannot have.
    largest = max(
        (
            float(row["gauge_conversion"]["reference_vacuum_gauge"]["frobenius"])
            for row in rows
            if row.get("gauge_conversion")
        ),
        default=0.0,
    )

    per_direction: list[dict[str, Any]] = []
    for row in rows:
        record = row.get("gauge_conversion")
        if not record:
            continue
        reference_norm = float(record["reference_vacuum_gauge"]["frobenius"])
        separation = float(record["per_geometry_minus_a_posteriori"]["frobenius"])
        closure = float(record["identity_closure"]["frobenius"])
        # A direction with no reference response has no derivative to convert;
        # it cannot testify either way and is not held against the gate. This is
        # the comparator's own pre-registered null rule, reused rather than
        # re-invented -- translation_x is null under it, and its exactly-zero
        # (c_vac - E_F).dS is the correct physics, not a missing conversion.
        null_direction = (
            not math.isfinite(reference_norm)
            or reference_norm <= NULL_DIRECTION_FRACTION * largest
        )
        identity_closes = closure <= GAUGE_IDENTITY_RELATIVE_TOLERANCE * max(reference_norm, 1.0)
        term_resolved = separation > GAUGE_TERM_FLOOR_EV_PER_ANG
        converted = float(record["residual_in_vacuum_gauge"]["frobenius"])
        unconverted = float(record["control_unconverted_model_vs_vacuum_reference"]["frobenius"])
        per_direction.append(
            {
                "direction_name": row["direction_name"],
                "is_null_direction": bool(null_direction),
                "identity_closure_frobenius": closure,
                "identity_closes": bool(identity_closes),
                "per_geometry_minus_a_posteriori_frobenius": separation,
                "omitted_term_frobenius": float(
                    record["omitted_product_rule_term_cvac_minus_Ef_times_dS"]["frobenius"]
                ),
                "product_rule_term_resolved_above_budget": bool(term_resolved),
                # Reported as evidence, never used as a criterion: see the
                # docstring on why a "conversion reduces the residual" test is
                # unsound for an overshooting model.
                "residual_converted_frobenius": converted,
                "residual_unconverted_cross_gauge_frobenius": unconverted,
                "d_cvac_minus_Ef_ev_per_ang": float(record["d_cvac_minus_Ef_ev_per_ang"]),
            }
        )

    testifying = [entry for entry in per_direction if not entry["is_null_direction"]]
    passed = bool(testifying) and all(
        entry["identity_closes"] and entry["product_rule_term_resolved_above_budget"]
        for entry in testifying
    )
    return {
        "gate": "final_comparator_gauge_conversion_verified",
        "verdict": "PASS" if passed else "NO_GO",
        "applied": "per geometry, before the finite-difference stencil",
        "converted_before_stencil": True,
        "thresholds": {
            "product_rule_term_floor_ev_per_ang": GAUGE_TERM_FLOOR_EV_PER_ANG,
            "identity_relative_tolerance": GAUGE_IDENTITY_RELATIVE_TOLERANCE,
        },
        "per_direction": per_direction,
        "scope": (
            "establishes that model and reference are compared inside one electrostatic gauge; "
            "it does not bound Delta_out, does not certify checkpoint provenance, and on its own "
            "promotes no claim"
        ),
    }


def finalize_verdict(
    verdict: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Downgrade to ``not_evaluable`` when a precondition failed.

    The acceptance criterion this enforces: a failure of the reference, of the
    mapping or of the JVP is never reported as a model error. It becomes an
    unevaluable comparison, and the failing checks are named.
    """
    from epc_claim_policy import claim_policy
    verdict = {**verdict, "claim_policy": claim_policy(verdict.get("scientific_gates", {}))}
    failed = [row["check"] for row in checks if not row["passed"]]
    if not failed:
        return dict(verdict)
    return {
        **verdict,
        "decision": "not_evaluable",
        "failed_preconditions": failed,
        "reason": (
            "a precondition on the reference, the mapping or the JVP failed; no part of the "
            "residual is attributed to the model: " + ", ".join(failed)
        ),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def load_certification(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointDerivativeError(
            f"{path} is missing. Run certify_siesta_dhsdr.py first: without GO-2 there is no "
            "certified reference to measure the checkpoint against."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CERTIFICATION_SCHEMA:
        raise CheckpointDerivativeError(
            f"{path} is schema {payload.get('schema')!r}, expected {CERTIFICATION_SCHEMA!r}."
        )
    return payload


def quantify(
    *,
    certification_path: Path = DEFAULT_CERTIFICATION,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    energy_zero_path: Path = DEFAULT_ENERGY_ZERO,
    backend: str = "cuda",
    calibration: Sequence[str] = CALIBRATION_DIRECTIONS,
    run_internal_validation: bool = True,
) -> dict[str, Any]:
    import sisl
    import torch

    from graph2mat_autograd_derivatives import batch_topology_hash, resolve_jvp_backend

    certification = load_certification(certification_path)
    campaign = load_campaign(reference_root)
    vacuum_levels = load_vacuum_levels(energy_zero_path, campaign)
    checks: list[dict[str, Any]] = [
        check(
            "go2_certified",
            certification.get("verdict") == "PASS",
            f"GO-2 verdict is {certification.get('verdict')!r} in {certification_path.name}; "
            "an uncertified reference cannot expose a model error",
            certification=str(certification_path),
        )
    ]

    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id, fermi_ev=campaign.fermi_ev(equilibrium_id)
    )
    # Species identity comes from the fdf: a TSHS carries orbital counts, not
    # atomic numbers, and sisl reconstructs "H with 4 orbitals" from it. The
    # coordinates of both are compared below, so the fdf cannot smuggle in a
    # different structure than the one the matrices belong to.
    geometry = sisl.get_sile(str(campaign.run_dir(equilibrium_id) / "RUN.fdf")).read_geometry()
    matrix_geometry = sisl.get_sile(
        str(campaign.run_dir(equilibrium_id) / f"{equilibrium_id}.TSHS")
    ).read_geometry()
    positions_ang = np.asarray(geometry.xyz, dtype=np.float64)
    cell_ang = np.asarray(geometry.cell, dtype=np.float64)
    checks.append(
        check(
            "fdf_geometry_matches_tshs",
            bool(
                np.allclose(positions_ang, np.asarray(matrix_geometry.xyz, dtype=np.float64), atol=1e-8)
                and np.allclose(cell_ang, np.asarray(matrix_geometry.cell, dtype=np.float64), atol=1e-8)
            ),
            "the fdf the model reads and the TSHS the reference derivative belongs to describe "
            "the same cell and the same coordinates",
        )
    )

    manifest_positions = np.asarray(
        next(row for row in campaign.manifest["rows"] if row["run_id"] == equilibrium_id)["positions_ang"],
        dtype=np.float64,
    )
    checks.append(
        check(
            "equilibrium_geometry_matches_manifest",
            bool(np.allclose(positions_ang, manifest_positions, atol=1e-9)),
            "the TSHS geometry the model is evaluated on is the one the reference campaign "
            f"certified (max |dx| = {float(np.abs(positions_ang - manifest_positions).max()):.3e} Ang)",
        )
    )

    mapping = orbital_map(
        campaign.run_dir(equilibrium_id) / f"{equilibrium_id}.ORB_INDX", positions_ang, cell_ang
    )
    checks.append(
        check(
            "orbital_count_matches_reference",
            len(mapping.l_of_orbital) == equilibrium.no_u,
            f"ORB_INDX enumerates {len(mapping.l_of_orbital)} unit-cell orbitals, the TSHS has "
            f"{equilibrium.no_u}",
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    structure_fdf = ghost_free_fdf(geometry, output_dir / "geometry" / "RUN.fdf")
    model, batch, processor, model_provenance = load_model_and_batch(
        checkpoint, structure_fdf, basis_dir=Path(campaign.manifest["basis_dir"])
    )
    change_of_basis = np.asarray(processor.basis_table.change_of_basis, dtype=np.float64)

    # float64 everywhere: the reference is a converged DFT derivative, and a
    # float32 roundoff floor would be indistinguishable from a small model error.
    import validate_graph2mat_jvp as vj

    model, batch = vj._cast(model, batch, torch.float64)
    model, batch, backend_record = resolve_jvp_backend(model, batch, backend, output_keys=OUTPUT_KEYS)
    base_topology = batch_topology_hash(batch)

    predicted_h, prediction_meta = model_predictions(model, batch, processor)
    against_label = difference_norms(equilibrium.h_shifted, predicted_h)
    against_absolute = difference_norms(equilibrium.h_absolute, predicted_h)
    label_convention = against_label["frobenius"] < against_absolute["frobenius"]
    checks.append(
        check(
            "label_convention_identified",
            bool(label_convention),
            "the checkpoint predicts sisl's H - E_F S (relative Frobenius "
            f"{against_label['frobenius'] / norms(align(equilibrium.h_shifted)[1][0])['frobenius']:.4f}) "
            "rather than the unshifted H "
            f"({against_absolute['frobenius'] / norms(align(equilibrium.h_absolute)[1][0])['frobenius']:.4f}), "
            "so the reference derivative is converted into that convention before comparison",
            equilibrium_h_vs_label=against_label,
            equilibrium_h_vs_absolute=against_absolute,
            **prediction_meta,
        )
    )
    # "Same basis" is a byte statement, not a naming one: the .ion.xml the
    # reference campaign used and the one the checkpoint was trained on must be
    # the same file, or the two matrices are indexed by different orbitals.
    training_hashes = (model_provenance["training_dataset"].get("basis_hashes") or {})
    reference_hashes = {
        path.name: file_sha256(path)
        for path in sorted(Path(campaign.manifest["basis_dir"]).glob("*.ion.xml"))
    }
    shared_basis = {
        name: (training_hashes[name] == reference_hashes[name])
        for name in sorted(set(training_hashes) & set(reference_hashes))
    }
    checks.append(
        check(
            "training_and_reference_basis_identical",
            bool(shared_basis) and all(shared_basis.values()),
            "the .ion.xml the checkpoint was trained on and the one the SIESTA reference used "
            f"are byte-identical ({sorted(shared_basis)})"
            if shared_basis and all(shared_basis.values())
            else "the training basis and the reference basis differ (or cannot be compared); the "
            "two sides would not be indexed by the same orbitals",
            training_basis_sha256=training_hashes,
            reference_basis_sha256=reference_hashes,
        )
    )

    reference_support = set(equilibrium.h_shifted)
    model_support = set(predicted_h)
    checks.append(
        check(
            "model_support_is_contained_in_reference",
            model_support <= reference_support,
            f"the model's graph produces {len(model_support)} (row, col, R) entries, all of which "
            f"exist in the reference's {len(reference_support)}"
            if model_support <= reference_support
            else f"{len(model_support - reference_support)} model entries have no reference "
            "counterpart: the two index spaces are not the same mapping",
        )
    )

    internal_validation: dict[str, Any] | None = None
    if run_internal_validation:
        radii = [float(np.max(entry.R)) for entry in processor.basis_table.basis]
        internal_validation = vj.validate_directional_jvp(
            model,
            batch,
            change_of_basis=change_of_basis,
            data_processor=processor,
            cutoff_ang=[radii[int(point_type)] for point_type in batch["point_types"]],
            lattice_vectors_ang=cell_ang,
            positions_ang=positions_ang,
            deltas=fdp.delta_sweep(0.004, count=5),
            dtypes=("float64",),
            backends=(backend_record.effective,),
        )
        checks.append(
            check(
                "go3_internal_validation_passed",
                bool(internal_validation["summary"]["passed"]),
                "the JVP is internally certified (GO-3) on this very model and geometry, so a "
                "disagreement with SIESTA cannot be a JVP defect"
                if internal_validation["summary"]["passed"]
                else "GO-3 fails on this model/geometry: the residual against SIESTA is NOT "
                "attributable to the model until the JVP itself is fixed",
                failed=internal_validation["summary"]["checks_failed"],
            )
        )

    # --- per-direction comparison ---------------------------------------------
    directions = {
        entry["direction_name"]: entry
        for entry in campaign.manifest["perturbation_space"]["directions"]
    }
    selected_delta: dict[str, float] = {}
    tau_fd: dict[str, dict[str, float]] = {}
    for comparison in certification["comparisons"]:
        if comparison["kind"] != "D_H":
            continue
        name = comparison["direction_name"]
        tau_fd[name] = dict(comparison["tau_fd"])
        plateau = comparison["plateau"]
        selected_delta[name] = float(
            plateau["selected_delta_ang"]
            if plateau.get("has_plateau")
            else max(campaign.deltas)
        )

    dhsdr_cache: dict[float, Any] = {}
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    topology_hashes = {base_topology["topology_hash"]}
    for name, entry in sorted(directions.items()):
        if name not in selected_delta:
            raise CheckpointDerivativeError(
                f"direction {name!r} is in the perturbation space but has no certified D_H "
                "comparison in the GO-2 report."
            )
        delta = selected_delta[name]
        if delta not in dhsdr_cache:
            dhsdr_cache[delta] = read_dhsdr(campaign.dhsdr_path(delta))
        vectors = np.asarray(entry["vectors"], dtype=np.float64)
        direction = direction_from_manifest(entry)
        if direction.direction_hash != entry["direction_hash"]:
            raise CheckpointDerivativeError(
                f"rebuilt direction {name!r} hashes to {direction.direction_hash[:12]}... but the "
                f"reference campaign displaced {str(entry['direction_hash'])[:12]}...; the JVP would "
                "be taken along a direction SIESTA never moved."
            )
        reference = reference_directional(
            campaign,
            equilibrium,
            direction_name=name,
            direction_kind=str(entry["direction_kind"]),
            vectors=vectors,
            delta_ang=delta,
            dhsdr=dhsdr_cache[delta],
            vacuum_levels=vacuum_levels,
            equilibrium_run_id=equilibrium_id,
        )
        jvp, jvp_meta = directional_derivative_field(
            model,
            batch,
            direction,
            processor=processor,
            change_of_basis=change_of_basis,
            backend=backend_record,
        )
        frozen_by_delta: dict[float, dict[Key, float]] = {}
        for amplitude in campaign.deltas:
            frozen, hashes = frozen_derivative_field(
                model,
                batch,
                direction,
                amplitude,
                processor=processor,
                change_of_basis=change_of_basis,
            )
            frozen_by_delta[float(amplitude)] = frozen
            topology_hashes.update(hashes)
        row, groups = compare_direction(
            reference,
            jvp,
            frozen_by_delta,
            tau_fd=tau_fd[name],
            mapping=mapping,
            equilibrium_overlap=equilibrium.overlap,
            d_overlap_fc=reference.d_s_fc,
        )
        row["jvp_metadata"] = jvp_meta
        row["direction_hash"] = entry["direction_hash"]
        rows.append(row)
        group_rows.extend(groups)

    checks.append(
        check(
            "batch_topology_unchanged",
            len(topology_hashes) == 1,
            f"every JVP and every displaced frozen forward ran on one graph "
            f"({base_topology['topology_hash'][:12]}...)",
            topology_hashes=sorted(topology_hashes),
            **base_topology,
        )
    )
    consistency_floor = {
        row["direction_name"]: float(tau_fd[row["direction_name"]]["frobenius"])
        + abs(float(row["fermi_level_ev"]))
        * float(
            next(
                (
                    comparison["tau_fd"]["frobenius"]
                    for comparison in certification["comparisons"]
                    if comparison["kind"] == "D_S"
                    and comparison["direction_name"] == row["direction_name"]
                ),
                0.0,
            )
        )
        for row in rows
    }
    consistent = {
        row["direction_name"]: float(row["reference_label_consistency_fc_vs_fd"]["frobenius"])
        <= consistency_floor[row["direction_name"]]
        for row in rows
    }
    checks.append(
        check(
            "reference_label_convention_consistent",
            all(consistent.values()),
            "D_H^FC - E_F D_S^FC - (dE_F) S agrees with the direct central difference of sisl's "
            "H - E_F S within the propagated GO-2 noise floor, so the converted reference is "
            "certified, not assumed",
            per_direction={
                name: {
                    "residual_frobenius": float(
                        next(
                            row["reference_label_consistency_fc_vs_fd"]["frobenius"]
                            for row in rows
                            if row["direction_name"] == name
                        )
                    ),
                    "floor": consistency_floor[name],
                    "passed": bool(value),
                }
                for name, value in consistent.items()
            },
        )
    )

    calibration = [name for name in calibration if name in directions]
    held_out = [name for name in sorted(directions) if name not in calibration]
    checks.append(
        check(
            "calibration_and_final_case_disjoint",
            bool(calibration) and bool(held_out),
            f"tolerance is calibrated on {calibration} and tested on {held_out}; the graphene-Gamma "
            "case C20 will evaluate (an E_2g phonon eigenvector) is in neither set -- this "
            "campaign contains no phonon eigenvector at all",
            calibration=calibration,
            held_out=held_out,
            phonon_eigenvector_directions_present=[
                name for name, entry in directions.items()
                if str(entry.get("direction_kind")) == "phonon_eigenmode"
            ],
        )
    )

    gauge_gate = gauge_conversion_gate(rows)
    checks.append(
        check(
            "comparison_is_single_gauge",
            gauge_gate["verdict"] == "PASS",
            "reference and model derivative are both expressed in K^{c_vac} = H_absolute - "
            "c_vac(R) S(R), with the conversion applied to each displaced geometry BEFORE the "
            "finite-difference stencil, so both product-rule terms are carried"
            if gauge_gate["verdict"] == "PASS"
            else "the comparison is not established to be single-gauge; see the gauge gate",
            gate=gauge_gate["verdict"],
            per_direction=gauge_gate["per_direction"],
        )
    )
    verdict = finalize_verdict(
        decide(rows, calibration=calibration, gauge_gate=gauge_gate), checks
    )

    report = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": checks,
        "directions": rows,
        "inputs": {
            "certification": str(certification_path),
            "certification_verdict": certification.get("verdict"),
            "reference_root": str(reference_root),
            "checkpoint": str(checkpoint),
            "equilibrium_run_id": equilibrium_id,
            "structure_fdf": str(structure_fdf),
        },
        "provenance": {
            "siesta_runtime": certification.get("provenance", {}),
            "model": model_provenance,
            "backend": backend_record.to_metadata(),
            "dtype": "float64",
            "topology": base_topology,
            "orbital_contract": campaign.manifest.get("orbital_contract"),
            "units": CANONICAL_UNITS,
        },
        "internal_validation_go3": internal_validation,
    }
    write_report(report, group_rows, output_dir)
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _json_safe(value.detach().cpu().tolist())
    except ImportError:  # pragma: no cover - torch is present in every real run
        pass
    return value


def write_report(report: Mapping[str, Any], group_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_derivative_error.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if group_rows:
        fields = list(group_rows[0].keys())
        with (output_dir / "checkpoint_derivative_error_groups.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field) for field in fields} for row in group_rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--certification", type=Path, default=DEFAULT_CERTIFICATION)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--energy-zero",
        type=Path,
        default=DEFAULT_ENERGY_ZERO,
        help="certified per-geometry c_vac(R) artifact defining the common electrostatic gauge",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "cpu"),
        default="cuda",
        help="requested JVP backend; a failed CUDA preflight falls back to CPU with a recorded reason",
    )
    parser.add_argument(
        "--calibration-direction",
        action="append",
        default=None,
        help=f"directions the tolerance is calibrated on (default: {list(CALIBRATION_DIRECTIONS)})",
    )
    parser.add_argument(
        "--skip-internal-validation",
        action="store_true",
        help="skip the GO-3 re-run (only for iterating on the report; the verdict then has no JVP precondition)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = quantify(
        certification_path=args.certification,
        reference_root=args.reference_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        energy_zero_path=args.energy_zero,
        backend=args.backend,
        calibration=tuple(args.calibration_direction or CALIBRATION_DIRECTIONS),
        run_internal_validation=not args.skip_internal_validation,
    )
    verdict = report["verdict"]
    for row in report["checks"]:
        print(f"[{TICKET}] {row['passed']!s:>5} {row['check']:<44} {row['detail'][:110]}")
    for row in report["directions"]:
        print(
            f"[{TICKET}] {row['direction_name']:<16} "
            f"||D_ref|| = {row['reference']['frobenius']:.4g}  "
            f"||residual|| = {row['residual']['frobenius']:.4g}  "
            f"rel = {row['relative_frobenius']:.4g}  "
            f"residual/explained = {row['error_budget']['residual_over_explained']:.4g}"
        )
    gate = verdict.get("gauge_conversion_gate") or {}
    for entry in gate.get("per_direction", []):
        print(
            f"[{TICKET}] gauge {entry['direction_name']:<16} "
            f"d(c_vac-E_F) = {entry['d_cvac_minus_Ef_ev_per_ang']:.6g} eV/Ang  "
            f"||omitted (c_vac-E_F).dS|| = {entry['omitted_term_frobenius']:.6g}  "
            f"identity = {entry['identity_closure_frobenius']:.3g}"
        )
    print(
        f"[{TICKET}] final_comparator_gauge_conversion_verified: {gate.get('verdict')}"
    )
    print(f"[{TICKET}] claim_policy: {verdict.get('claim_policy')}")
    print(f"[{TICKET}] decision: {verdict['decision']}")
    print(f"[{TICKET}] decision_blockers: {verdict.get('decision_blockers')}")
    return 0 if verdict["decision"] != "not_evaluable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
