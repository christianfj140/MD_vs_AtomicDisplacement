#!/usr/bin/env python3
"""Prospective Graph2Mat derivative-loss smoke and short weight sweep."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "Comparison/scripts", ROOT / "shared", ROOT / "scripts/torch_serialization_compat"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import quantify_checkpoint_derivative_error as c14  # noqa: E402
from benchmark_manifest import file_sha256  # noqa: E402
from certify_siesta_dhsdr import load_campaign, read_tshs  # noqa: E402
from graph2mat_autograd_derivatives import (  # noqa: E402
    compute_graph2mat_directional_derivative,
    derivative_prediction_to_sparse_matrices,
    flatten_graph2mat_predictions,
)
from read_siesta_dhsdr import read_dhsdr  # noqa: E402
from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402

SOURCE = ROOT / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints/spectral-best-epoch=005-step=00084.ckpt"
OUTPUT = ROOT / "Comparison/results/epc/derivative_remediation_graphene_gamma_20260908"
CALIBRATION = ("atom0000_x", "translation_x")
VALIDATION = "random_seed0"
JOINT_CALIBRATION = ("atom0000_x", "translation_x", "random_seed0")
JOINT_K_POINTS = ("gamma", "M", "K", "k_generic_1")
COORDINATE_DIRECTIONS = tuple(
    f"atom{atom:04d}_{axis}" for atom in range(2) for axis in ("x", "y", "z")
)
WEIGHTS = (0.03, 0.1, 0.3)
EIGENSPACES = ROOT / "Comparison/results/epc/eigenspaces/graphene"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_protocol(source: Path, output: Path, steps: int, learning_rate: float,
                    subspace_reference_root: Path | None, joint: bool = False,
                    snapshot_runs: Path | None = None, all_coordinates: bool = False) -> dict:
    calibration = (
        JOINT_CALIBRATION + tuple(name for name in COORDINATE_DIRECTIONS if name != "atom0000_x")
        if all_coordinates else JOINT_CALIBRATION if joint else CALIBRATION
    )
    payload = {
        "schema": "graphene_gamma_derivative_remediation_protocol_v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "calibration_directions": list(calibration),
        "validation_direction": f"{VALIDATION}@k_generic_2" if joint else VALIDATION,
        "excluded_result": (
            "explicit graphene Gamma E2g modal central differences; atom-coordinate dHSdR is used"
            if all_coordinates else "graphene Gamma E2g; never used for optimization or selection"
        ),
        "steps_per_weight": steps,
        "learning_rate": learning_rate,
        "derivative_weights": list(WEIGHTS),
        "selection": (
            "minimum held-out tau_model-subspace error subject to equilibrium relative drift <= 0.02"
            if subspace_reference_root
            else "minimum validation relative label error subject to equilibrium relative drift <= 0.02"
        ),
        "reference": "certified SIESTA dHSdR in the checkpoint H-E_F*S label convention",
        "no_new_dft": True,
        "loss_space": "tau_model_calibration_subspaces" if subspace_reference_root else "global_derivative_labels",
        "subspace_reference_root": str(subspace_reference_root) if subspace_reference_root else None,
        "snapshot_distillation_runs": str(snapshot_runs) if snapshot_runs else None,
    }
    path = output / "protocol.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        for key in ("source_checkpoint_sha256", "steps_per_weight", "learning_rate", "derivative_weights"):
            if frozen[key] != payload[key]:
                raise RuntimeError(f"requested {key} differs from frozen protocol")
        return frozen
    write_json(path, payload)
    return payload


def setup(checkpoint: Path, work: Path, backend: str):
    import sisl
    import validate_graph2mat_jvp as vj
    from graph2mat_autograd_derivatives import resolve_jvp_backend

    campaign = load_campaign(c14.DEFAULT_REFERENCE_ROOT)
    certification = c14.load_certification(c14.DEFAULT_CERTIFICATION)
    vacuum_levels = c14.load_vacuum_levels(c14.DEFAULT_ENERGY_ZERO, campaign)
    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id,
        fermi_ev=campaign.fermi_ev(equilibrium_id),
    )
    geometry = sisl.get_sile(str(campaign.run_dir(equilibrium_id) / "RUN.fdf")).read_geometry()
    fdf = c14.ghost_free_fdf(geometry, work / "geometry/RUN.fdf")
    model, batch, processor, _ = c14.load_model_and_batch(
        checkpoint, fdf, basis_dir=Path(campaign.manifest["basis_dir"])
    )
    model, batch = vj._cast(model, batch, torch.float64)
    model, batch, backend_record = resolve_jvp_backend(
        model, batch, backend, output_keys=c14.OUTPUT_KEYS
    )

    selected = {}
    for row in certification["comparisons"]:
        if row["kind"] == "D_H":
            plateau = row["plateau"]
            selected[row["direction_name"]] = float(
                plateau["selected_delta_ang"] if plateau.get("has_plateau") else max(campaign.deltas)
            )
    directions, references = {}, {}
    for entry in campaign.manifest["perturbation_space"]["directions"]:
        name = entry["direction_name"]
        direction = c14.direction_from_manifest(entry)
        if direction.direction_hash != entry["direction_hash"]:
            raise RuntimeError(f"direction hash mismatch for {name}")
        delta = selected[name]
        directions[name] = direction
        references[name] = c14.reference_directional(
            campaign, equilibrium,
            direction_name=name,
            direction_kind=entry["direction_kind"],
            vectors=np.asarray(entry["vectors"], dtype=np.float64),
            delta_ang=delta,
            dhsdr=read_dhsdr(campaign.dhsdr_path(delta)),
            vacuum_levels=vacuum_levels,
            equilibrium_run_id=equilibrium_id,
        ).d_label_fc
    return model, batch, processor, directions, references, backend_record


def add_coordinate_references(directions, references) -> None:
    import fd_perturbation_space as fdp

    campaign = load_campaign(c14.DEFAULT_REFERENCE_ROOT)
    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id,
        fermi_ev=campaign.fermi_ev(equilibrium_id),
    )
    derivative = read_dhsdr(campaign.dhsdr_path(0.01))
    for atom in range(2):
        for axis in ("x", "y", "z"):
            name = f"atom{atom:04d}_{axis}"
            direction = fdp.one_hot(2, atom, axis)
            directions[name] = direction
            references[name] = c14.combine(
                (1.0, c14.contract_fc(derivative, direction.vectors, "D_H")),
                (-equilibrium.fermi_ev, c14.contract_fc(derivative, direction.vectors, "D_S")),
            )


def reference_labels(model, batch, processor, fields):
    """Invert the existing labels_to mapping once using unique sentinel values."""
    with torch.no_grad():
        output = model(batch)
    sizes = [output[key].numel() for key in c14.OUTPUT_KEYS]
    sentinel = torch.arange(1, sum(sizes) + 1, dtype=output[c14.OUTPUT_KEYS[0]].dtype)
    encoded, offset = {}, 0
    for key, size in zip(c14.OUTPUT_KEYS, sizes):
        encoded[key] = sentinel[offset : offset + size].reshape(output[key].shape)
        offset += size
    orders: list = []
    matrix = derivative_prediction_to_sparse_matrices(
        processor, c14.cpu_batch(batch), encoded, supercell_orders=orders
    )[0]
    mapping = c14.sparse_field(matrix, orders[0])
    ids = {int(round(value)) for value in mapping.values()}
    if ids != set(range(1, len(sentinel) + 1)):
        raise RuntimeError("Graph2Mat label mapping is not one-to-one on this topology")

    targets = {}
    for name, field in fields.items():
        grouped = {index: [] for index in ids}
        for key, value in mapping.items():
            grouped[int(round(value))].append(field.get(key, 0.0))
        if max(max(values) - min(values) for values in grouped.values()) > 1e-7:
            raise RuntimeError(f"Hermitian partners disagree while mapping {name}")
        targets[name] = torch.as_tensor(
            [sum(grouped[index]) / len(grouped[index]) for index in range(1, len(sentinel) + 1)],
            dtype=sentinel.dtype,
        )
    return targets, mapping


def subspace_projectors(mapping, size: int, device, reference_root: Path, joint: bool = False):
    """Linear label -> C†D_H(k)C maps for the frozen tau_model splits."""
    eigenspaces = {}
    for path in EIGENSPACES.glob("eigenspace_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        coefficients = np.fromfile(path.parent / payload["vectors_file"], dtype=np.complex128)
        coefficients = coefficients.reshape((payload["norbits"], payload["state_count"]), order="F")
        eigenspaces[payload["label"]] = (payload["k_fractional"], coefficients)
    requests = (
        {
            "atom0000_x": JOINT_K_POINTS,
            VALIDATION: JOINT_K_POINTS + ("k_generic_2",),
        }
        if joint else {"atom0000_x": ("gamma", "M"), VALIDATION: ("k_generic_2",)}
    )
    direction_metadata = json.loads(
        (reference_root / "path__graph2mat_jvp.json").read_text(encoding="utf-8")
    )["directions"]
    projectors = {}
    for direction, labels in requests.items():
        projectors[direction] = []
        for label in labels:
            k, coefficients = eigenspaces[label]
            operator = np.zeros((coefficients.shape[0], coefficients.shape[0], size), dtype=np.complex128)
            for (row, column, image), sentinel in mapping.items():
                operator[row, column, int(round(sentinel)) - 1] += np.exp(2j * np.pi * np.dot(k, image))
            delta = float(direction_metadata[direction]["basis_response_delta_ang"])
            delta_tag = str(delta).replace(".", "p")
            raw_path = reference_root / "arrays" / f"raw_derivative__siesta_reference__{direction}__d{delta_tag}.npz"
            raw = np.load(raw_path)
            reference_dense = sum(
                block * np.exp(2j * np.pi * np.dot(k, image))
                for block, image in zip(raw["D_H"], raw["isc_off"])
            )
            response_path = reference_root / "arrays" / f"pao_covariant_response__siesta_reference__{direction}__{label}.npy"
            response = np.load(response_path) if response_path.exists() else reference_dense
            denominator = float(np.linalg.norm(coefficients.conj().T @ response @ coefficients))
            projectors[direction].append((
                label,
                torch.as_tensor(operator, device=device),
                torch.as_tensor(coefficients, device=device),
                torch.as_tensor(reference_dense, device=device),
                denominator,
            ))
    return projectors


def subspace_error(prediction, projectors):
    prediction = prediction.to(torch.complex128)
    errors = []
    for entry in projectors:
        _, operator, coefficients, reference, denominator, *offdiagonal = entry
        dense = torch.einsum("ijn,n->ij", operator, prediction)
        block = coefficients.conj().T @ (dense - reference) @ coefficients
        if offdiagonal and offdiagonal[0]:
            block = block - torch.diag(torch.diagonal(block))
        errors.append(torch.linalg.norm(block) / max(denominator, 1e-15))
    return torch.stack(errors)


def coordinate_projectors(mapping, size: int, device, fields):
    eigenspaces = {}
    for path in EIGENSPACES.glob("eigenspace_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        coefficients = np.fromfile(path.parent / payload["vectors_file"], dtype=np.complex128)
        eigenspaces[payload["label"]] = (
            payload["k_fractional"],
            coefficients.reshape((payload["norbits"], payload["state_count"]), order="F"),
        )
    result = {}
    for name in COORDINATE_DIRECTIONS:
        result[name] = []
        for label in JOINT_K_POINTS:
            k, coefficients = eigenspaces[label]
            operator = np.zeros((coefficients.shape[0], coefficients.shape[0], size), dtype=np.complex128)
            for (row, column, image), sentinel in mapping.items():
                operator[row, column, int(round(sentinel)) - 1] += np.exp(2j * np.pi * np.dot(k, image))
            reference = np.zeros((coefficients.shape[0], coefficients.shape[0]), dtype=np.complex128)
            for (row, column, image), value in fields[name].items():
                reference[row, column] += value * np.exp(2j * np.pi * np.dot(k, image))
            reference_block = coefficients.conj().T @ reference @ coefficients
            reference_offdiagonal = reference_block - np.diag(np.diag(reference_block))
            result[name].append((
                label, torch.as_tensor(operator, device=device),
                torch.as_tensor(coefficients, device=device),
                torch.as_tensor(reference, device=device),
                float(np.linalg.norm(reference_offdiagonal)), True,
            ))
    return result


def flat_prediction(model, batch):
    output = model(batch)
    return flatten_graph2mat_predictions(output, output_keys=c14.OUTPUT_KEYS)[0]


def derivative_flat(model, batch, direction, change_of_basis, *, create_graph=False):
    result = compute_graph2mat_directional_derivative(
        model, batch, direction,
        change_of_basis=change_of_basis,
        output_keys=c14.OUTPUT_KEYS,
        create_graph=create_graph,
    )
    return flatten_graph2mat_predictions(result.derivative, output_keys=c14.OUTPUT_KEYS)[0]


def metrics(model, batch, directions, targets, source_equilibrium, change_of_basis):
    rows = {}
    for name, direction in directions.items():
        prediction = derivative_flat(model, batch, direction, change_of_basis)
        target = targets[name].to(prediction)
        rows[name] = float(torch.linalg.norm(prediction - target) / torch.linalg.norm(target).clamp_min(1e-12))
    with torch.no_grad():
        current = flat_prediction(model, batch)
    drift = float(torch.linalg.norm(current - source_equilibrium) / torch.linalg.norm(source_equilibrium))
    return rows, drift


def load_snapshot_batches(runs_path: Path, processor, device, dtype=torch.float64):
    from graph2mat.bindings.torch.data import TorchBasisMatrixData
    from torch_geometric.loader.dataloader import DataLoader

    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    paths = [(runs_path.parent / path).resolve() for path in payload["train"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing snapshot RUN.fdf: {missing[0]}")
    data = [TorchBasisMatrixData.new(path, data_processor=processor, labels=False) for path in paths]
    batches = []
    for batch in DataLoader(data, batch_size=32, shuffle=False):
        batch = batch.to(device)
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                batch[key] = value.to(dtype)
        batches.append(batch)
    return batches, len(paths)


def snapshot_anchors(model, batches):
    with torch.no_grad():
        return [flat_prediction(model, batch).detach().cpu() for batch in batches]


def train_candidate(model, batch, processor, directions, targets, weight, steps, learning_rate,
                    projectors=None, calibration=CALIBRATION, snapshot_batches=None,
                    snapshot_targets=None):
    change_of_basis = np.asarray(processor.basis_table.change_of_basis)
    with torch.no_grad():
        source_equilibrium = flat_prediction(model, batch).detach()
    strongest = max(float(target.square().mean()) for target in targets.values())
    floor = strongest * 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps + 1):
        errors, drift = metrics(model, batch, directions, targets, source_equilibrium, change_of_basis)
        history.append({"step": step, "relative_error": errors, "equilibrium_relative_drift": drift})
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        for name in calibration:
            prediction = derivative_flat(
                model, batch, directions[name], change_of_basis, create_graph=True
            )
            target = targets[name].to(prediction)
            if projectors and name in projectors:
                loss = weight * subspace_error(prediction, projectors[name]).square().mean()
            else:
                loss = weight * (prediction - target).square().mean() / max(float(target.square().mean()), floor)
            (loss / len(calibration)).backward()
        current = flat_prediction(model, batch)
        anchor = (current - source_equilibrium).square().mean() / source_equilibrium.square().mean()
        anchor.backward()
        if snapshot_batches:
            index = step % len(snapshot_batches)
            prediction = flat_prediction(model, snapshot_batches[index])
            target = snapshot_targets[index].to(prediction)
            ((prediction - target).square().mean() / target.square().mean().clamp_min(1e-12)).backward()
        gradient_norm = torch.sqrt(sum(
            parameter.grad.square().sum() for parameter in model.parameters()
            if parameter.grad is not None
        ))
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        history[-1]["gradient_norm"] = float(gradient_norm)
    return history


def save_checkpoint(source: Path, destination: Path, model, provenance: dict) -> None:
    allow_graph2mat_checkpoint_globals()
    raw = torch.load(source, map_location="cpu", weights_only=False)
    for key, value in model.state_dict().items():
        full_key = f"model.{key}"
        if full_key not in raw["state_dict"]:
            raise RuntimeError(f"checkpoint has no state key {full_key}")
        raw["state_dict"][full_key] = value.detach().cpu().to(raw["state_dict"][full_key])
    raw["optimizer_states"] = []
    raw["lr_schedulers"] = []
    raw["derivative_remediation"] = provenance
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--subspace-reference-root", type=Path)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--all-coordinates", action="store_true")
    parser.add_argument("--snapshot-runs", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    reference_root = args.subspace_reference_root.resolve() if args.subspace_reference_root else None
    snapshot_runs = args.snapshot_runs.resolve() if args.snapshot_runs else None
    if args.joint and (reference_root is None or snapshot_runs is None):
        parser.error("--joint requires --subspace-reference-root and --snapshot-runs")
    if args.all_coordinates and not args.joint:
        parser.error("--all-coordinates requires --joint")
    protocol = freeze_protocol(
        source, output, args.steps, args.learning_rate, reference_root, args.joint,
        snapshot_runs, args.all_coordinates,
    )
    if args.plan_only:
        return 0

    base_model, batch, processor, directions, fields, backend_record = setup(
        source, output, args.backend
    )
    if args.all_coordinates:
        add_coordinate_references(directions, fields)
    raw_targets, mapping = reference_labels(base_model, batch, processor, fields)
    targets = {key: value.to(batch["positions"].device) for key, value in raw_targets.items()}
    projectors = subspace_projectors(
        mapping, len(next(iter(targets.values()))), batch["positions"].device, reference_root,
        args.joint,
    ) if reference_root else None
    training_projectors = (
        {
            name: [entry for entry in entries if entry[0] != "k_generic_2"]
            for name, entries in projectors.items()
        }
        if args.joint else projectors
    )
    if args.all_coordinates:
        training_projectors.update(coordinate_projectors(
            mapping, len(next(iter(targets.values()))), batch["positions"].device, fields
        ))
    snapshot_batches, snapshot_count = (
        load_snapshot_batches(snapshot_runs, processor, batch["positions"].device)
        if snapshot_runs else ([], 0)
    )
    anchors = snapshot_anchors(base_model, snapshot_batches)
    calibration = (
        JOINT_CALIBRATION + tuple(name for name in COORDINATE_DIRECTIONS if name != "atom0000_x")
        if args.all_coordinates else JOINT_CALIBRATION if args.joint else CALIBRATION
    )
    initial_state = copy.deepcopy(base_model.state_dict())
    candidates = []
    for weight in WEIGHTS:
        base_model.load_state_dict(initial_state)
        history = train_candidate(
            base_model, batch, processor, directions, targets, weight, args.steps, args.learning_rate,
            training_projectors, calibration, snapshot_batches, anchors,
        )
        final = history[-1]
        subspace_validation = (
            max(float(value) for value in subspace_error(
                derivative_flat(base_model, batch, directions[VALIDATION], np.asarray(processor.basis_table.change_of_basis)),
                [entry for entry in projectors[VALIDATION] if entry[0] == "k_generic_2"],
            ))
            if projectors else None
        )
        candidate = {
            "weight": weight,
            "history": history,
            "validation_relative_error": final["relative_error"][VALIDATION],
            "subspace_validation_relative_error": subspace_validation,
            "equilibrium_relative_drift": final["equilibrium_relative_drift"],
            "eligible": final["equilibrium_relative_drift"] <= 0.02,
            "state": copy.deepcopy(base_model.state_dict()),
        }
        candidates.append(candidate)
        print(json.dumps({key: value for key, value in candidate.items() if key != "state"}, sort_keys=True), flush=True)

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("all derivative weights exceeded the frozen equilibrium-drift guard")
    selected = min(
        eligible,
        key=lambda candidate: candidate["subspace_validation_relative_error"]
        if projectors else candidate["validation_relative_error"],
    )
    base_model.load_state_dict(selected.pop("state"))
    for candidate in candidates:
        candidate.pop("state", None)
    checkpoint = output / "selected_derivative_remediated.ckpt"
    provenance = {
        "schema": protocol["schema"],
        "protocol_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
        "selected_weight": selected["weight"],
    }
    save_checkpoint(source, checkpoint, base_model, provenance)
    report = {
        "schema": "graphene_gamma_derivative_remediation_v1",
        "protocol": "protocol.json",
        "backend": backend_record.to_metadata(),
        "candidates": candidates,
        "selected_weight": selected["weight"],
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": file_sha256(checkpoint),
        "snapshot_count": snapshot_count,
        "snapshot_batches_per_candidate": len(snapshot_batches),
        "status": "SMOKE_PASS",
    }
    write_json(output / "remediation_results.json", report)
    print(f"selected weight={selected['weight']}; checkpoint={checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
