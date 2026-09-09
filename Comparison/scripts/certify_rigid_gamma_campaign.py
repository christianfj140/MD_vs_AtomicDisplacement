#!/usr/bin/env python3
"""E-F_001-S43: GO-9 certification of the rigid-Gamma raw-derivative campaign.

S42 (``run_tbg_pure_graph2mat_campaign.epc_gamma_rigid``, ``docs/epc_s42_matbg_gamma_rigid_execution.md``)
implemented the S41-authorised rung of the rigid-MATBG-Gamma campaign but was
never exercised end to end -- not even with the GPU/model boundary mocked
(S42 Sec. 6, an explicit OPEN item). This ticket is that missing validation:
reproducibility across a cold-cache/warm-cache cycle, order-independence and
survival of a deliberate mid-campaign interruption, a guard that no
``synthetic_test_displacement`` artifact this stage writes is ever mistaken
for a satisfied physical dependency (``basis_response``/``eigenspace``/
``pao_covariant_response``/``g``/``phonon``), and an honest resource-budget check against
the roadmap's own preflight margins (Section X: 28 GiB VRAM / 62 GB RAM / 12%
free disk).

Every suite below runs the *real* ``epc_gamma_rigid()`` (or, for the
order/interruption suite, its own ``epc_raw_derivative_for_direction``
helper) against the real, already-frozen S41 protocol and the real (31, 30)
geometry -- only the GPU/model boundary (``load_model_and_batch``,
``resolve_jvp_backend``, ``batch_topology_hash``,
``directional_derivative_field``, ``frozen_derivative_field``) is faked, and
every run is isolated to a temporary ``EPC_ROOT``/``EPC_ARRAYS_DIR``/
``EPC_STATUS_PATH`` so this certifier can never itself pollute the real
campaign's ``artifact_status.json`` (the exact bug this ticket found and
fixed in ``tests/test_run_tbg_gamma_rigid_epc.py``, which forgot to isolate
``EPC_STATUS_PATH`` and was silently overwriting the production status file
with a synthetic ``"synthetic failure"`` entry on every test run).
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark_matbg_synthetic_displacements as matbg_bench  # noqa: E402
import run_tbg_pure_graph2mat_campaign as camp  # noqa: E402

SCHEMA = "rigid_gamma_campaign_go9_certification_v1"
TICKET = "E-F_001-S43"
MEMO = "docs/epc_s43_rigid_gamma_campaign_certification.md"
GATE = "GO-9"
FINAL_PUBLICATION_LABEL = "rigid_tbg_model"
FORBIDDEN_PUBLICATION_LABELS = ("quantitative_relaxed_matbg_prediction", "g_mn_nu", "phonon_eigenmode")

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/certification/matbg"
REPORT_NAME = "s43_rigid_gamma_campaign_go9_certification.json"

# A real, content-stable file used only as "the checkpoint" for file_sha256 --
# no model is ever loaded in this certifier (load_model_and_batch is faked).
_FAKE_CHECKPOINT_A = SCRIPT_DIR / "certify_rigid_gamma_campaign.py"
_FAKE_CHECKPOINT_B = SCRIPT_DIR / "run_tbg_pure_graph2mat_campaign.py"


class RigidGammaCampaignCertificationError(RuntimeError):
    """A GO-9 certification was built, read, or trusted in a way this gate forbids."""


def check(name: str, passed: bool, detail: str, *, applicable: bool = True, **extra: Any) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed) if applicable else True,
        "applicable": bool(applicable),
        "detail": detail,
        **extra,
    }


# --------------------------------------------------------------------------- #
# Shared mocked-stage runner: real protocol, real geometry, faked GPU/model
# --------------------------------------------------------------------------- #


class _FakeBackend:
    requested = "cpu"
    effective = "cpu"
    preflight = "not_required"
    reason = "certifier fake backend: resolve_jvp_backend's own CUDA preflight is exercised elsewhere"

    def to_metadata(self) -> dict:
        return {
            "requested_backend": self.requested,
            "effective_backend": self.effective,
            "backend_preflight": self.preflight,
            "backend_fallback_reason": self.reason,
        }


class _FakeBasisTable:
    change_of_basis = np.eye(3)


class _FakeProcessor:
    basis_table = _FakeBasisTable()


def _fake_field_for_direction(direction) -> dict[tuple[int, int, tuple[int, int, int]], float]:
    """Deterministic per-direction field: same value for JVP and every delta.

    Function of ``direction.direction_hash`` only, so it is stable across
    processes/restarts and distinct per direction -- exactly what makes
    ``jvp_equals_frozen_within_backend_margin`` pass deterministically
    (flat delta plateau, JVP == frozen exactly) without faking away the
    real per-direction bookkeeping this stage is being certified for.
    """
    seed = int(direction.direction_hash[:8], 16)
    value = 1.0 + (seed % 997) / 997.0
    return {(0, 0, (0, 0, 0)): value, (1, 0, (0, 0, 0)): -value}


def _make_fakes() -> tuple[Any, Any, dict[str, int]]:
    calls = {"jvp": 0, "frozen": 0}

    def fake_jvp(model, batch, direction, *, processor, change_of_basis, backend):
        calls["jvp"] += 1
        return _fake_field_for_direction(direction), {"backend": "certifier_fake_jvp"}

    def fake_frozen(model, batch, direction, delta, *, processor, change_of_basis):
        calls["frozen"] += 1
        return _fake_field_for_direction(direction), [f"certifier-fake-topology-{delta}"]

    return fake_jvp, fake_frozen, calls


def _run_mocked_gamma_rigid_stage(tmp_root: Path, *, jvp_fn, frozen_fn, checkpoint: Path) -> dict:
    """Runs the real ``epc_gamma_rigid()`` end to end, isolated to ``tmp_root``."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(camp, "EPC_ROOT", tmp_root))
        stack.enter_context(patch.object(camp, "EPC_ARRAYS_DIR", tmp_root / "arrays"))
        stack.enter_context(patch.object(camp, "EPC_STATUS_PATH", tmp_root / "artifact_status.json"))
        stack.enter_context(
            patch.object(camp.epc_c14, "load_model_and_batch", lambda *a, **k: (None, None, _FakeProcessor(), {}))
        )
        stack.enter_context(
            patch.object(
                camp.epc_g2m_deriv, "resolve_jvp_backend",
                lambda model, batch, requested, output_keys: (model, batch, _FakeBackend()),
            )
        )
        stack.enter_context(
            patch.object(camp.epc_g2m_deriv, "batch_topology_hash", lambda batch: {"topology_hash": "fake-topology"})
        )
        stack.enter_context(patch.object(camp.epc_c14, "directional_derivative_field", jvp_fn))
        stack.enter_context(patch.object(camp.epc_c14, "frozen_derivative_field", frozen_fn))
        return camp.epc_gamma_rigid(checkpoint, Path("unused.fdf"), Path("unused_basis"), requested_backend="cpu")


def _strip_volatile(value: Any) -> Any:
    """Drop fields that legitimately differ across runs that used a different
    temporary output directory (timestamps, the per-call ``reused`` cache-hit
    flag, the absolute ``array_path``) so two runs can be compared for the
    equality that actually matters: the substantive result data.
    """
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in ("generated_at", "updated_at", "reused", "array_path")
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# Suite A: gate authorization against the real, on-disk S41 protocol
# --------------------------------------------------------------------------- #


def _certify_gate_authorization() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = camp.epc_prereg.load_protocol()
    gate = camp.epc_authorize(protocol)
    checks.append(
        check(
            "real_protocol_authorizes_expected_rung",
            gate["result_status"] == camp.EPC_RESULT_STATUS
            and "graph2mat_derivative_validated_no_full_ks_coupling" in gate["authorised_claim"],
            f"result_status={gate['result_status']!r} authorised_claim={gate['authorised_claim'][:80]!r}...",
        )
    )

    tampered = dict(protocol)
    tampered["frozen_content_sha256"] = "tampered"
    rejected = False
    try:
        camp.epc_authorize(tampered)
    except camp.EpcGammaRigidError:
        rejected = True
    checks.append(check("tampered_protocol_is_rejected", rejected, "edited protocol must raise EpcGammaRigidError"))

    positions = camp.epc_prereg.read_xyz_positions_ang(camp.epc_prereg.DEFAULT_POSITIONS_XYZ)
    directions = camp.epc_prereg.build_candidate_directions(positions)
    frozen_sectors = protocol["phonon"]["candidate_sectors"]
    mismatches = [
        name for name, direction in directions.items()
        if name not in frozen_sectors or direction.direction_hash != frozen_sectors[name]["direction_hash"]
    ]
    checks.append(
        check(
            "rebuilt_directions_match_frozen_protocol_hashes", not mismatches,
            f"{len(directions)} directions rebuilt from the real (31,30) geometry; mismatches={mismatches}",
        )
    )

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


# --------------------------------------------------------------------------- #
# Suite B: cold-cache / warm-cache / restart-from-disk reproducibility
# --------------------------------------------------------------------------- #


def _certify_reproducibility(tmp_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    root = tmp_dir / "reproducibility"

    jvp_fn, frozen_fn, calls = _make_fakes()
    cold = _run_mocked_gamma_rigid_stage(root, jvp_fn=jvp_fn, frozen_fn=frozen_fn, checkpoint=_FAKE_CHECKPOINT_A)
    calls_after_cold = dict(calls)
    checks.append(
        check(
            "cold_cache_completes_all_directions",
            all(row["state"] == "completed" for row in cold["directions"].values()),
            f"directions={list(cold['directions'])} states={[row['state'] for row in cold['directions'].values()]}",
        )
    )

    warm = _run_mocked_gamma_rigid_stage(root, jvp_fn=jvp_fn, frozen_fn=frozen_fn, checkpoint=_FAKE_CHECKPOINT_A)
    checks.append(
        check(
            "warm_cache_reuses_without_recompute", calls == calls_after_cold,
            f"jvp/frozen call counts unchanged across warm-cache re-run: {calls_after_cold} == {calls}",
        )
    )
    checks.append(
        check(
            "cold_and_warm_results_are_identical", _strip_volatile(cold) == _strip_volatile(warm),
            "epc_gamma_rigid_summary.json content (minus timestamps) is bitwise identical cold vs warm",
        )
    )

    # A fresh in-memory reload from the persisted artifact_status.json (the
    # cheapest honest stand-in for "the process was killed and restarted"):
    # a third call should still be a pure cache hit.
    restarted = _run_mocked_gamma_rigid_stage(root, jvp_fn=jvp_fn, frozen_fn=frozen_fn, checkpoint=_FAKE_CHECKPOINT_A)
    checks.append(
        check(
            "restart_from_disk_reuses_without_recompute", calls == calls_after_cold,
            f"a third invocation against the same on-disk status still hits cache: {calls}",
        )
    )
    checks.append(
        check(
            "restart_result_matches_cold_result", _strip_volatile(cold) == _strip_volatile(restarted),
            "result after simulated restart equals the original cold-cache result",
        )
    )

    # A changed checkpoint must invalidate the signature and force recompute
    # -- proves cache hits above were signature-checked, not unconditional.
    changed = _run_mocked_gamma_rigid_stage(root, jvp_fn=jvp_fn, frozen_fn=frozen_fn, checkpoint=_FAKE_CHECKPOINT_B)
    checks.append(
        check(
            "checkpoint_change_invalidates_cache", calls["jvp"] > calls_after_cold["jvp"],
            f"jvp calls after checkpoint change: {calls_after_cold['jvp']} -> {calls['jvp']}",
        )
    )
    checks.append(
        check(
            "checkpoint_change_result_still_agrees", changed["claim"] == cold["claim"],
            "a different checkpoint recomputes but the fake fields still agree JVP==frozen",
        )
    )

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


# --------------------------------------------------------------------------- #
# Suite C: task reordering + deliberate interruption, at the per-artifact
# bookkeeping level (epc_raw_derivative_for_direction), against the real
# five S41-frozen directions.
# --------------------------------------------------------------------------- #


def _certify_reorder_and_interruption(tmp_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    positions = camp.epc_prereg.read_xyz_positions_ang(camp.epc_prereg.DEFAULT_POSITIONS_XYZ)
    directions = camp.epc_prereg.build_candidate_directions(positions)
    deltas = [float(value) for value in camp.epc_prereg.load_protocol()["perturbation"]["delta_sweep_ang"]]
    names = list(directions)

    def run_order(order: list[str], arrays_dir: Path, jvp_fn, frozen_fn) -> dict:
        status: dict = {"artifacts": {}}
        for name in order:
            camp.epc_raw_derivative_for_direction(
                name=name, direction=directions[name], model=None, batch=None, processor=_FakeProcessor(),
                change_of_basis=np.eye(3), deltas=deltas, checkpoint=_FAKE_CHECKPOINT_A, topology_hash="fake-topology",
                backend=_FakeBackend(), status=status,
            )
        return status

    with patch.object(camp, "EPC_ARRAYS_DIR", tmp_dir / "reorder_a" / "arrays"), \
         patch.object(camp, "EPC_STATUS_PATH", tmp_dir / "reorder_a" / "status.json"):
        jvp_a, frozen_a, _ = _make_fakes()
        with patch.object(camp.epc_c14, "directional_derivative_field", jvp_a), \
             patch.object(camp.epc_c14, "frozen_derivative_field", frozen_a):
            status_forward = run_order(names, tmp_dir / "reorder_a" / "arrays", jvp_a, frozen_a)

    with patch.object(camp, "EPC_ARRAYS_DIR", tmp_dir / "reorder_b" / "arrays"), \
         patch.object(camp, "EPC_STATUS_PATH", tmp_dir / "reorder_b" / "status.json"):
        jvp_b, frozen_b, _ = _make_fakes()
        with patch.object(camp.epc_c14, "directional_derivative_field", jvp_b), \
             patch.object(camp.epc_c14, "frozen_derivative_field", frozen_b):
            status_reversed = run_order(list(reversed(names)), tmp_dir / "reorder_b" / "arrays", jvp_b, frozen_b)

    checks.append(
        check(
            "reordered_processing_yields_identical_artifacts",
            _strip_volatile(status_forward) == _strip_volatile(status_reversed),
            f"forward order {names} vs reversed order {list(reversed(names))} produce identical "
            "per-direction artifacts (each keyed independently, no shared mutable state)",
        )
    )

    # Deliberate interruption: direction[0] completes; direction[1] "crashes"
    # (simulated exception), status is persisted to disk mid-campaign; a
    # fresh in-memory status (simulated restart) resumes and completes
    # direction[1] without recomputing direction[0] or diverging from the
    # forward-order result above.
    interrupt_dir = tmp_dir / "interruption"
    status_path = interrupt_dir / "status.json"
    with patch.object(camp, "EPC_ARRAYS_DIR", interrupt_dir / "arrays"), patch.object(camp, "EPC_STATUS_PATH", status_path):
        jvp_c, frozen_c, calls_c = _make_fakes()
        with patch.object(camp.epc_c14, "directional_derivative_field", jvp_c), \
             patch.object(camp.epc_c14, "frozen_derivative_field", frozen_c):
            status_before_crash: dict = {"artifacts": {}}
            first = camp.epc_raw_derivative_for_direction(
                name=names[0], direction=directions[names[0]], model=None, batch=None, processor=_FakeProcessor(),
                change_of_basis=np.eye(3), deltas=deltas, checkpoint=_FAKE_CHECKPOINT_A, topology_hash="fake-topology",
                backend=_FakeBackend(), status=status_before_crash,
            )
            calls_after_first = dict(calls_c)

        def boom(*args, **kwargs):
            raise RuntimeError("deliberate certifier interruption")

        with patch.object(camp.epc_c14, "directional_derivative_field", boom), \
             patch.object(camp.epc_c14, "frozen_derivative_field", frozen_c):
            crashed = camp.epc_raw_derivative_for_direction(
                name=names[1], direction=directions[names[1]], model=None, batch=None, processor=_FakeProcessor(),
                change_of_basis=np.eye(3), deltas=deltas, checkpoint=_FAKE_CHECKPOINT_A, topology_hash="fake-topology",
                backend=_FakeBackend(), status=status_before_crash,
            )
        checks.append(check("interrupted_direction_recorded_failed", crashed["state"] == "failed", crashed.get("error", "")))

        # Simulated restart: forget the in-memory dict, reload from disk.
        status_restarted = camp.read_json(status_path)
        checks.append(
            check(
                "restart_preserves_completed_sibling_untouched",
                status_restarted["artifacts"][f"raw_derivative__{names[0]}"]["state"] == "completed",
                "the direction that completed before the crash is still 'completed' after restart",
            )
        )
        with patch.object(camp.epc_c14, "directional_derivative_field", jvp_c), \
             patch.object(camp.epc_c14, "frozen_derivative_field", frozen_c):
            resumed_first = camp.epc_raw_derivative_for_direction(
                name=names[0], direction=directions[names[0]], model=None, batch=None, processor=_FakeProcessor(),
                change_of_basis=np.eye(3), deltas=deltas, checkpoint=_FAKE_CHECKPOINT_A, topology_hash="fake-topology",
                backend=_FakeBackend(), status=status_restarted,
            )
            calls_after_resumed_first = dict(calls_c)
            recovered = camp.epc_raw_derivative_for_direction(
                name=names[1], direction=directions[names[1]], model=None, batch=None, processor=_FakeProcessor(),
                change_of_basis=np.eye(3), deltas=deltas, checkpoint=_FAKE_CHECKPOINT_A, topology_hash="fake-topology",
                backend=_FakeBackend(), status=status_restarted,
            )
        checks.append(
            check(
                "resumed_completed_direction_not_recomputed", calls_after_resumed_first == calls_after_first,
                f"jvp/frozen call counts unchanged for the already-completed direction across restart: "
                f"{calls_after_first} == {calls_after_resumed_first}",
            )
        )
        checks.append(
            check(
                "crashed_direction_recomputed_after_restart", calls_c["jvp"] > calls_after_resumed_first["jvp"],
                f"the direction that failed is recomputed (not falsely cache-hit) after restart: "
                f"jvp {calls_after_resumed_first['jvp']} -> {calls_c['jvp']}",
            )
        )
        checks.append(
            check(
                "recovered_direction_matches_uninterrupted_result",
                _strip_volatile(recovered) == _strip_volatile(status_forward["artifacts"][f"raw_derivative__{names[1]}"]),
                "the direction that crashed then resumed produces the same result as an uninterrupted run",
            )
        )
        checks.append(
            check(
                "resumed_first_direction_matches_uninterrupted_result",
                _strip_volatile(resumed_first) == _strip_volatile(status_forward["artifacts"][f"raw_derivative__{names[0]}"]),
                "the pre-crash direction still matches the uninterrupted-run result after restart",
            )
        )

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


# --------------------------------------------------------------------------- #
# Suite D: no synthetic artifact satisfies a physical dependency
# --------------------------------------------------------------------------- #


def _certify_synthetic_dependency_guard(tmp_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    jvp_fn, frozen_fn, _ = _make_fakes()
    summary = _run_mocked_gamma_rigid_stage(
        tmp_dir / "synthetic_guard", jvp_fn=jvp_fn, frozen_fn=frozen_fn, checkpoint=_FAKE_CHECKPOINT_A
    )

    blocked_not_attempted = {
        key: entry.get("status") for key, entry in summary["blocked"].items()
    }
    checks.append(
        check(
            "physical_dependencies_stay_not_attempted",
            all(status == "not_attempted" for status in blocked_not_attempted.values())
            and set(blocked_not_attempted) == {"basis_response", "eigenspace", "pao_covariant_response", "g", "phonon"},
            f"blocked={blocked_not_attempted}",
        )
    )
    checks.append(
        check(
            "top_level_artifact_kind_is_synthetic", summary["artifact_kind"] == "synthetic_test_displacement",
            f"artifact_kind={summary['artifact_kind']!r}",
        )
    )
    per_direction_kinds = {name: row.get("artifact_kind") for name, row in summary["directions"].items()}
    checks.append(
        check(
            "every_direction_artifact_kind_is_synthetic",
            all(kind == "synthetic_test_displacement" for kind in per_direction_kinds.values()),
            f"per_direction_artifact_kind={per_direction_kinds}",
        )
    )
    checks.append(
        check(
            "forbidden_publication_labels_are_declared",
            set(FORBIDDEN_PUBLICATION_LABELS) <= set(summary["forbidden_publication_labels"]),
            f"forbidden_publication_labels={summary['forbidden_publication_labels']}",
        )
    )
    # The forbidden labels are only allowed to appear inside the
    # forbidden-label list itself, never as the live claim/result_status.
    live_claim_leak = summary["claim"] in FORBIDDEN_PUBLICATION_LABELS or summary["result_status"] in FORBIDDEN_PUBLICATION_LABELS
    checks.append(
        check(
            "forbidden_labels_never_used_as_a_live_claim", not live_claim_leak,
            f"claim={summary['claim']!r} result_status={summary['result_status']!r}",
        )
    )
    checks.append(
        check(
            "result_status_stays_candidate_never_final",
            summary["result_status"] == "candidate_rigid_tbg" and summary["result_status"] != FINAL_PUBLICATION_LABEL,
            f"result_status={summary['result_status']!r} (final publication label {FINAL_PUBLICATION_LABEL!r} is a "
            "separate, later step -- roadmap S41 Sec. 7)",
        )
    )
    checks.append(
        check(
            "claim_never_exceeds_the_authorised_rung",
            summary["claim"] in ("NO_GO", summary["preregistration"]["authorised_claim"]),
            f"claim={summary['claim']!r}",
        )
    )

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


# --------------------------------------------------------------------------- #
# Suite E: resource budget -- honestly blocked, no real GPU run exists yet
# --------------------------------------------------------------------------- #


def _certify_resource_budget() -> dict[str, Any]:
    margins = matbg_bench.resource_margins()
    reason = (
        "no real GPU run of epc_gamma_rigid() against the production (31,30) checkpoint has ever "
        "been measured (docs/epc_s42_matbg_gamma_rigid_execution.md Sec. 6, explicit OPEN item); "
        "the roadmap's own compute policy requires a one-direction-at-a-time preflight before a "
        "five-direction batch, not a resource claim fabricated from live machine headroom"
    )
    return {
        "status": "NOT_RUN",
        "blocked_by": "S42 Sec. 6 (no measured production run)",
        "checks": [
            check(
                "campaign_peak_resource_usage_within_roadmap_margins", True, reason, applicable=False,
                requirement="roadmap Section X preflight (28 GiB VRAM / 62 GB RAM / 12% free disk)",
            )
        ],
        "live_machine_headroom_snapshot": margins,
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def certify_rigid_gamma_campaign_go9(tmp_dir: Path) -> dict[str, Any]:
    gate_authorization = _certify_gate_authorization()
    reproducibility = _certify_reproducibility(tmp_dir)
    reorder_and_interruption = _certify_reorder_and_interruption(tmp_dir)
    synthetic_dependency_guard = _certify_synthetic_dependency_guard(tmp_dir)
    resource_budget = _certify_resource_budget()

    suites = {
        "gate_authorization_suite": gate_authorization,
        "reproducibility_suite": reproducibility,
        "reorder_and_interruption_suite": reorder_and_interruption,
        "synthetic_dependency_guard_suite": synthetic_dependency_guard,
        "resource_budget_suite": resource_budget,
    }
    go9 = all(suite["status"] == "PASS" for suite in suites.values())
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **suites,
        "go9_status": "GO-9" if go9 else "NO-GO-9",
        "rigid_gamma_campaign_certified": bool(go9),
        "final_publication_label": FINAL_PUBLICATION_LABEL,
        "forbidden_publication_labels": list(FORBIDDEN_PUBLICATION_LABELS),
    }


def require_go9_certified_rigid_gamma_campaign(certification: dict) -> dict:
    if certification.get("gate") != GATE or certification.get("go9_status") != "GO-9":
        raise RigidGammaCampaignCertificationError(
            f"GO-9 is not certified ({certification.get('go9_status')!r}): no rigid-Gamma MATBG "
            f"result may be published as {FINAL_PUBLICATION_LABEL!r}, and "
            f"{FORBIDDEN_PUBLICATION_LABELS!r} stay forbidden regardless"
        )
    return certification


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="s43_rigid_gamma_certify_") as tmpdir:
        certification = certify_rigid_gamma_campaign_go9(Path(tmpdir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / REPORT_NAME
    output.write_text(json.dumps(_json_safe(certification), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"S43 rigid-Gamma campaign {GATE} certification: {certification['go9_status']}  ->  {output}")
    for suite_name in (
        "gate_authorization_suite", "reproducibility_suite", "reorder_and_interruption_suite",
        "synthetic_dependency_guard_suite", "resource_budget_suite",
    ):
        suite = certification[suite_name]
        print(f"  {suite_name}: {suite['status']}")
        for entry in suite["checks"]:
            state = "n/a " if not entry["applicable"] else ("ok  " if entry["passed"] else "FAIL")
            print(f"    [{state}] {entry['check']}: {entry['detail']}")
    return 0 if certification["go9_status"] == "GO-9" else 1


if __name__ == "__main__":
    raise SystemExit(main())
