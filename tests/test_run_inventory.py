"""Fase 0: reproducible run inventory (repo SHAs, dirty state, import paths).

Second half: C01 / E-F_001-S1, the frozen identity of the SIESTA runtime.
No test here launches SIESTA; every probe runs against a shell fixture.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from run_inventory import (  # noqa: E402
    DO_NOT_REUSE,
    GO2_FOLLOWUP,
    PROBE_ENV_KEYS,
    SIESTA_RUNTIME_REF_SCHEMA,
    SIESTA_RUNTIME_SCHEMA,
    UPSTREAM_EXPECTATIONS,
    canonical_record_bytes,
    collect_run_inventory,
    collect_siesta_runtime,
    git_repository_state,
    module_import_state,
    reproducibility_status,
    resolve_executable_chain,
    validate_siesta_runtime_ref,
    write_siesta_runtime_record,
)

X11_NOISE = "Authorization required, but no authorization protocol specified"
BANNER = (
    "Executable      : siesta\n"
    "Version         : 5.4.2-11-g4e9a46060\n"
    "Architecture    : x86_64\n"
    "Compiler version: GNU-13.3.0\n"
    "Compiler flags  : -fallow-argument-mismatch -O3 -march=native\n"
    "Parallelisations: MPI\n"
)


def _fixture_binary(path: Path, stdout: str, *, stderr: str = "", extra: str = "") -> Path:
    """A fake `siesta` whose --version prints a controlled banner."""
    path.write_text(
        "#!/bin/sh\n"
        f"cat <<'EOF'\n{stdout}\nEOF\n"
        f"cat >&2 <<'EOF'\n{stderr}\nEOF\n"
        f"{extra}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _upstream_fetch(token: str, *, override: dict | None = None):
    """Fake GitLab API returning the real, verified upstream answers."""
    facts = UPSTREAM_EXPECTATIONS[token]
    tag_commit = (override or {}).get("tag_commit", facts["tag_commit"])
    first_parent = [facts["commit"]] + ["0" * 40] * 10 + [tag_commit]

    def fetch(endpoint: str):
        if endpoint.startswith("/commits?"):
            return [{"id": sha} for sha in first_parent], None
        if endpoint.startswith("/commits/"):
            return {"id": facts["commit"]}, None
        if endpoint.startswith("/tags/"):
            return {"commit": {"id": tag_commit}}, None
        if endpoint.startswith("/merge_base"):
            return {"id": facts["ancestor_commit"]}, None
        if endpoint.startswith("/files/"):
            return {"blob_id": facts["dhsdr_blob"]}, None
        raise AssertionError(f"unexpected endpoint {endpoint}")

    return fetch


def _offline_fetch(endpoint: str):
    return None, "URLError('no network')"


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "a.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        check=True,
    )


def test_clean_repo_is_pinned_clean(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    state = git_repository_state(repo)
    assert state["commit"] and len(state["commit"]) == 40
    assert state["dirty"] is False
    assert reproducibility_status({"r": state}) == "pinned_clean"


def test_dirty_repo_is_pinned_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "b.txt").write_text("dirty", encoding="utf-8")
    state = git_repository_state(repo)
    assert state["dirty"] is True
    assert reproducibility_status({"r": state}) == "pinned_dirty"


def test_non_git_directory_is_unpinned(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    state = git_repository_state(plain)
    assert state["error"] == "not_git_repo"
    assert state["commit"] is None
    assert reproducibility_status({"r": state}) == "unpinned"


def test_missing_path_is_unavailable(tmp_path):
    state = git_repository_state(tmp_path / "nope")
    assert state["error"] == "path_missing"
    assert reproducibility_status({"r": state}) == "unavailable"


def test_import_from_unexpected_path_is_flagged(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    inventory = collect_run_inventory({"MD_vs_AtomicDisplacement": REPO_ROOT, "graph2mat": repo})
    g2m = inventory["imports"]["graph2mat"]
    # graph2mat really imports from its own checkout, not from tmp repo.
    if g2m.get("module_path"):
        assert g2m["matches_inspected_repo"] is False
        assert inventory.get("warnings")


def test_module_import_state_reports_this_interpreter():
    state = module_import_state("json")
    assert state["module_path"].endswith("json/__init__.py")


def test_full_inventory_has_required_blocks():
    inventory = collect_run_inventory()
    assert inventory["schema"] == "run_inventory_v1"
    assert set(inventory["repositories"]) == {"MD_vs_AtomicDisplacement", "graph2mat", "DeepH-pack"}
    for state in inventory["repositories"].values():
        assert "commit" in state and "dirty" in state
    assert inventory["python"]["executable"]
    assert inventory["reproducibility_status"] in {
        "pinned_clean",
        "pinned_dirty",
        "unpinned",
        "unavailable",
    }


# --------------------------------------------------------------------------
# C01 / E-F_001-S1: frozen SIESTA runtime identity
# --------------------------------------------------------------------------


@pytest.fixture()
def hermetic_path(tmp_path, monkeypatch):
    """No real `siesta` on PATH, but coreutils still reachable for the fixtures."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    return tmp_path


def test_siesta_binary_resolution_hash_and_mutation(hermetic_path):
    tmp_path = hermetic_path
    real = _fixture_binary(tmp_path / "siesta.real", BANNER)
    (tmp_path / "hop").symlink_to("siesta.real")  # relative hop
    (tmp_path / "siesta").symlink_to(tmp_path / "hop")  # absolute hop
    chain = resolve_executable_chain(tmp_path / "siesta")
    assert chain["symlink_cycle"] is False
    assert chain["chain"] == [str(tmp_path / "siesta"), str(tmp_path / "hop"), str(real)]
    assert chain["final_target"] == str(real)

    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    cycle = resolve_executable_chain(tmp_path / "a")
    assert cycle["symlink_cycle"] is True and cycle["final_target"] is None

    # Rewrites itself while being probed: the two hashes must disagree.
    mutating = _fixture_binary(
        tmp_path / "mutating", BANNER, extra='printf "#\\n" >> "$0"'
    )
    record = collect_siesta_runtime(mutating, repo_root=tmp_path)
    assert record["binary"]["sha256_before_probes"] != record["binary"]["sha256_after_probes"]
    assert record["binary"]["sha256"] is None
    assert record["classification"]["status"] == "UNVERIFIED"
    assert "binary_changed_during_collection" in record["classification"]["reasons"]
    assert DO_NOT_REUSE in record["required_followup"]


def test_siesta_strict_probe_captures_bytes_and_rejects_noise(hermetic_path):
    tmp_path = hermetic_path
    noisy = _fixture_binary(
        tmp_path / "siesta", BANNER, stderr="\n".join([X11_NOISE] * 4)
    )
    record = collect_siesta_runtime(noisy, repo_root=tmp_path)
    probe = record["probe"]
    attempt = probe["attempts"][-1]

    assert probe["strict"]["valid"] is True
    assert probe["strict"]["version"] == "5.4.2-11-g4e9a46060"
    assert probe["strict"]["build_info"]["Compiler version"] == "GNU-13.3.0"
    assert attempt["argv"] == [str(noisy), "--version"]
    assert attempt["cwd_inside_repository"] is False
    assert attempt["env"]["values"]["LC_ALL"] == "C"
    assert set(attempt["env"]["values"]) <= set(PROBE_ENV_KEYS)
    # X11 noise survives byte for byte; it is never normalised away.
    assert base64.b64decode(attempt["stderr_b64"]).decode() == "\n".join([X11_NOISE] * 4) + "\n"
    assert base64.b64decode(attempt["stdout_b64"]).decode() == BANNER + "\n"

    only_noise = _fixture_binary(tmp_path / "noise", X11_NOISE, stderr=X11_NOISE)
    strict = collect_siesta_runtime(only_noise, repo_root=tmp_path)["probe"]["strict"]
    assert strict["valid"] is False and "no_valid_version_line" in strict["reasons"]

    bare = _fixture_binary(tmp_path / "bare", "Version         : 5.4.2-11-g4e9a46060")
    strict = collect_siesta_runtime(bare, repo_root=tmp_path)["probe"]["strict"]
    assert strict["valid"] is False and "build_information_block_missing" in strict["reasons"]


def test_siesta_version_string_never_promotes_to_verified_source(hermetic_path):
    tmp_path = hermetic_path
    binary = _fixture_binary(tmp_path / "siesta", BANNER)
    token = "5.4.2-11-g4e9a46060"

    record = collect_siesta_runtime(
        binary,
        repo_root=tmp_path,
        verify_upstream=True,
        fetch=_upstream_fetch(token),
        attestation={"claims": "built from 5.4.2-11", "signature": "trust me"},
    )
    upstream = record["source_link"]["upstream"]
    assert upstream["status"] == "consistent"
    assert all(check["status"] == "match" for check in upstream["checks"])
    # Complete upstream evidence + an attestation nobody can verify: still binary-only.
    assert record["source_link"]["attestation"]["trusted"] is False
    assert record["source_link"]["verdict"] == "unlinked"
    assert record["classification"]["status"] == "VERIFIED_BINARY_ONLY"
    assert record["source_candidate"]["evidence_class"] == "embedded_version_string_only"

    offline = collect_siesta_runtime(
        binary, repo_root=tmp_path, verify_upstream=True, fetch=_offline_fetch
    )
    assert offline["source_link"]["upstream"]["status"] == "unavailable"
    assert offline["classification"]["status"] == "VERIFIED_BINARY_ONLY"

    contradicted = collect_siesta_runtime(
        binary,
        repo_root=tmp_path,
        verify_upstream=True,
        fetch=_upstream_fetch(token, override={"tag_commit": "f" * 40}),
    )
    assert contradicted["source_link"]["upstream"]["status"] == "contradicted"
    assert contradicted["source_link"]["verdict"] == "contradicted"
    assert contradicted["classification"]["status"] == "UNVERIFIED"
    assert "upstream_evidence_contradicted" in contradicted["classification"]["reasons"]


def test_siesta_binary_only_carries_go2_obligations_and_unobserved_expectations(hermetic_path):
    tmp_path = hermetic_path
    record = collect_siesta_runtime(
        _fixture_binary(tmp_path / "siesta", BANNER), repo_root=tmp_path
    )
    assert record["classification"]["status"] == "VERIFIED_BINARY_ONLY"
    assert record["required_followup"] == list(GO2_FOLLOWUP)
    assert DO_NOT_REUSE not in record["required_followup"]
    expectations = record["source_candidate"]["expectations"]
    assert {item["id"] for item in expectations} >= {
        "DHDR_DSDR_UNITS", "DHSDR_LAYOUT", "DHDR_SIGN_POST_5CFDA23A",
        "FC_DHDR_TOLERANCE_IS_NOOP", "DSDR_ORBITAL_FILTER", "DHSDR_FILENAME",
    }
    assert all(item["observed_on_binary"] is False for item in expectations)
    assert record["compute_policy"]["effective_backend"] == "not_applicable"
    assert record["libraries"]["linkage_closure"] == "ldd_static_only"


def test_siesta_probe_never_falls_back_to_ambient_secrets(hermetic_path, monkeypatch):
    tmp_path = hermetic_path
    monkeypatch.setenv("TOP_SECRET_TOKEN", "must-not-reach-the-probe")
    binary = _fixture_binary(
        tmp_path / "siesta", BANNER,
        extra='[ -z "$TOP_SECRET_TOKEN" ] && exit 7',
    )

    record = collect_siesta_runtime(binary, repo_root=tmp_path)

    assert record["probe"]["env_mode"] == "minimal"
    assert len(record["probe"]["attempts"]) == 1
    assert record["probe"]["attempts"][0]["returncode"] == 7
    assert "TOP_SECRET_TOKEN" not in json.dumps(record)
    assert "must-not-reach-the-probe" not in json.dumps(record)


def test_siesta_record_deterministic_idempotent_and_ref_tamper_detected(hermetic_path):
    tmp_path = hermetic_path
    binary = _fixture_binary(tmp_path / "siesta", BANNER)
    first = collect_siesta_runtime(binary, repo_root=tmp_path)
    second = collect_siesta_runtime(binary, repo_root=tmp_path)
    assert first["record_sha256"] == second["record_sha256"]
    assert first == second

    ref = write_siesta_runtime_record(first, "provenance", repo_root=tmp_path)
    stored = tmp_path / ref["record_path"]
    assert ref["schema"] == SIESTA_RUNTIME_REF_SCHEMA
    assert stored.read_bytes() == canonical_record_bytes(first)
    before = stored.stat().st_mtime_ns
    assert write_siesta_runtime_record(second, "provenance", repo_root=tmp_path) == ref
    assert stored.stat().st_mtime_ns == before  # identical rewrite is a no-op

    checked = validate_siesta_runtime_ref(
        ref, effective_executable=binary, repo_root=tmp_path
    )
    assert checked["valid"] is True and checked["effective_binary_checked"] is True
    assert checked["record"]["schema"] == SIESTA_RUNTIME_SCHEMA

    unchecked = validate_siesta_runtime_ref(ref, repo_root=tmp_path)
    assert unchecked["valid"] is False
    assert "effective_executable_not_supplied" in unchecked["reasons"]

    record_only = validate_siesta_runtime_ref(
        ref, require_effective=False, repo_root=tmp_path
    )
    assert record_only["valid"] is True
    assert "effective_executable_not_supplied" in record_only["warnings"]

    wrong_binary = validate_siesta_runtime_ref(
        ref, effective_sha256="0" * 64, repo_root=tmp_path
    )
    assert wrong_binary["valid"] is False
    assert "effective_binary_sha256_mismatch" in wrong_binary["reasons"]

    tampered_ref = dict(ref, binary_sha256="0" * 64)
    assert "ref_binary_sha256_mismatch" in validate_siesta_runtime_ref(
        tampered_ref, repo_root=tmp_path
    )["reasons"]

    payload = json.loads(stored.read_text(encoding="utf-8"))
    payload["classification"]["status"] = "VERIFIED_SOURCE"
    stored.write_bytes(canonical_record_bytes(payload))
    tampered = validate_siesta_runtime_ref(ref, repo_root=tmp_path)
    assert tampered["valid"] is False
    assert "record_sha256_mismatch" in tampered["reasons"]


def test_run_inventory_legacy_cli_and_ops_inventory_unchanged(hermetic_path):
    tmp_path = hermetic_path
    inventory = collect_run_inventory()
    assert set(inventory) >= {"schema", "repositories", "python", "imports", "reproducibility_status"}

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "shared" / "run_inventory.py")],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "shared")},
    )
    assert json.loads(completed.stdout)["schema"] == "run_inventory_v1"

    sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts" / "ops"))
    from write_reproducibility_inventory import siesta_runtime_views  # noqa: PLC0415

    assert siesta_runtime_views(None)["siesta_runtime_validation"] is None
    views = siesta_runtime_views(
        str(_fixture_binary(tmp_path / "siesta", BANNER)), output_dir=tmp_path / "prov"
    )
    assert set(views["siesta"]) == {"name", "path", "resolved_path", "sha256", "version_probe"}
    assert set(views["siesta"]["version_probe"]) == {"command", "returncode", "stdout", "stderr"}
    assert set(views["siesta_linkage"]) == {"command", "returncode", "stdout", "stderr"}
    assert views["siesta"]["version_probe"]["stdout"].startswith("Executable")
    assert views["siesta_runtime_provenance"]["schema"] == SIESTA_RUNTIME_REF_SCHEMA
    assert views["siesta_runtime_validation"] == {
        "valid": True,
        "reasons": [],
        "warnings": [],
        "effective_binary_checked": True,
    }


# ---------------------------------------------------------------------------
# C02 / E-F_001-S2: EPC reuse inventory
# ---------------------------------------------------------------------------

from run_inventory import (  # noqa: E402
    EPC_INVENTORY_SCHEMA,
    INVALID,
    MISSING,
    PRESENT_UNVERIFIED,
    PRESENT_VALID,
    classify_epc_artifact,
    collect_epc_inventory,
    epc_artifact_status,
    epc_checkpoint_records,
    epc_stencil_records,
)

SINGLE_POINT_FDF = "SystemLabel s\nMD.TypeOfRun                     CG\nMD.NumCGsteps                    0\n"
MD_FDF = (
    "SystemLabel s\nMD.TypeOfRun Verlet\nMD.Steps 12\nLua.Script md_store.lua\n"
)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _stencil(root: Path, *, fdf: str, sample_id: str = "s0", declared_sha: str | None = None) -> Path:
    """One derivative stencil manifest with a single sample."""
    sample_dir = root / "structures" / sample_id
    run_fdf = _write(sample_dir / "RUN.fdf", fdf)
    _write(
        sample_dir / "metadata.json",
        json.dumps({"materialized_fdf_sha256": declared_sha or _sha(run_fdf), "single_point": True}),
    )
    manifest = root / "derivative_stencil_manifest.json"
    _write(
        manifest,
        json.dumps(
            {
                "expected_total_structure_samples": 1,
                "samples": [{"sample_id": sample_id, "sample_dir": str(sample_dir)}],
            }
        ),
    )
    return manifest


def _campaign(repo: Path, *, hamiltonian_consumer_sha: str | None = None) -> Path:
    """A minimal but structurally real TBG campaign tree."""
    campaign = repo / "Comparison" / "results" / "tbg_pure_graph2mat"
    checkpoint = _write(repo / "Comparison" / "results" / "train" / "best.ckpt", "weights")
    overlaps = _write(campaign / "overlap" / "overlaps.h5", "S-matrix")
    run_fdf = _write(campaign / "overlap" / "RUN.fdf", SINGLE_POINT_FDF)
    orb_indx = _write(campaign / "overlap" / "s.ORB_INDX", "orbitals")
    hamiltonian = _write(campaign / "prediction" / "solver_input" / "h.h5", "H-matrix")
    source = _write(campaign / "prediction" / "raw" / "pred" / "ML_prediction.HSX", "raw")

    _write(campaign / "status.json", json.dumps({"checkpoint": str(checkpoint)}))
    _write(
        campaign / "precision_gate.json",
        json.dumps({"status": "passed", "checkpoint": "best.ckpt", "score_eV": 0.003, "threshold_eV": 0.01}),
    )
    _write(
        campaign / "prediction" / "raw" / "prediction_summary.json",
        json.dumps({"checkpoint": str(checkpoint)}),
    )
    _write(
        campaign / "prediction" / "solver_input" / "hamiltonians_pred.manifest.json",
        json.dumps(
            {
                "output": str(hamiltonian),
                "output_sha256": _sha(hamiltonian),
                "source": str(source),
                "source_sha256": _sha(source),
                "orb_indx": str(orb_indx),
            }
        ),
    )
    _write(
        campaign / "overlap" / "overlap_manifest.json",
        json.dumps(
            {
                "campaign_contract": "geometry_plus_exact_overlap_no_reference_hamiltonian",
                "identity_overlap_used": False,
                "reference_hamiltonian_generated": False,
                "diagnostics": {"status": "valid", "no_identity_overlap": True, "source": "TS.onlyS"},
                "effective_run_fdf": str(run_fdf),
                "effective_run_fdf_sha256": _sha(run_fdf),
                "export": {"overlaps_h5": str(overlaps), "overlaps_h5_sha256": _sha(overlaps)},
            }
        ),
    )
    _write(
        campaign / "spectra" / "dos" / "solver_manifest.json",
        json.dumps(
            {
                "hamiltonian_sha256": hamiltonian_consumer_sha or _sha(hamiltonian),
                "overlap_sha256": _sha(overlaps),
            }
        ),
    )
    _write(
        campaign / "summary" / "spectral_results.json",
        json.dumps({"spectra": [{"projection": {"eigenvectors_persisted": False}}]}),
    )
    _write(campaign / "target" / "moire_geometry.json", json.dumps({"twist_deg": 1.084549}))
    return campaign


def test_declared_hash_that_matches_is_the_only_road_to_present_valid(tmp_path):
    artifact = _write(tmp_path / "a.h5", "payload")
    valid = classify_epc_artifact(
        "hamiltonian_H", artifact, path_source="manifest:m.json#output",
        declared_sha256=_sha(artifact), declared_by="m.json#output_sha256", repo_root=tmp_path,
    )
    assert valid["status"] == PRESENT_VALID
    assert valid["observed_sha256"] == _sha(artifact)

    undeclared = classify_epc_artifact(
        "hamiltonian_H", artifact, path_source="manifest:m.json#output", repo_root=tmp_path
    )
    assert undeclared["status"] == PRESENT_UNVERIFIED

    mismatch = classify_epc_artifact(
        "hamiltonian_H", artifact, path_source="manifest:m.json#output",
        declared_sha256="0" * 64, repo_root=tmp_path,
    )
    assert mismatch["status"] == INVALID

    absent = classify_epc_artifact(
        "hamiltonian_H", tmp_path / "gone.h5", path_source="manifest:m.json#output",
        declared_sha256="0" * 64, repo_root=tmp_path,
    )
    assert absent["status"] == MISSING


def test_corroborating_evidence_can_invalidate_but_never_promote():
    from run_inventory import _evidence

    corroborating = [_evidence("precision_gate_passed", "pass", "0.003 eV")]
    assert epc_artifact_status(corroborating) == PRESENT_UNVERIFIED

    hashed = [_evidence("sha256_matches_declared", "pass", "", verifies_content=True)]
    assert epc_artifact_status(hashed) == PRESENT_VALID
    assert epc_artifact_status(hashed + [_evidence("precision_gate_passed", "fail", "")]) == INVALID


def test_checkpoint_without_manifest_is_unverified_and_failed_gate_invalidates(tmp_path):
    campaign = _campaign(tmp_path)
    record = epc_checkpoint_records(tmp_path, campaign)[0]
    assert record["status"] == PRESENT_UNVERIFIED
    assert record["declared_sha256"] is None
    checks = {item["check"]: item["outcome"] for item in record["evidence"]}
    assert checks["declared_sha256"] == "absent"
    assert checks["precision_gate_passed"] == "pass"

    checkpoint = Path(json.loads((campaign / "status.json").read_text())["checkpoint"])
    _write(
        checkpoint.parent / "checkpoint_manifest.json",
        json.dumps({"checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha(checkpoint)}),
    )
    assert epc_checkpoint_records(tmp_path, campaign)[0]["status"] == PRESENT_VALID

    _write(
        campaign / "precision_gate.json",
        json.dumps({"status": "failed", "checkpoint": "best.ckpt", "score_eV": 0.9, "threshold_eV": 0.01}),
    )
    failed = epc_checkpoint_records(tmp_path, campaign)[0]
    assert failed["status"] == INVALID
    assert "precision_gate_passed" in [item["check"] for item in failed["evidence"] if item["outcome"] == "fail"]


def test_checkpoint_with_incompatible_hash_is_never_reusable(tmp_path):
    campaign = _campaign(tmp_path)
    checkpoint = Path(json.loads((campaign / "status.json").read_text())["checkpoint"])
    _write(
        checkpoint.parent / "checkpoint_manifest.json",
        json.dumps({"checkpoint_path": str(checkpoint), "checkpoint_sha256": "0" * 64}),
    )
    record = epc_checkpoint_records(tmp_path, campaign)[0]
    assert record["status"] == INVALID
    inventory = collect_epc_inventory(
        repo_root=tmp_path, campaign_root=campaign, include_run_inventory=False
    )
    assert "tbg_spectral_loss_checkpoint" not in [item["label"] for item in inventory["reuse"]]


def test_md_contaminated_stencil_is_invalid_and_single_point_is_valid(tmp_path):
    _stencil(tmp_path / "Comparison" / "results" / "clean", fdf=SINGLE_POINT_FDF)
    _stencil(tmp_path / "Comparison" / "results" / "verlet", fdf=MD_FDF)
    _stencil(tmp_path / "Comparison" / "results" / "tampered", fdf=SINGLE_POINT_FDF, declared_sha="0" * 64)

    by_label = {
        Path(record["label"]).parent.name: record for record in epc_stencil_records(tmp_path)
    }
    assert by_label["clean"]["status"] == PRESENT_VALID
    assert by_label["clean"]["members"] == {"total": 1, "by_status": {PRESENT_VALID: 1}}

    assert by_label["verlet"]["status"] == INVALID
    contamination = [item["detail"] for item in by_label["verlet"]["evidence"] if item["outcome"] == "fail"]
    assert any("md_contaminated_reference" in detail for detail in contamination)

    assert by_label["tampered"]["status"] == INVALID
    assert any(
        "materialized_fdf_sha256_mismatch" in item["detail"]
        for item in by_label["tampered"]["evidence"]
        if item["outcome"] == "fail"
    )


def test_consumer_disagreeing_with_producer_invalidates_the_hamiltonian(tmp_path):
    campaign = _campaign(tmp_path, hamiltonian_consumer_sha="0" * 64)
    inventory = collect_epc_inventory(
        repo_root=tmp_path, campaign_root=campaign, include_run_inventory=False
    )
    record = next(item for item in inventory["artifacts"] if item["label"] == "tbg_predicted_hamiltonian_h5")
    assert record["status"] == INVALID
    assert "solver_consumers_agree_on_hamiltonian" in [
        item["check"] for item in record["evidence"] if item["outcome"] == "fail"
    ]


def test_inventory_enumerates_reuse_invalidation_and_absence(tmp_path):
    campaign = _campaign(tmp_path)
    _stencil(tmp_path / "Comparison" / "results" / "clean", fdf=SINGLE_POINT_FDF)
    _stencil(tmp_path / "Comparison" / "results" / "verlet", fdf=MD_FDF)

    inventory = collect_epc_inventory(
        repo_root=tmp_path, campaign_root=campaign, include_run_inventory=False
    )
    assert inventory["schema"] == EPC_INVENTORY_SCHEMA
    labels = {item["label"] for item in inventory["reuse"]}
    assert {"tbg_predicted_hamiltonian_h5", "tbg_exact_overlap_h5", "tbg_overlap_effective_run_fdf"} <= labels
    # every reuse names the byte-level check that justified it
    assert all(item["evidence"] for item in inventory["reuse"])

    unverified = {item["label"] for item in inventory["unverified"]}
    assert {"tbg_spectral_loss_checkpoint", "tbg_orb_indx", "tbg_moire_geometry"} <= unverified
    assert not (unverified & labels)

    missing = {item["label"] for item in inventory["missing"]}
    assert {"tbg_eigenpairs", "force_constants_search", "dhsdr_search"} <= missing

    kinds = inventory["summary_by_kind"]
    assert set(kinds) >= {
        "graph2mat_checkpoint", "geometry", "hamiltonian_H", "overlap_S",
        "orb_indx", "force_constants", "dhsdr", "eigenpairs", "derivative_stencil",
    }
    assert sum(inventory["summary"].values()) == len(inventory["artifacts"])


def test_inventory_is_idempotent_and_launches_no_calculation(tmp_path, monkeypatch):
    import run_inventory

    campaign = _campaign(tmp_path)
    real_run = subprocess.run

    def only_git(argv, *args, **kwargs):
        # No SIESTA, no Graph2Mat, no training: the inventory reads and hashes.
        assert argv[0] == "git", f"inventory launched {argv[0]!r}"
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(run_inventory.subprocess, "run", only_git)
    first = collect_epc_inventory(repo_root=tmp_path, campaign_root=campaign, include_run_inventory=False)
    second = collect_epc_inventory(repo_root=tmp_path, campaign_root=campaign, include_run_inventory=False)
    assert first["inventory_sha256"] == second["inventory_sha256"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_inventory_publishes_the_epc_artifact_dag_contract(tmp_path):
    """S3: preflight and UI read what invalidates what from the inventory."""
    inventory = collect_epc_inventory(
        repo_root=tmp_path, campaign_root=_campaign(tmp_path), include_run_inventory=False
    )
    kinds = inventory["artifact_dag"]["kinds"]
    assert kinds["geometry"]["depends_on"] == {}
    assert kinds["electronic_hamiltonian"]["depends_on"]["geometry"] == ["geometry"]
    assert kinds["pao_projected_mode_coupling"]["depends_on"]["phonon"] == ["physical_phonon"]
    assert kinds["pao_projected_mode_coupling"]["forbids_synthetic_ancestry"] is True
    # broadening is declared on the observable and nowhere upstream
    assert "broadening" in kinds["integrated_observable"]["signature_fields"]
    assert not any(
        "broadening" in spec["signature_fields"]
        for kind, spec in kinds.items()
        if kind != "integrated_observable"
    )
