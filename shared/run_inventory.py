"""Reproducible run inventory: which code (repo SHAs + imported checkouts) ran.

Single source of truth for the three-repo provenance block required by every
scientific manifest (training, derivatives, mixing, metrics, UI payloads).

``reproducibility_status`` semantics:

- ``pinned_clean``: every repository has a resolvable SHA and a clean tree.
- ``pinned_dirty``: SHAs resolve but at least one tree has local changes.
- ``unpinned``: at least one repository path exists but has no resolvable SHA.
- ``unavailable``: at least one repository could not be inspected at all.

Only ``pinned_clean`` may aspire to ``paper_ready``; everything else is
diagnostic and must surface a visible warning downstream.

SIESTA runtime identity (C01 / E-F_001-S1)
------------------------------------------
``collect_siesta_runtime`` freezes the identity of the SIESTA executable that
later produces H, S, forces and ``dHSdR.nc``: content-addressed hashes of the
binary and of its ``ldd`` linkage closure, the exact ``--version`` probe bytes,
and a provenance classification.

WHAT THIS RECORD CANNOT ESTABLISH, even when every step succeeds:

- that the binary computes what its source says it computes;
- that the compiler translated that source faithfully;
- that the units, sign, layout or sparsification of ``dHSdR.nc`` are the
  expected ones (those are ``source_candidate_expectations``, never observations);
- that results from another machine are comparable at ulp level;
- that historical artifacts were produced by this same file.

It validates no symmetry, no convergence and no physical observable. It
establishes identity and a provenance chain, nothing else. GO-2 remains the
mandatory physical gate.

EPC reuse inventory (C02 / E-F_001-S2)
--------------------------------------
``collect_epc_inventory`` classifies every artifact an EPC campaign could reuse
— checkpoints, geometries, H, S, ORB_INDX, FC, ``dHSdR.nc``, eigenpairs and
derivative stencils — as ``PRESENT_VALID``, ``PRESENT_UNVERIFIED``, ``MISSING``
or ``INVALID``, and records the evidence behind each verdict.

The presence of a path in a script is never evidence. ``PRESENT_VALID``
requires a producing manifest (or the git index) to have pinned the content and
a check that recompares those bytes now; corroborating checks can invalidate an
artifact but never promote it. Nothing here runs SIESTA, Graph2Mat or training.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from artifact_signature import (
    EPC_ARTIFACT_KINDS,
    EPC_ARTIFACT_SIGNATURE_SCHEMA,
)
from benchmark_manifest import (
    canonical_sha256,
    extract_siesta_version_from_text,
    file_sha256,
    looks_like_siesta_version,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORIES = {
    "MD_vs_AtomicDisplacement": REPO_ROOT,
    "graph2mat": REPO_ROOT.parent / "graph2mat",
    "DeepH-pack": REPO_ROOT.parent / "DeepH-pack",
}

RUN_INVENTORY_SCHEMA = "run_inventory_v1"


def _git_output(path: Path, args: list[str], timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_repository_state(path: str | Path | None) -> dict[str, Any]:
    """Commit/branch/dirty for one repository path (never raises)."""
    if path in (None, ""):
        return {"path": None, "commit": None, "branch": None, "dirty": None, "error": "no_path"}
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return {
            "path": str(candidate),
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": "path_missing",
        }
    root_text = _git_output(candidate, ["rev-parse", "--show-toplevel"])
    if not root_text:
        return {
            "path": str(candidate),
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": "not_git_repo",
        }
    root = Path(root_text)
    commit = _git_output(root, ["rev-parse", "HEAD"])
    dirty_text = _git_output(root, ["status", "--porcelain"], timeout=30.0)
    return {
        "path": str(root),
        "commit": commit,
        "branch": _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(dirty_text) if dirty_text is not None else None,
    }


def module_import_state(module_name: str, python_executable: str | Path | None = None) -> dict[str, Any]:
    """Where ``module_name`` actually imports from (this or another python)."""
    if python_executable in (None, "", str(sys.executable)):
        import importlib

        try:
            module = importlib.import_module(module_name)
            module_path = str(Path(getattr(module, "__file__", "") or "").resolve())
            return {"module": module_name, "module_path": module_path or None}
        except Exception as exc:  # noqa: BLE001 - inventory must never crash the run
            return {"module": module_name, "module_path": None, "error": repr(exc)}
    script = (
        f"import importlib, pathlib; "
        f"m = importlib.import_module({module_name!r}); "
        f"print(pathlib.Path(m.__file__).resolve())"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {"module": module_name, "module_path": None, "error": repr(exc)}
    if completed.returncode != 0:
        return {
            "module": module_name,
            "module_path": None,
            "error": completed.stderr.strip() or "import_failed",
        }
    return {"module": module_name, "module_path": completed.stdout.strip()}


def _import_matches_repo(import_state: dict[str, Any], repo_state: dict[str, Any]) -> bool | None:
    module_path = import_state.get("module_path")
    repo_path = repo_state.get("path")
    if not module_path or not repo_path:
        return None
    return str(module_path).startswith(str(repo_path).rstrip("/") + "/")


def reproducibility_status(repositories: dict[str, dict[str, Any]]) -> str:
    statuses = list(repositories.values())
    if not statuses:
        return "unavailable"
    if any(s.get("error") in ("path_missing", "no_path") for s in statuses):
        return "unavailable"
    if any(not s.get("commit") for s in statuses):
        return "unpinned"
    if any(s.get("dirty") for s in statuses):
        return "pinned_dirty"
    if any(s.get("dirty") is None for s in statuses):
        return "unpinned"
    return "pinned_clean"


def collect_run_inventory(
    repositories: dict[str, str | Path] | None = None,
    *,
    deeph_python: str | Path | None = None,
    graph2mat_python: str | Path | None = None,
) -> dict[str, Any]:
    """Full run inventory (repos + python + real import locations).

    ``deeph_python`` / ``graph2mat_python`` point at the interpreters that
    actually run each backend when they differ from ``sys.executable``.
    """
    repo_paths = {k: Path(v) for k, v in (repositories or DEFAULT_REPOSITORIES).items()}
    repo_states = {name: git_repository_state(path) for name, path in repo_paths.items()}

    try:
        import torch

        torch_version = torch.__version__
        default_dtype = str(torch.get_default_dtype()).replace("torch.", "")
    except Exception:  # noqa: BLE001 - torch-less callers still get an inventory
        torch_version = None
        default_dtype = None

    imports: dict[str, Any] = {}
    if "graph2mat" in repo_states:
        state = module_import_state("graph2mat", graph2mat_python)
        state["matches_inspected_repo"] = _import_matches_repo(state, repo_states["graph2mat"])
        imports["graph2mat"] = state
    if "DeepH-pack" in repo_states:
        state = module_import_state("deeph", deeph_python)
        state["matches_inspected_repo"] = _import_matches_repo(state, repo_states["DeepH-pack"])
        imports["deeph"] = state

    inventory = {
        "schema": RUN_INVENTORY_SCHEMA,
        "repositories": repo_states,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "torch_version": torch_version,
            "default_dtype": default_dtype,
        },
        "imports": imports,
        "reproducibility_status": reproducibility_status(repo_states),
    }
    mismatches = [
        name for name, state in imports.items() if state.get("matches_inspected_repo") is False
    ]
    if mismatches:
        inventory["warnings"] = [
            f"module '{name}' imports from outside the inspected repository "
            f"({imports[name].get('module_path')}); the inspected SHA does not "
            "describe the executed code"
            for name in mismatches
        ]
    return inventory


# ---------------------------------------------------------------------------
# SIESTA runtime identity (C01 / E-F_001-S1)
# ---------------------------------------------------------------------------

SIESTA_RUNTIME_SCHEMA = "siesta_runtime_provenance_v1"
SIESTA_RUNTIME_REF_SCHEMA = "siesta_runtime_ref_v1"
SIESTA_RUNTIME_DIR = Path("Comparison/results/provenance/siesta")

# Fixed by the module, never a caller-supplied parameter: an inherited
# environment is exactly why the 2026-07-28 probe carried X11 noise on stderr.
PROBE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")
PROBE_TIMEOUT = 30.0
# The only authorised launch of the binary. `-v`/`-V` are NOT probed: their
# semantics are not guaranteed and they could start a calculation.
VERSION_ARGV = ("--version",)

COMPUTE_POLICY = {
    "requested_backend": "not_applicable",
    "effective_backend": "not_applicable",
    "fallback": False,
    "reason": "C01 only hashes and probes a binary; no numerical workload exists",
}

GO2_FOLLOWUP = (
    "GO2_EXHAUSTIVE_FD_DH_DS",
    "GO2_VERIFY_DH_DS_UNITS_NUMERICALLY",
    "GO2_TEST_DHDR_DSDR_SPARSIFICATION",
    "GO2_DHS_BEHAVIOURAL_FINGERPRINT",
    "GO2_HOLD_MPI_SCF_AND_PHYSICS_INPUTS_FIXED",
    "GO2_NO_SINGLE_PRECISION_HSX_AS_SOLE_ELEMENTWISE_FD_REFERENCE",
)
DO_NOT_REUSE = "DO_NOT_REUSE_FOR_SCIENTIFIC_REFERENCE"

# Claims about dHSdR.nc that C06/C09 must still verify numerically. They are
# read from the 5.4.2 source, never observed on the local executable.
SOURCE_CANDIDATE_EXPECTATIONS = tuple(
    {"id": key, "expectation": text, "observed_on_binary": False}
    for key, text in (
        (
            "DHDR_DSDR_UNITS",
            "dH/dR in Ry/Bohr and dS/dR in 1/Bohr, despite the NetCDF attribute "
            'unit="Ry" on dH and no attribute at all on dS; the file metadata is '
            "incorrect/incomplete and must not be used to infer units",
        ),
        (
            "DHSDR_LAYOUT",
            "layout /DISPLACEMENTS/<atom>/<axis>/ with axis in {1,2,3}; supercell "
            "decoding j_o=((j-1) mod no_u)+1, i_s=floor((j-1)/no_u)+1, image 1 = origin cell",
        ),
        (
            "DHDR_SIGN_POST_5CFDA23A",
            "sign convention after upstream fix 5cfda23a (5.4.0 had H_p/H_m inverted)",
        ),
        (
            "FC_DHDR_TOLERANCE_IS_NOOP",
            "FC.dHdR.Tolerance is apparently a no-op (discard=.false. accumulated with "
            ".and.), so dH comes out dense; the C09 disk/RAM preflight cannot rely on "
            "that threshold",
        ),
        (
            "DSDR_ORBITAL_FILTER",
            "dS carries an additional filter restricted to orbitals of the displaced atom",
        ),
        (
            "DHSDR_FILENAME",
            "effective output name is SystemLabel.dHSdR.nc (dot, not hyphen)",
        ),
    )
)

# ASCII strings whose presence hints at (but never proves) dHSdR support.
FEATURE_SCAN_TOKENS = (
    "dHSdR.nc",
    "DISPLACEMENTS",
    "isc_off",
    "nsc",
    "FC.Save.dHS",
    "FC.dHdR.Tolerance",
    "FC.dSdR.Tolerance",
)

# Empty on purpose: this repository has no attestation verifier, so no
# VERIFIED_SOURCE promotion is reachable today.
TRUSTED_ATTESTATION_VERIFIERS: tuple[str, ...] = ()

GITLAB_API = "https://gitlab.com/api/v4/projects/siesta-project%2Fsiesta/repository"
# Claims to test live against upstream; nothing here is a pre-baked PASS.
UPSTREAM_EXPECTATIONS = {
    "5.4.2-11-g4e9a46060": {
        "commit": "4e9a460606aa14320e56cb0edd88e19d717a46b2",
        "tag_commit": "e486d12067b96ff688179f0496d0ec21b6fae0ab",
        "ancestor_commit": "5cfda23a8d3408e2138464940ef5f3b16cf50fc5",
        "dhsdr_source_path": "Src/dhsdr_m.F90",
        "dhsdr_blob": "e330a324ec3a7f2bcff2c5da0a2805809d73c094",
    },
}

_DESCRIBE = re.compile(r"^(?P<tag>\d+\.\d+(?:\.\d+)?)-(?P<distance>\d+)-g(?P<abbrev>[0-9a-f]{7,40})$")
_VERSION_LINE = re.compile(r"^[ \t]*Version[ \t]*:[ \t]*(\S+)", re.MULTILINE)
_KEY_VALUE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9 _/+.()-]*?)[ \t]*:[ \t]*(.*?)[ \t]*$")
_LDD_ENTRY = re.compile(r"^\s*(?P<soname>\S+)\s+=>\s+(?P<path>/\S+)")
# ldd prints ASLR load addresses; they change every run and would destroy the
# content address, so they are stripped and the normalisation is declared.
_LDD_ADDRESS = re.compile(r"[ \t]*\(0x[0-9a-f]+\)")
_ASCII_RUN = re.compile(rb"[ -~]{4,}")


def _stat_block(path: Path, *, follow: bool) -> dict[str, Any]:
    try:
        info = path.stat() if follow else path.lstat()
    except OSError as exc:
        return {"error": exc.strerror or repr(exc)}
    if stat.S_ISLNK(info.st_mode):
        kind = "symlink"
    elif stat.S_ISREG(info.st_mode):
        kind = "regular"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    else:
        kind = "other"
    return {
        "type": kind,
        "mode": stat.filemode(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "size": info.st_size,
        "inode": info.st_ino,
        "nlink": info.st_nlink,
    }


def resolve_executable_chain(path: str | Path, *, max_hops: int = 40) -> dict[str, Any]:
    """Full symlink chain (relative hops included) with cycle detection."""
    current = Path(path).expanduser().absolute()
    chain: list[str] = []
    seen: set[str] = set()
    cycle = False
    for _ in range(max_hops):
        key = str(current)
        if key in seen:
            cycle = True
            break
        seen.add(key)
        chain.append(key)
        if not current.is_symlink():
            break
        target = Path(os.readlink(current))
        if not target.is_absolute():
            target = current.parent / target
        current = Path(os.path.normpath(target))
    else:
        cycle = True
    return {
        "requested": str(path),
        "chain": chain,
        "hops": max(len(chain) - 1, 0),
        "symlink_cycle": cycle,
        "final_target": None if cycle else str(current),
    }


def _probe_environment() -> dict[str, str]:
    env = {key: os.environ[key] for key in PROBE_ENV_KEYS if key in os.environ}
    env["LC_ALL"] = "C"
    return env


def _recorded_environment(env: dict[str, str]) -> dict[str, Any]:
    """The complete allowlisted environment used by the probe."""
    return {
        "values": {key: env[key] for key in PROBE_ENV_KEYS if key in env},
        "disclosure": "values for PROBE_ENV_KEYS only",
    }


def _capture(argv: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    """Run ``argv`` in a throwaway cwd, keeping stdout/stderr as exact bytes."""
    record: dict[str, Any] = {
        "argv": list(argv),
        # The literal temporary path is volatile; what matters (and what is
        # recorded) is that the probe never ran inside the repository tree.
        "cwd": "ephemeral_temporary_directory",
        "cwd_inside_repository": False,
        "env": _recorded_environment(env),
        "timeout_s": PROBE_TIMEOUT,
        "timed_out": False,
        "returncode": None,
        "signal": None,
        "error": None,
    }
    stdout = stderr = b""
    try:
        with tempfile.TemporaryDirectory(prefix="siesta_probe_") as workdir:
            completed = subprocess.run(
                list(argv),
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=PROBE_TIMEOUT,
                cwd=workdir,
                env=env,
                check=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
            record["returncode"] = completed.returncode
            if completed.returncode < 0:
                record["signal"] = -completed.returncode
    except subprocess.TimeoutExpired as exc:
        record["timed_out"] = True
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
        record["error"] = "timeout"
    except Exception as exc:  # noqa: BLE001 - a failed probe is evidence, not a crash
        record["error"] = repr(exc)
    record["stdout_b64"] = base64.b64encode(stdout).decode("ascii")
    record["stderr_b64"] = base64.b64encode(stderr).decode("ascii")
    record["stdout_text"] = stdout.decode("utf-8", errors="replace")
    record["stderr_text"] = stderr.decode("utf-8", errors="replace")
    return record


def parse_build_info(text: str) -> dict[str, str]:
    """``key: value`` lines of the SIESTA banner, with no closed key list."""
    info: dict[str, str] = {}
    for line in text.splitlines():
        match = _KEY_VALUE.match(line)
        if match:
            info.setdefault(match.group(1).strip(), match.group(2).strip())
    return info


def evaluate_strict_probe(capture: dict[str, Any]) -> dict[str, Any]:
    """A valid strict probe: rc 0, a ``Version :`` line, and real build info."""
    stdout = str(capture.get("stdout_text") or "")
    stderr = str(capture.get("stderr_text") or "")
    banner = _VERSION_LINE.search(stdout)
    build_info = parse_build_info(stdout)
    versions_seen = sorted({match for match in _VERSION_LINE.findall(stdout + "\n" + stderr)})
    reasons: list[str] = []
    if capture.get("returncode") != 0:
        reasons.append("nonzero_returncode")
    if banner is None or not looks_like_siesta_version(banner.group(1)):
        reasons.append("no_valid_version_line")
    if len([key for key in build_info if key != "Version"]) < 2:
        reasons.append("build_information_block_missing")
    version = extract_siesta_version_from_text(stdout)
    if banner is not None and version != banner.group(1):
        reasons.append("version_extraction_disagrees_with_banner")
    if len(versions_seen) > 1:
        reasons.append("incompatible_versions_in_probe")
    return {
        "valid": not reasons,
        "version": banner.group(1) if banner and not reasons else None,
        "build_info": build_info,
        "versions_seen": versions_seen,
        "reasons": sorted(reasons),
    }


def probe_siesta_version(resolved: Path) -> dict[str, Any]:
    """``<binary> --version`` under a fixed minimal environment."""
    argv = [str(resolved), *VERSION_ARGV]
    attempts = [_capture(argv, env=_probe_environment())]
    return {
        "env_mode": "minimal",
        "env_keys": list(PROBE_ENV_KEYS),
        "attempts": attempts,
        "strict": evaluate_strict_probe(attempts[-1]),
    }


def _hash_target(path: Path, cache: dict[str, str | None]) -> str | None:
    key = str(path)
    if key not in cache:
        try:
            cache[key] = file_sha256(path) if path.is_file() else None
        except OSError:
            cache[key] = None
    return cache[key]


def collect_linkage(resolved: Path, cache: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Hash every ``ldd`` dependency through its symlink chain.

    The binary SHA256 does not pin the arithmetic: on Debian/Ubuntu
    ``libblas.so.3``/``liblapack.so.3`` are update-alternatives symlinks, so the
    implementation (reference BLAS / OpenBLAS / MKL) — and with it the reduction
    order of every sum — can change without touching a single byte of SIESTA.
    """
    cache = {} if cache is None else cache
    result: dict[str, Any] = {
        "linkage_closure": "ldd_static_only",
        "closure_caveat": "dlopen, plugins and dispatch environment variables (e.g. OpenBLAS "
        "kernel selection) are outside this closure",
        "ldd_normalization": "aslr_load_addresses_stripped",
        "libraries": [],
    }
    try:
        # Same minimal environment as the probe: LC_ALL=C keeps the output
        # locale-independent and LD_LIBRARY_PATH must match what SIESTA loads.
        completed = subprocess.run(
            ["ldd", str(resolved)],
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            env=_probe_environment(),
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - best effort, never a gate
        result["ldd"] = {"returncode": None, "error": repr(exc)}
        result["unavailable_reason"] = "ldd_failed"
        return result
    stdout = _LDD_ADDRESS.sub("", completed.stdout)
    result["ldd"] = {
        "returncode": completed.returncode,
        "stdout_normalized": stdout,
        "stderr_normalized": _LDD_ADDRESS.sub("", completed.stderr),
    }
    if completed.returncode != 0:
        result["unavailable_reason"] = f"ldd_returncode_{completed.returncode}"
        return result
    libraries = []
    for line in stdout.splitlines():
        match = _LDD_ENTRY.match(line)
        if not match:
            continue
        chain = resolve_executable_chain(match.group("path"))
        final = Path(chain["final_target"]) if chain["final_target"] else None
        libraries.append(
            {
                "soname": match.group("soname"),
                "path": match.group("path"),
                "resolved_path": chain["final_target"],
                "symlink_chain": chain["chain"],
                "sha256": _hash_target(final, cache) if final else None,
            }
        )
    result["libraries"] = sorted(libraries, key=lambda row: (row["soname"], row["path"]))
    return result


def _diagnostic_command(argv: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            env=_probe_environment(),  # LC_ALL=C: localised output is not reproducible
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics never gate
        return {"argv": argv, "returncode": None, "error": repr(exc)}
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def feature_surface_scan(resolved: Path) -> dict[str, Any]:
    """Count dHSdR-related ASCII strings. Diagnostic only, never a gate."""
    scan: dict[str, Any] = {
        "interpretation": "presence does not prove behaviour; absence (stripping, "
        "optimisation) invalidates nothing",
        "tokens": [],
    }
    try:
        blob = resolved.read_bytes()
    except OSError as exc:
        scan["error"] = exc.strerror or repr(exc)
        return scan
    strings = b"\n".join(_ASCII_RUN.findall(blob)).decode("ascii", errors="ignore")
    scan["tokens"] = [
        {"token": token, "occurrences": strings.count(token)} for token in FEATURE_SCAN_TOKENS
    ]
    return scan


def _walk_key(payload: Any, key: str) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == key and isinstance(value, str) and value.strip():
                found.append(value.strip())
            else:
                found.extend(_walk_key(value, key))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_walk_key(item, key))
    return found


def config_siesta_commands(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    """``siesta_command`` declarations in ``Comparison/config/*.json``."""
    rows: list[dict[str, str]] = []
    for path in sorted((repo_root / "Comparison" / "config").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken config is not this ticket's problem
            continue
        for command in sorted(set(_walk_key(payload, "siesta_command"))):
            rows.append({"config": str(path.relative_to(repo_root)), "siesta_command": command})
    return rows


def default_siesta_runtime(repo_root: Path = REPO_ROOT) -> str | None:
    """The binary this repository actually launches: PATH first, then configs."""
    found = shutil.which("siesta")
    if found:
        return found
    declarations = config_siesta_commands(repo_root)
    return declarations[0]["siesta_command"] if declarations else None


def invocation_audit(
    requested: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    path_env: str | None = None,
    cache: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """The two launch forms this repository uses must be the same bytes.

    No CLI option declares arbitrary equivalences: there is no third launch form
    in this repository, and the observed evidence says the two agree.
    """
    cache = {} if cache is None else cache
    declarations = config_siesta_commands(repo_root)
    which_path = shutil.which("siesta", path=path_env) if path_env else shutil.which("siesta")
    origins = [("requested", str(requested))]
    origins += [("config", row["siesta_command"]) for row in declarations]
    if which_path:
        origins.append(("which", which_path))

    resolutions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for origin, command in origins:
        if (origin, command) in seen:
            continue
        seen.add((origin, command))
        chain = resolve_executable_chain(command)
        final = Path(chain["final_target"]) if chain["final_target"] else None
        resolutions.append(
            {
                "origin": origin,
                "command": command,
                "resolved_path": chain["final_target"],
                "sha256": _hash_target(final, cache) if final else None,
            }
        )
    distinct = sorted({row["sha256"] for row in resolutions if row["sha256"]})
    notes: list[str] = []
    if not which_path:
        notes.append("which_unresolved_siesta_not_on_probe_path")
    if not declarations:
        notes.append("no_config_siesta_command_declared")
    return {
        "resolutions": resolutions,
        "config_declarations": declarations,
        "which": {"token": "siesta", "path_env": path_env, "resolved": which_path},
        "distinct_sha256": distinct,
        "agree": len(distinct) == 1,
        "notes": notes,
    }


def _gitlab_get(endpoint: str, timeout: float = 20.0) -> tuple[Any, str | None]:
    try:
        with urllib.request.urlopen(GITLAB_API + endpoint, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except Exception as exc:  # noqa: BLE001 - network failure is "unavailable", not a verdict
        return None, repr(exc)


def _check(name: str, endpoint: str, expected: Any, observed: Any, error: str | None = None) -> dict[str, Any]:
    if error is not None or observed is None:
        status = "unavailable"
    elif observed == expected:
        status = "match"
    else:
        status = "contradicted"
    return {
        "name": name,
        "endpoint": endpoint,
        "expected": expected,
        "observed": observed,
        "status": status,
        "error": error,
        "response_sha256": canonical_sha256({"endpoint": endpoint, "observed": observed}),
    }


def verify_upstream_evidence(version_token: str | None, *, fetch=_gitlab_get) -> dict[str, Any]:
    """Test the source-candidate claims against upstream SIESTA (read-only)."""
    describe = _DESCRIBE.match(version_token or "")
    expectations = UPSTREAM_EXPECTATIONS.get(version_token or "")
    if not describe or not expectations:
        return {
            "requested": True,
            "status": "not_applicable_for_version_token",
            "version_token": version_token,
            "checks": [],
        }
    tag = describe.group("tag")
    distance = int(describe.group("distance"))
    abbrev = describe.group("abbrev")
    commit = expectations["commit"]

    checks = [
        _check("describe_abbrev_matches_expected_commit", "offline", abbrev, commit[: len(abbrev)])
    ]

    endpoint = f"/commits/{commit}"
    payload, error = fetch(endpoint)
    checks.append(
        _check(
            "commit_exists_upstream",
            endpoint,
            commit,
            (payload or {}).get("id") if isinstance(payload, dict) else None,
            error,
        )
    )

    endpoint = f"/tags/{urllib.parse.quote(tag, safe='')}"
    payload, error = fetch(endpoint)
    observed = (payload or {}).get("commit", {}).get("id") if isinstance(payload, dict) else None
    checks.append(_check("tag_points_at_expected_commit", endpoint, expectations["tag_commit"], observed, error))

    endpoint = f"/commits?ref_name={commit}&first_parent=true&per_page=100"
    payload, error = fetch(endpoint)
    observed = None
    if isinstance(payload, list):
        ids = [item.get("id") for item in payload if isinstance(item, dict)]
        # SIESTA versions come from `git describe --first-parent`; the full DAG
        # distance is 22, the first-parent distance is 11.
        observed = ids.index(expectations["tag_commit"]) if expectations["tag_commit"] in ids else None
    checks.append(_check("first_parent_distance_from_tag", endpoint, distance, observed, error))

    ancestor = expectations["ancestor_commit"]
    endpoint = (
        "/merge_base?"
        + urllib.parse.urlencode({"refs[]": ancestor}, safe="[]")
        + "&"
        + urllib.parse.urlencode({"refs[]": commit}, safe="[]")
    )
    payload, error = fetch(endpoint)
    observed = (payload or {}).get("id") if isinstance(payload, dict) else None
    checks.append(_check("derivative_sign_fix_is_ancestor", endpoint, ancestor, observed, error))

    source_path = urllib.parse.quote(expectations["dhsdr_source_path"], safe="")
    endpoint = f"/files/{source_path}?ref={commit}"
    payload, error = fetch(endpoint)
    observed = (payload or {}).get("blob_id") if isinstance(payload, dict) else None
    checks.append(_check("dhsdr_source_blob", endpoint, expectations["dhsdr_blob"], observed, error))

    statuses = {check["status"] for check in checks}
    if "contradicted" in statuses:
        status = "contradicted"
    elif "unavailable" in statuses:
        status = "unavailable"
    else:
        status = "consistent"
    return {
        "requested": True,
        "status": status,
        "version_token": version_token,
        "describe": {"tag": tag, "distance": distance, "abbrev": abbrev},
        "checks": checks,
    }


def _classify(
    binary: dict[str, Any],
    probe: dict[str, Any],
    audit: dict[str, Any],
    upstream: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if binary.get("symlink_cycle") or not binary.get("final_target"):
        reasons.append("binary_target_unresolved")
    else:
        if binary.get("type") != "regular":
            reasons.append("binary_target_not_regular_file")
        if not binary.get("readable"):
            reasons.append("binary_target_not_readable")
        if not binary.get("executable"):
            reasons.append("binary_target_not_executable")
        if binary.get("hashed") and not binary.get("sha256_stable"):
            reasons.append("binary_changed_during_collection")
    strict = probe.get("strict", {})
    if not strict.get("valid"):
        reasons.append("strict_version_probe_failed")
    if "incompatible_versions_in_probe" in strict.get("reasons", []):
        reasons.append("incompatible_versions_in_probe")
    if not audit.get("agree"):
        reasons.append("invocation_paths_disagree")
    if upstream.get("status") == "contradicted":
        reasons.append("upstream_evidence_contradicted")

    if reasons:
        return {
            "status": "UNVERIFIED",
            "reasons": sorted(set(reasons)),
            "reusable_as_scientific_reference": False,
            "ceiling_reason": "diagnostic record only",
        }
    return {
        "status": "VERIFIED_BINARY_ONLY",
        "reasons": ["binary_identity_established_source_link_absent"],
        "reusable_as_scientific_reference": True,
        "ceiling_reason": "no trusted attestation verifier and no reproducible rebuild; the "
        "embedded version string is not evidence of the source tree "
        "(Tools/version_generate.sh runs git describe without --dirty, uses git diff-index "
        "which ignores untracked files, and only runs at configure time)",
    }


def collect_siesta_runtime(
    siesta_runtime: str | Path,
    *,
    verify_upstream: bool = False,
    repo_root: Path = REPO_ROOT,
    attestation: Any = None,
    fetch=_gitlab_get,
) -> dict[str, Any]:
    """Immutable, content-addressed identity of one SIESTA executable."""
    hash_cache: dict[str, str | None] = {}
    chain = resolve_executable_chain(siesta_runtime)
    final = Path(chain["final_target"]) if chain["final_target"] else None

    binary: dict[str, Any] = {
        "requested": str(siesta_runtime),
        "requested_absolute": str(Path(siesta_runtime).expanduser().absolute()),
        "symlink_chain": chain["chain"],
        "symlink_hops": chain["hops"],
        "symlink_cycle": chain["symlink_cycle"],
        "final_target": chain["final_target"],
        "lstat": _stat_block(Path(chain["chain"][0]), follow=False) if chain["chain"] else {},
        "stat": _stat_block(final, follow=True) if final else {},
    }
    binary["type"] = binary["stat"].get("type")
    binary["readable"] = bool(final and os.access(final, os.R_OK))
    binary["executable"] = bool(final and os.access(final, os.X_OK))
    usable = bool(final and final.is_file() and binary["readable"] and binary["executable"])
    binary["hashed"] = usable

    binary["sha256_before_probes"] = _hash_target(final, {}) if usable else None
    if usable:
        probe = probe_siesta_version(final)
        linkage = collect_linkage(final, hash_cache)
        binary["diagnostics"] = {
            "file": _diagnostic_command(["file", str(final)]),
            "readelf_notes": _diagnostic_command(["readelf", "-n", str(final)]),
            "feature_surface_scan": feature_surface_scan(final),
        }
    else:
        probe = {
            "env_mode": "not_attempted",
            "env_keys": list(PROBE_ENV_KEYS),
            "attempts": [],
            "strict": {"valid": False, "version": None, "build_info": {}, "versions_seen": [],
                       "reasons": ["binary_not_usable"]},
        }
        linkage = {"linkage_closure": "ldd_static_only", "libraries": [],
                   "unavailable_reason": "binary_not_usable"}
        binary["diagnostics"] = {}
    binary["sha256_after_probes"] = _hash_target(final, {}) if usable else None
    binary["sha256_stable"] = (
        binary["sha256_before_probes"] is not None
        and binary["sha256_before_probes"] == binary["sha256_after_probes"]
    )
    binary["sha256"] = binary["sha256_before_probes"] if binary["sha256_stable"] else None
    if binary["sha256"]:
        hash_cache[str(final)] = binary["sha256"]

    strict = probe["strict"]
    version_token = strict.get("version")
    audit = invocation_audit(
        siesta_runtime,
        repo_root=repo_root,
        path_env=_probe_environment().get("PATH"),
        cache=hash_cache,
    )
    upstream = (
        verify_upstream_evidence(version_token, fetch=fetch)
        if verify_upstream
        else {"requested": False, "status": "not_requested", "checks": []}
    )

    describe = _DESCRIBE.match(version_token or "")
    source_candidate = {
        "version_token": version_token,
        "describe": describe.groupdict() if describe else None,
        "build_info": strict.get("build_info", {}),
        "evidence_class": "embedded_version_string_only",
        "why_not_source_proof": [
            "Tools/version_generate.sh runs `git describe` without --dirty and appends the "
            "suffix from `git diff-index`, which ignores untracked files",
            "CMakeLists.txt regenerates the version at configure time only; `cmake --build` "
            "does not refresh it",
            "a clean checkout, CMakeCache.txt, blob matches or ELF strings do not bind binary "
            "to source either",
        ],
        "expectations": [dict(item) for item in SOURCE_CANDIDATE_EXPECTATIONS],
    }
    source_link = {
        "attestation": {
            "supplied": attestation,
            "trusted_verifier_available": bool(TRUSTED_ATTESTATION_VERIFIERS),
            "trusted": False,
            "reason": "no trusted attestation verifier exists in this repository",
        },
        "reproducible_rebuild": {
            "attempted": False,
            "reason": "built with -O3 -march=native; a bit-identical rebuild is not a realistic "
            "target and C01 does not recompile SIESTA",
        },
        "upstream": upstream,
        "verdict": "contradicted" if upstream.get("status") == "contradicted" else "unlinked",
    }

    classification = _classify(binary, probe, audit, upstream)
    followup = list(GO2_FOLLOWUP)
    if classification["status"] == "UNVERIFIED":
        followup = [DO_NOT_REUSE, *followup]

    payload: dict[str, Any] = {
        "schema": SIESTA_RUNTIME_SCHEMA,
        "binary": binary,
        "libraries": linkage,
        "probe": probe,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "invocation_audit": audit,
        "source_candidate": source_candidate,
        "source_link": source_link,
        "classification": classification,
        "required_followup": followup,
        "compute_policy": dict(COMPUTE_POLICY),
    }
    payload["record_sha256"] = canonical_sha256(payload)
    return payload


def canonical_record_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def validate_siesta_runtime_record(payload: Any) -> list[str]:
    """Structural + hash reasons why ``payload`` is not a usable record."""
    if not isinstance(payload, dict):
        return ["record_not_a_mapping"]
    reasons: list[str] = []
    if payload.get("schema") != SIESTA_RUNTIME_SCHEMA:
        reasons.append("unknown_record_schema")
    missing = [
        block
        for block in (
            "binary", "libraries", "probe", "platform", "invocation_audit",
            "source_candidate", "source_link", "classification", "required_followup",
            "compute_policy", "record_sha256",
        )
        if block not in payload
    ]
    if missing:
        reasons.append("record_missing_blocks")
    body = {key: value for key, value in payload.items() if key != "record_sha256"}
    if payload.get("record_sha256") != canonical_sha256(body):
        reasons.append("record_sha256_mismatch")
    status = payload.get("classification", {}).get("status") if isinstance(payload.get("classification"), dict) else None
    if status not in {"VERIFIED_SOURCE", "VERIFIED_BINARY_ONLY", "UNVERIFIED"}:
        reasons.append("unknown_classification_status")
    return sorted(set(reasons))


def write_siesta_runtime_record(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Write ``<record_sha256>.json`` create-if-absent and return its reference."""
    reasons = validate_siesta_runtime_record(payload)
    if reasons:
        raise ValueError(f"refusing to write an invalid record: {reasons}")
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = repo_root / directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['record_sha256']}.json"
    blob = canonical_record_bytes(payload)
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError(f"{path} exists with different content; refusing to overwrite")
    else:
        path.write_bytes(blob)
    try:
        record_path = str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        record_path = str(path.resolve())
    return {
        "schema": SIESTA_RUNTIME_REF_SCHEMA,
        "record_path": record_path,
        "record_sha256": payload["record_sha256"],
        "binary_sha256": payload["binary"].get("sha256"),
        "classification": payload["classification"]["status"],
    }


def validate_siesta_runtime_ref(
    ref: Any,
    *,
    effective_executable: str | Path | None = None,
    effective_sha256: str | None = None,
    require_effective: bool = True,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Check a downstream reference and, if given, the executable that ran.

    Validating the JSON alone never proves which binary ran: in a scientific
    calculation ``effective_executable`` or ``effective_sha256`` is mandatory.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(ref, dict):
        return {"valid": False, "reasons": ["ref_not_a_mapping"], "warnings": [], "record": None}
    if ref.get("schema") != SIESTA_RUNTIME_REF_SCHEMA:
        reasons.append("unknown_ref_schema")

    payload: dict[str, Any] | None = None
    raw = ref.get("record_path")
    path = Path(str(raw)) if raw else None
    if path is None:
        reasons.append("ref_missing_record_path")
    else:
        candidate = (repo_root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError:
            reasons.append("record_path_escapes_repository_root")
            candidate = None
        if candidate is not None:
            if not candidate.is_file():
                reasons.append("record_file_missing")
            else:
                blob = candidate.read_bytes()
                try:
                    payload = json.loads(blob.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    reasons.append("record_not_readable_json")
                    payload = None
                if isinstance(payload, dict):
                    if canonical_record_bytes(payload) != blob:
                        reasons.append("record_not_canonical_bytes")
                    if candidate.stem != payload.get("record_sha256"):
                        reasons.append("record_filename_does_not_match_hash")
                    reasons.extend(validate_siesta_runtime_record(payload))

    if isinstance(payload, dict):
        status = payload.get("classification", {}).get("status")
        if payload.get("record_sha256") != ref.get("record_sha256"):
            reasons.append("ref_record_sha256_mismatch")
        if payload.get("binary", {}).get("sha256") != ref.get("binary_sha256"):
            reasons.append("ref_binary_sha256_mismatch")
        if status != ref.get("classification"):
            reasons.append("ref_classification_mismatch")
        if status == "UNVERIFIED":
            reasons.append("record_is_unverified")

    observed: str | None = None
    if effective_executable is not None:
        chain = resolve_executable_chain(effective_executable)
        target = Path(chain["final_target"]) if chain["final_target"] else None
        observed = _hash_target(target, {}) if target else None
        if observed is None:
            reasons.append("effective_executable_unhashable")
    if effective_sha256 is not None:
        if observed is not None and observed != effective_sha256:
            reasons.append("effective_executable_disagrees_with_supplied_sha256")
        observed = observed or effective_sha256
    if observed is None:
        (reasons if require_effective else warnings).append(
            "effective_executable_not_supplied"
        )
    elif observed != ref.get("binary_sha256"):
        reasons.append("effective_binary_sha256_mismatch")

    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "warnings": warnings,
        "effective_binary_checked": observed is not None,
        "record": payload,
    }


# ---------------------------------------------------------------------------
# EPC reuse inventory (C02 / E-F_001-S2)
# ---------------------------------------------------------------------------
# What exists, what may be reused, and — for every reuse — the byte-level
# evidence that justifies it. Presence of a path in a script is never evidence:
# a producing manifest must declare the artifact, and a check must compare
# bytes, before anything reaches PRESENT_VALID.

EPC_INVENTORY_SCHEMA = "epc_reuse_inventory_v1"

PRESENT_VALID = "PRESENT_VALID"
PRESENT_UNVERIFIED = "PRESENT_UNVERIFIED"
MISSING = "MISSING"
INVALID = "INVALID"
EPC_STATUSES = (PRESENT_VALID, PRESENT_UNVERIFIED, MISSING, INVALID)

EPC_INVENTORY_DIR = Path("Comparison/results/provenance/epc")
TBG_CAMPAIGN_ROOT = Path("Comparison/results/tbg_pure_graph2mat")
# Systems the roadmap actually walks (graphene Γ → K → AB bilayer → MATBG).
EPC_MATERIAL_GEOMETRIES = (
    "graphene",
    "bilayer_graphene_AB",
    "twisted_bilayer_graphene_1p084549deg",
)
# Directories searched for artifacts that no manifest declares. Recorded in the
# record so a MISSING verdict stays auditable instead of being an assumption.
EPC_SEARCH_ROOTS = ("Comparison/results", "Comparison/workspaces", "materials")
FC_PATTERNS = ("**/*.FC",)
DHSDR_PATTERNS = ("**/*dHSdR*",)

EPC_INVENTORY_LIMITATIONS = (
    "PRESENT_VALID means the bytes match what the producer declared; it is not a "
    "statement about physical correctness (GO-1..GO-10 remain open)",
    "everything under Comparison/results/ is gitignored: the only provenance those "
    "artifacts have is the sha256 their producing manifest wrote down",
    "this inventory launches no SIESTA, no Graph2Mat and no training",
)

_OUTCOMES = ("pass", "fail", "absent")
_FDF_MD_DIRECTIVE = re.compile(r"^[ \t]*(MD\.\S+|Lua\.Script)[ \t]+(\S+)", re.IGNORECASE | re.MULTILINE)
_ANALYZE_PHONONS_PATTERNS = re.compile(r"patterns\s*=\s*\(([^)]*)\)")


def _evidence(check: str, outcome: str, detail: str = "", *, verifies_content: bool = False) -> dict[str, Any]:
    if outcome not in _OUTCOMES:
        raise ValueError(f"outcome must be one of {_OUTCOMES}, got {outcome!r}")
    return {
        "check": check,
        "outcome": outcome,
        "detail": str(detail),
        "verifies_content": bool(verifies_content),
    }


# Public names for the producers that must speak this same vocabulary (the C07
# phonon archiver classifies its own copies with these rules).
evidence_item = _evidence


def epc_artifact_status(evidence: list[dict[str, Any]]) -> str:
    """PRESENT_VALID requires a check that actually compared bytes.

    Corroborating evidence (a passing precision gate, agreeing consumers) can
    invalidate an artifact but can never promote it: an artifact whose content
    nobody pinned stays PRESENT_UNVERIFIED, never REUSED.
    """
    if any(item["outcome"] == "fail" for item in evidence):
        return INVALID
    if any(item["outcome"] == "pass" and item["verifies_content"] for item in evidence):
        return PRESENT_VALID
    return PRESENT_UNVERIFIED


def _repo_relative(path: str | Path, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(repo_root).resolve()))
    except (ValueError, OSError):
        return str(path)


def _json_or_none(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify_epc_artifact(
    kind: str,
    path: str | Path | None,
    *,
    path_source: str,
    declared_sha256: str | None = None,
    declared_by: str | None = None,
    evidence: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    label: str | None = None,
    repo_root: Path = REPO_ROOT,
    expect: str = "file",
) -> dict[str, Any]:
    """Classify one artifact as PRESENT_VALID/PRESENT_UNVERIFIED/MISSING/INVALID.

    ``expect="directory"`` is for grouped artifacts (a whole reference set),
    whose content is verified by the supplied per-member evidence rather than
    by one sha256.
    """
    record: dict[str, Any] = {
        "kind": kind,
        "label": label or kind,
        "path": None,
        "path_source": path_source,
        "declared_by": declared_by,
        "declared_sha256": declared_sha256,
        "observed_sha256": None,
    }
    checks = list(evidence)
    if path in (None, ""):
        record["status"] = MISSING
        record["evidence"] = [
            _evidence("path_declared", "absent", "no producer declares a path for this artifact"),
            *checks,
        ]
        return record

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path(repo_root) / target
    record["path"] = _repo_relative(target, repo_root)

    if not target.exists():
        record["status"] = MISSING
        record["evidence"] = [_evidence("path_exists", "absent", record["path"]), *checks]
        return record

    if expect == "directory":
        if not target.is_dir():
            checks.insert(0, _evidence("path_is_directory", "fail", f"{record['path']} is not a directory"))
    elif not target.is_file():
        checks.insert(0, _evidence("path_is_regular_file", "fail", f"{record['path']} is not a regular file"))
    elif declared_sha256:
        observed = file_sha256(target)
        record["observed_sha256"] = observed
        matches = observed == str(declared_sha256)
        checks.insert(
            0,
            _evidence(
                "sha256_matches_declared",
                "pass" if matches else "fail",
                f"declared={declared_sha256} observed={observed} declared_by={declared_by}",
                verifies_content=True,
            ),
        )
    else:
        checks.insert(
            0,
            _evidence("declared_sha256", "absent", f"no manifest declares a sha256 for {record['path']}"),
        )

    record["evidence"] = checks
    record["status"] = epc_artifact_status(checks)
    return record


def _git_pin_evidence(path: Path, repo_root: Path) -> dict[str, Any]:
    """A tracked, unmodified file is pinned by the commit the run inventory records."""
    relative = _repo_relative(path, repo_root)
    if _git_output(repo_root, ["ls-files", "--error-unmatch", "--", relative]) is None:
        return _evidence("content_pinned_by_git_commit", "absent", f"{relative} is not tracked by git")
    dirty = _git_output(repo_root, ["status", "--porcelain", "--", relative], timeout=30.0)
    if dirty:
        return _evidence(
            "content_pinned_by_git_commit", "fail", f"{relative} has uncommitted local changes: {dirty}"
        )
    commit = _git_output(repo_root, ["rev-parse", "HEAD"]) or "unknown"
    return _evidence(
        "content_pinned_by_git_commit", "pass", f"{relative} clean at {commit}", verifies_content=True
    )


def _agreement_evidence(check: str, declared: Any, consumers: dict[str, Any]) -> dict[str, Any]:
    """Consumers that recorded an upstream hash must agree with the producer."""
    stated = {name: value for name, value in consumers.items() if value}
    if not declared or not stated:
        return _evidence(check, "absent", "no consumer recorded an upstream hash")
    disagreeing = {name: value for name, value in stated.items() if str(value) != str(declared)}
    if disagreeing:
        return _evidence(check, "fail", f"declared={declared} consumers={disagreeing}")
    return _evidence(check, "pass", f"{len(stated)} consumer(s) agree on {declared}")


def _declared_checkpoint_sha256(checkpoint: Path) -> tuple[str | None, str | None]:
    """The ``checkpoint_manifest.json`` (if any) that pins this exact file."""
    for ancestor in list(checkpoint.parents)[:5]:
        manifest = ancestor / "checkpoint_manifest.json"
        payload = _json_or_none(manifest)
        if not isinstance(payload, dict) or not payload.get("checkpoint_sha256"):
            continue
        declared = payload.get("checkpoint_path")
        if declared and Path(str(declared)).resolve() == checkpoint.resolve():
            return str(payload["checkpoint_sha256"]), str(manifest)
    return None, None


def epc_checkpoint_records(repo_root: Path, campaign_root: Path) -> list[dict[str, Any]]:
    status = _json_or_none(campaign_root / "status.json") or {}
    gate = _json_or_none(campaign_root / "precision_gate.json") or {}
    prediction = _json_or_none(campaign_root / "prediction" / "raw" / "prediction_summary.json") or {}
    declared = status.get("checkpoint")
    status_source = f"manifest:{_repo_relative(campaign_root / 'status.json', repo_root)}#checkpoint"

    evidence = [
        _agreement_evidence(
            "campaign_stages_name_one_checkpoint",
            declared,
            {"prediction/raw/prediction_summary.json#checkpoint": prediction.get("checkpoint")},
        )
    ]

    gate_status = str(gate.get("status") or "")
    gate_checkpoint = str(gate.get("checkpoint") or "")
    if not gate:
        evidence.append(_evidence("precision_gate_passed", "absent", "no precision_gate.json in the campaign"))
    elif gate_status != "passed":
        evidence.append(_evidence("precision_gate_passed", "fail", f"precision gate status={gate_status!r}"))
    elif declared and gate_checkpoint != Path(str(declared)).name:
        evidence.append(
            _evidence(
                "precision_gate_passed",
                "fail",
                f"gate scored {gate_checkpoint!r}, campaign uses {Path(str(declared)).name!r}",
            )
        )
    else:
        evidence.append(
            _evidence(
                "precision_gate_passed",
                "pass",
                f"{gate_checkpoint} score_eV={gate.get('score_eV')} threshold_eV={gate.get('threshold_eV')}",
            )
        )
    evidence.append(
        _evidence(
            "derivative_fidelity_established",
            "absent",
            "the checkpoint was selected on spectral accuracy; nothing here says anything "
            "about dH/dR (roadmap blocker B6, decided by C14)",
        )
    )

    sha256, declared_by = (
        _declared_checkpoint_sha256(Path(str(declared))) if declared else (None, None)
    )
    return [
        classify_epc_artifact(
            "graph2mat_checkpoint",
            declared,
            path_source=status_source,
            declared_sha256=sha256,
            declared_by=declared_by,
            evidence=evidence,
            label="tbg_spectral_loss_checkpoint",
            repo_root=repo_root,
        )
    ]


def epc_geometry_records(repo_root: Path, campaign_root: Path) -> list[dict[str, Any]]:
    records = []
    for name in EPC_MATERIAL_GEOMETRIES:
        run_fdf = repo_root / "materials" / name / "RUN.fdf"
        records.append(
            classify_epc_artifact(
                "geometry",
                run_fdf,
                path_source="repository_convention:materials/<system>/RUN.fdf",
                evidence=[_git_pin_evidence(run_fdf, repo_root)],
                label=f"materials/{name}",
                repo_root=repo_root,
            )
        )

    overlap_manifest = campaign_root / "overlap" / "overlap_manifest.json"
    overlap = _json_or_none(overlap_manifest) or {}
    source = f"manifest:{_repo_relative(overlap_manifest, repo_root)}"
    records.append(
        classify_epc_artifact(
            "geometry",
            overlap.get("effective_run_fdf"),
            path_source=f"{source}#effective_run_fdf",
            declared_sha256=overlap.get("effective_run_fdf_sha256"),
            declared_by=f"{source}#effective_run_fdf_sha256",
            label="tbg_overlap_effective_run_fdf",
            repo_root=repo_root,
        )
    )
    records.append(
        classify_epc_artifact(
            "geometry",
            campaign_root / "target" / "moire_geometry.json",
            path_source="repository_convention:<campaign>/target/moire_geometry.json",
            label="tbg_moire_geometry",
            repo_root=repo_root,
        )
    )
    return records


def _solver_manifests(campaign_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in sorted(campaign_root.glob("spectra/*/solver_manifest.json")):
        payload = _json_or_none(path)
        if isinstance(payload, dict):
            found.append((path, payload))
    return found


def epc_matrix_records(repo_root: Path, campaign_root: Path) -> list[dict[str, Any]]:
    """H, S and ORB_INDX of the TBG production chain."""
    h_manifest = campaign_root / "prediction" / "solver_input" / "hamiltonians_pred.manifest.json"
    hamiltonian = _json_or_none(h_manifest) or {}
    h_source = f"manifest:{_repo_relative(h_manifest, repo_root)}"

    overlap_manifest = campaign_root / "overlap" / "overlap_manifest.json"
    overlap = _json_or_none(overlap_manifest) or {}
    o_source = f"manifest:{_repo_relative(overlap_manifest, repo_root)}"
    export = overlap.get("export") if isinstance(overlap.get("export"), dict) else {}

    solvers = _solver_manifests(campaign_root)
    consumers_h = {
        f"{_repo_relative(path, repo_root)}#hamiltonian_sha256": payload.get("hamiltonian_sha256")
        for path, payload in solvers
    }
    consumers_s = {
        f"{_repo_relative(path, repo_root)}#overlap_sha256": payload.get("overlap_sha256")
        for path, payload in solvers
    }

    contract = str(overlap.get("campaign_contract") or "")
    identity_used = overlap.get("identity_overlap_used")
    diagnostics = overlap.get("diagnostics") if isinstance(overlap.get("diagnostics"), dict) else {}
    if not overlap:
        overlap_contract = _evidence("exact_pao_overlap_contract", "absent", "no overlap manifest")
    elif identity_used is not False or diagnostics.get("no_identity_overlap") is not True:
        overlap_contract = _evidence(
            "exact_pao_overlap_contract", "fail", f"identity_overlap_used={identity_used!r}"
        )
    elif str(diagnostics.get("status")) != "valid":
        overlap_contract = _evidence(
            "exact_pao_overlap_contract", "fail", f"overlap diagnostics status={diagnostics.get('status')!r}"
        )
    else:
        overlap_contract = _evidence(
            "exact_pao_overlap_contract",
            "pass",
            f"{contract}; source={diagnostics.get('source')}",
        )

    records = [
        classify_epc_artifact(
            "hamiltonian_H",
            hamiltonian.get("output"),
            path_source=f"{h_source}#output",
            declared_sha256=hamiltonian.get("output_sha256"),
            declared_by=f"{h_source}#output_sha256",
            evidence=[
                _agreement_evidence(
                    "solver_consumers_agree_on_hamiltonian", hamiltonian.get("output_sha256"), consumers_h
                ),
                _evidence(
                    "dft_reference_hamiltonian_available",
                    "absent",
                    "the TBG contract declares reference_hamiltonian_generated="
                    f"{overlap.get('reference_hamiltonian_generated')!r}: this H is a Graph2Mat "
                    "prediction, not a DFT reference",
                ),
            ],
            label="tbg_predicted_hamiltonian_h5",
            repo_root=repo_root,
        ),
        classify_epc_artifact(
            "hamiltonian_H",
            hamiltonian.get("source"),
            path_source=f"{h_source}#source",
            declared_sha256=hamiltonian.get("source_sha256"),
            declared_by=f"{h_source}#source_sha256",
            label="tbg_predicted_hamiltonian_hsx",
            repo_root=repo_root,
        ),
        classify_epc_artifact(
            "overlap_S",
            export.get("overlaps_h5"),
            path_source=f"{o_source}#export.overlaps_h5",
            declared_sha256=export.get("overlaps_h5_sha256"),
            declared_by=f"{o_source}#export.overlaps_h5_sha256",
            evidence=[
                _agreement_evidence(
                    "solver_consumers_agree_on_overlap", export.get("overlaps_h5_sha256"), consumers_s
                ),
                overlap_contract,
            ],
            label="tbg_exact_overlap_h5",
            repo_root=repo_root,
        ),
        classify_epc_artifact(
            "overlap_S",
            overlap.get("raw_onlys"),
            path_source=f"{o_source}#raw_onlys",
            declared_sha256=overlap.get("raw_onlys_sha256"),
            declared_by=f"{o_source}#raw_onlys_sha256",
            label="tbg_siesta_raw_onlyS",
            repo_root=repo_root,
        ),
        classify_epc_artifact(
            "orb_indx",
            hamiltonian.get("orb_indx"),
            path_source=f"{h_source}#orb_indx",
            evidence=[
                _evidence(
                    "orbital_contract_hash_declared",
                    "absent",
                    "the exporter records the ORB_INDX path but no hash; the orbital contract "
                    "of C03 has to hash it itself",
                )
            ],
            label="tbg_orb_indx",
            repo_root=repo_root,
        ),
    ]
    return records


def epc_eigenpair_records(repo_root: Path, campaign_root: Path) -> list[dict[str, Any]]:
    """What the solver output itself declares about persisted eigenpairs (C15)."""
    summary_path = campaign_root / "summary" / "spectral_results.json"
    summary = _json_or_none(summary_path)
    declarations: list[str] = []
    denials = False

    def _walk(payload: Any, trail: str) -> None:
        nonlocal denials
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key == "eigenvectors_persisted":
                    declarations.append(f"{trail}.{key}={value!r}")
                    denials = denials or not value
                else:
                    _walk(value, f"{trail}.{key}")
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                _walk(value, f"{trail}[{index}]")

    _walk(summary, _repo_relative(summary_path, repo_root))
    if declarations:
        note = _evidence(
            "solver_declares_eigenvectors_persisted",
            "fail" if denials else "pass",
            "; ".join(sorted(declarations)[:6]),
        )
    else:
        note = _evidence(
            "solver_declares_eigenvectors_persisted", "absent", "no solver output declares the flag"
        )
    return [
        classify_epc_artifact(
            "eigenpairs",
            None,
            path_source="manifest:<campaign>/summary/spectral_results.json#eigenvectors_persisted",
            evidence=[note],
            label="tbg_eigenpairs",
            repo_root=repo_root,
        )
    ]


def _dedupe(paths: list[Path]) -> list[Path]:
    """One record per artifact: the search roots reach the same tree twice.

    ``Comparison/workspaces`` links back into ``Comparison/results``, so the
    same manifest is discovered under two names; identity is the resolved path.
    """
    unique: dict[Path, Path] = {}
    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path
        unique.setdefault(key, path)
    return sorted(unique.values())


def _search(repo_root: Path, patterns: tuple[str, ...]) -> list[Path]:
    hits: list[Path] = []
    for root in EPC_SEARCH_ROOTS:
        base = repo_root / root
        if not base.is_dir():
            continue
        for pattern in patterns:
            hits.extend(path for path in base.glob(pattern) if path.is_file())
    return _dedupe(hits)


def _phonon_archiver_evidence(repo_root: Path) -> dict[str, Any]:
    source = repo_root / "Comparison" / "scripts" / "analyze_phonons.py"
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        return _evidence("phonon_archiver_preserves_dhsdr", "absent", f"{source} unreadable")
    match = _ANALYZE_PHONONS_PATTERNS.search(text)
    if not match:
        return _evidence("phonon_archiver_preserves_dhsdr", "absent", "no copy pattern tuple found")
    patterns = match.group(1).strip()
    outcome = "pass" if "dHSdR" in patterns else "fail"
    return _evidence(
        "phonon_archiver_preserves_dhsdr",
        outcome,
        f"analyze_phonons.copy_outputs patterns = ({patterns}) — C07 open" if outcome == "fail" else patterns,
    )


def epc_phonon_records(repo_root: Path) -> list[dict[str, Any]]:
    """.FC and *dHSdR.nc: nothing produced them yet, and that is a search result."""
    records = []
    for kind, patterns, evidence in (
        ("force_constants", FC_PATTERNS, []),
        ("dhsdr", DHSDR_PATTERNS, [_phonon_archiver_evidence(repo_root)]),
    ):
        hits = _search(repo_root, patterns)
        source = f"repository_convention_search:{list(EPC_SEARCH_ROOTS)}:{list(patterns)}"
        if not hits:
            records.append(
                classify_epc_artifact(
                    kind,
                    None,
                    path_source=source,
                    evidence=[
                        _evidence("filesystem_search", "absent", f"0 matches for {list(patterns)}"),
                        *evidence,
                    ],
                    label=f"{kind}_search",
                    repo_root=repo_root,
                )
            )
            continue
        for hit in hits:
            records.append(
                classify_epc_artifact(
                    kind,
                    hit,
                    path_source=source,
                    evidence=list(evidence),
                    label=_repo_relative(hit, repo_root),
                    repo_root=repo_root,
                )
            )
    return records


def _stencil_sample_status(sample: dict[str, Any]) -> tuple[str, str]:
    """Verify one stencil sample: fdf bytes, and that SIESTA will not move it.

    An inherited ``MD.TypeOfRun Verlet`` makes SIESTA evolve the geometry away
    from the ±δ displacement, so the stored TSHS is not the finite-difference
    reference it claims to be.
    """
    sample_dir = Path(str(sample.get("sample_dir") or ""))
    metadata_path = sample_dir / "metadata.json"
    run_fdf = sample_dir / "RUN.fdf"
    if not run_fdf.is_file():
        return INVALID, "run_fdf_missing"
    metadata = _json_or_none(metadata_path)
    if not isinstance(metadata, dict):
        return PRESENT_UNVERIFIED, "metadata_missing"

    directives = {
        name.lower(): value.strip().lower()
        for name, value in _FDF_MD_DIRECTIVE.findall(run_fdf.read_text(encoding="utf-8", errors="ignore"))
    }
    # Single point means exactly one SCF at the written positions: CG with zero
    # steps. Anything else (Verlet, an MD block, a Lua driver, a non-zero
    # relaxation) lets SIESTA move the atoms off the ±δ displacement.
    moving = sorted(
        name
        for name, value in directives.items()
        if (name == "md.typeofrun" and value != "cg")
        or (name == "md.numcgsteps" and value != "0")
        or (name.startswith("md.") and name not in ("md.typeofrun", "md.numcgsteps"))
        or name == "lua.script"
    )
    if moving:
        return INVALID, f"md_contaminated_reference:{moving}"

    declared = metadata.get("materialized_fdf_sha256")
    if not declared:
        return PRESENT_UNVERIFIED, "no_materialized_fdf_sha256"
    if file_sha256(run_fdf) != str(declared):
        return INVALID, "materialized_fdf_sha256_mismatch"
    return PRESENT_VALID, "single_point_fdf_bytes_match_metadata"


def _worst_status(statuses: list[str]) -> str:
    for status in (INVALID, MISSING, PRESENT_UNVERIFIED, PRESENT_VALID):
        if status in statuses:
            return status
    return MISSING


worst_status = _worst_status


def epc_stencil_records(repo_root: Path, roots: tuple[str, ...] = EPC_SEARCH_ROOTS) -> list[dict[str, Any]]:
    """One record per derivative-stencil manifest, verified sample by sample."""
    manifests: list[Path] = []
    for root in roots:
        base = repo_root / root
        if base.is_dir():
            manifests.extend(base.glob("**/derivative_stencil_manifest.json"))

    records = []
    for manifest_path in _dedupe(manifests):
        payload = _json_or_none(manifest_path)
        source = f"manifest:{_repo_relative(manifest_path, repo_root)}#samples"
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            records.append(
                classify_epc_artifact(
                    "derivative_stencil",
                    manifest_path,
                    path_source=source,
                    evidence=[_evidence("stencil_manifest_readable", "fail", "no samples list")],
                    label=_repo_relative(manifest_path, repo_root),
                    repo_root=repo_root,
                )
            )
            continue

        samples = payload["samples"]
        tally: dict[str, int] = {}
        reasons: dict[str, int] = {}
        for sample in samples:
            status, reason = _stencil_sample_status(sample)
            tally[status] = tally.get(status, 0) + 1
            reasons[reason] = reasons.get(reason, 0) + 1

        declared_count = payload.get("expected_total_structure_samples")
        count_ok = declared_count in (None, len(samples))
        evidence = [
            _evidence(
                "stencil_sample_count",
                "pass" if count_ok else "fail",
                f"manifest lists {len(samples)} samples, declares {declared_count}",
            ),
            _evidence(
                "stencil_samples_single_point_and_hashed",
                "fail" if tally.get(INVALID) else ("pass" if tally.get(PRESENT_VALID) == len(samples) else "absent"),
                f"tally={dict(sorted(tally.items()))} reasons={dict(sorted(reasons.items()))}",
                verifies_content=tally.get(PRESENT_VALID) == len(samples),
            ),
        ]
        record = classify_epc_artifact(
            "derivative_stencil",
            manifest_path,
            path_source=source,
            evidence=evidence,
            label=_repo_relative(manifest_path, repo_root),
            repo_root=repo_root,
        )
        record["members"] = {"total": len(samples), "by_status": dict(sorted(tally.items()))}
        records.append(record)
    return records


def epc_siesta_reference_records(
    repo_root: Path,
    roots: tuple[str, ...] = EPC_SEARCH_ROOTS,
    *,
    runtime_version_token: str | None = None,
) -> list[dict[str, Any]]:
    """SIESTA H/S (TSHS) and ORB_INDX of the derivative references.

    Revalidated with the existing ``reference_provenance`` validator, which
    rehashes the reference and the ORB_INDX; nothing here trusts the stored
    record on its word.
    """
    try:
        from reference_provenance import validate_positive_reference_provenance
    except ImportError:  # noqa: BLE001 - the inventory must still run
        return []

    found: list[Path] = []
    for root in roots:
        base = repo_root / root
        if base.is_dir():
            found.extend(base.glob("**/siesta_reference_provenance.json"))

    groups: dict[str, list[Path]] = {}
    for path in _dedupe(found):
        # The reusable unit is the reference directory of one campaign, not one
        # displaced snapshot.
        groups.setdefault(str(path.parent.parent), []).append(path)

    records = []
    for group, paths in sorted(groups.items()):
        errors: dict[str, int] = {}
        versions: dict[str, int] = {}
        valid = 0
        for path in sorted(paths):
            payload = _json_or_none(path)
            if not isinstance(payload, dict):
                errors["unreadable_provenance_record"] = errors.get("unreadable_provenance_record", 0) + 1
                continue
            versions[str(payload.get("siesta_version") or "")] = (
                versions.get(str(payload.get("siesta_version") or ""), 0) + 1
            )
            found = validate_positive_reference_provenance(
                payload,
                sample_dir=path.parent,
                reference_path=Path(str(payload.get("reference_path") or "")),
            )
            if not found:
                valid += 1
            for reason in found:
                errors[reason] = errors.get(reason, 0) + 1

        total = len(paths)
        all_valid = valid == total and total > 0
        group_evidence = [
            _evidence(
                "reference_provenance_revalidated",
                "fail" if valid < total else "pass",
                f"{valid}/{total} snapshots revalidate; errors={dict(sorted(errors.items())[:6])}",
                verifies_content=all_valid,
            ),
            _evidence(
                "siesta_version_recorded",
                "fail" if "" in versions else "pass",
                f"versions={dict(sorted(versions.items()))}",
            ),
        ]
        # C01 is a dependency of C02: a reference produced by an unidentified
        # runtime cannot be reused as a SIESTA reference, whatever it hashes to.
        if not runtime_version_token:
            group_evidence.append(
                _evidence("siesta_version_matches_c01_runtime", "absent", "no C01 runtime record to compare against")
            )
        else:
            foreign = sorted(set(versions) - {runtime_version_token})
            group_evidence.append(
                _evidence(
                    "siesta_version_matches_c01_runtime",
                    "fail" if foreign else "pass",
                    f"c01={runtime_version_token} references={dict(sorted(versions.items()))}",
                )
            )
        label = _repo_relative(group, repo_root)
        for kind, what in (
            ("hamiltonian_H", "TSHS H"),
            ("overlap_S", "TSHS S"),
            ("orb_indx", "ORB_INDX"),
        ):
            record = classify_epc_artifact(
                kind,
                group,
                path_source=f"manifest:{label}/*/siesta_reference_provenance.json",
                evidence=list(group_evidence),
                label=f"{label} [{what}]",
                repo_root=repo_root,
                expect="directory",
            )
            record["members"] = {"total": total, "revalidated": valid}
            records.append(record)
    return records


def epc_siesta_runtime_block(repo_root: Path) -> dict[str, Any]:
    """C01 is a dependency of C02: EPC reuse needs an identified runtime."""
    directory = repo_root / SIESTA_RUNTIME_DIR
    records = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not records:
        return {
            "status": MISSING,
            "detail": f"no C01 runtime record under {_repo_relative(directory, repo_root)}; "
            "run `python shared/run_inventory.py --siesta-runtime <binary>` first",
        }
    latest = max(records, key=lambda path: path.stat().st_mtime)
    payload = _json_or_none(latest)
    classification = (
        payload.get("classification", {}).get("status") if isinstance(payload, dict) else None
    )
    ref = {
        "schema": SIESTA_RUNTIME_REF_SCHEMA,
        "record_path": _repo_relative(latest, repo_root),
        "record_sha256": (payload or {}).get("record_sha256"),
        "binary_sha256": (payload or {}).get("binary", {}).get("sha256"),
        "classification": classification,
    }
    validation = validate_siesta_runtime_ref(ref, require_effective=False, repo_root=repo_root)
    validation.pop("record", None)
    return {
        "status": PRESENT_VALID if validation["valid"] else INVALID,
        "ref": ref,
        "version_token": (payload or {}).get("source_candidate", {}).get("version_token"),
        "validation": validation,
        "detail": "VERIFIED_BINARY_ONLY keeps GO-2 exhaustive finite differences mandatory"
        if classification == "VERIFIED_BINARY_ONLY"
        else str(classification),
    }


def epc_orbital_contract_block(repo_root: Path) -> dict[str, Any]:
    """C03 is a dependency of every Graph2Mat<->SIESTA comparison.

    The checkpoint half of the gate needs torch, so it is left to the C03 CLI
    (``python shared/orbital_contract.py``), which writes the full artifact;
    the inventory reports the cheap, file-only half so a preflight cannot start
    a comparison on top of a ghost/basis/ORB_INDX mismatch.
    """
    try:
        from orbital_contract import DECLARED_ONLY as ORBITAL_DECLARED_ONLY
        from orbital_contract import INVALID as ORBITAL_INVALID
        from orbital_contract import certify_systems
    except ImportError as exc:  # noqa: BLE001 - the inventory must still run
        return {"status": PRESENT_UNVERIFIED, "detail": f"orbital_contract unavailable: {exc}"}

    report = certify_systems(repo_root=repo_root, checkpoint=None)
    status = {
        ORBITAL_INVALID: INVALID,
        ORBITAL_DECLARED_ONLY: PRESENT_UNVERIFIED,
    }.get(report["status"], PRESENT_VALID)
    return {
        "status": status,
        "orbital_contract_status": report["status"],
        "basis_contract_hash": report["basis_contract_hash"],
        "systems": {
            record["label"]: {
                "orbital_contract_hash": record["orbital_contract_hash"],
                "atom_count": record["atom_count"],
                "orbital_count": record["orbital_count"],
                "status": record["status"],
                "failed": [item["check"] for item in record["checks"] if item["outcome"] == "fail"],
                "unverified": [item["check"] for item in record["checks"] if item["outcome"] == "absent"],
            }
            for record in report["systems"]
        },
        "cross_checks": report["cross_checks"],
        "detail": "checkpoint half of C03 not evaluated here; run shared/orbital_contract.py for it",
    }


def epc_artifact_dag_contract() -> dict[str, Any]:
    """Publish the S3 signature/invalidation DAG next to the reuse inventory.

    The preflight and the UI read the contract from here instead of re-deriving
    "what invalidates what"; the signatures themselves are produced by
    ``artifact_signature.epc_artifact_node`` inside each producer's manifest.
    """
    return {
        "schema": EPC_ARTIFACT_SIGNATURE_SCHEMA,
        "kinds": {
            kind: {
                "signature_fields": sorted(spec["fields"]),
                "depends_on": {name: list(allowed) for name, allowed in sorted(spec["deps"].items())},
                "optional_dependencies": sorted(spec.get("optional_deps", ())),
                "repeated_dependencies": sorted(spec.get("multi_deps", ())),
                "forbids_synthetic_ancestry": bool(spec.get("requires_non_synthetic_ancestry")),
            }
            for kind, spec in sorted(EPC_ARTIFACT_KINDS.items())
        },
    }


def collect_epc_inventory(
    *,
    repo_root: Path = REPO_ROOT,
    campaign_root: Path | None = None,
    include_run_inventory: bool = True,
) -> dict[str, Any]:
    """Classify every artifact EPC could reuse, with the evidence for each."""
    repo_root = Path(repo_root)
    campaign = Path(campaign_root) if campaign_root else repo_root / TBG_CAMPAIGN_ROOT

    runtime = epc_siesta_runtime_block(repo_root)

    artifacts: list[dict[str, Any]] = []
    artifacts += epc_checkpoint_records(repo_root, campaign)
    artifacts += epc_geometry_records(repo_root, campaign)
    artifacts += epc_matrix_records(repo_root, campaign)
    artifacts += epc_eigenpair_records(repo_root, campaign)
    artifacts += epc_phonon_records(repo_root)
    artifacts += epc_stencil_records(repo_root)
    artifacts += epc_siesta_reference_records(
        repo_root, runtime_version_token=runtime.get("version_token")
    )

    summary: dict[str, int] = {status: 0 for status in EPC_STATUSES}
    by_kind: dict[str, dict[str, int]] = {}
    for record in artifacts:
        summary[record["status"]] = summary.get(record["status"], 0) + 1
        bucket = by_kind.setdefault(record["kind"], {status: 0 for status in EPC_STATUSES})
        bucket[record["status"]] += 1

    payload: dict[str, Any] = {
        "schema": EPC_INVENTORY_SCHEMA,
        "campaign_root": _repo_relative(campaign, repo_root),
        "search_roots": list(EPC_SEARCH_ROOTS),
        "siesta_runtime": runtime,
        "artifacts": artifacts,
        "summary": summary,
        "summary_by_kind": {kind: dict(sorted(counts.items())) for kind, counts in sorted(by_kind.items())},
        "reuse": [
            {"kind": record["kind"], "label": record["label"], "path": record["path"],
             "sha256": record["observed_sha256"],
             "evidence": [item for item in record["evidence"] if item["verifies_content"] and item["outcome"] == "pass"]}
            for record in artifacts
            if record["status"] == PRESENT_VALID
        ],
        "invalidated": [
            {"kind": record["kind"], "label": record["label"], "path": record["path"],
             "reasons": [item["check"] for item in record["evidence"] if item["outcome"] == "fail"],
             "evidence": [item for item in record["evidence"] if item["outcome"] == "fail"]}
            for record in artifacts
            if record["status"] == INVALID
        ],
        "unverified": [
            {"kind": record["kind"], "label": record["label"], "path": record["path"],
             "reasons": [item["check"] for item in record["evidence"] if item["outcome"] == "absent"]}
            for record in artifacts
            if record["status"] == PRESENT_UNVERIFIED
        ],
        "missing": [
            {"kind": record["kind"], "label": record["label"], "path": record["path"]}
            for record in artifacts
            if record["status"] == MISSING
        ],
        "limitations": list(EPC_INVENTORY_LIMITATIONS),
        "orbital_contract": epc_orbital_contract_block(repo_root),
        "artifact_dag": epc_artifact_dag_contract(),
        "compute_policy": dict(COMPUTE_POLICY) | {
            "reason": "C02 only reads and hashes existing files; no numerical workload exists"
        },
    }
    if include_run_inventory:
        payload["run_inventory"] = collect_run_inventory()
    payload["inventory_sha256"] = canonical_sha256(artifacts)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run inventory / SIESTA runtime identity / EPC reuse")
    parser.add_argument("--siesta-runtime", default=None, help="SIESTA executable to freeze")
    parser.add_argument("--output-dir", type=Path, default=SIESTA_RUNTIME_DIR)
    parser.add_argument("--verify-upstream", action="store_true")
    parser.add_argument("--epc-inventory", action="store_true", help="classify EPC-reusable artifacts")
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="write the EPC inventory here")
    args = parser.parse_args(argv)

    if args.epc_inventory:
        inventory = collect_epc_inventory(campaign_root=args.campaign_root)
        destination = args.output or (REPO_ROOT / EPC_INVENTORY_DIR / "epc_reuse_inventory.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The full enumeration lives in the written file; stdout stays readable.
        print(json.dumps({
            "output": _repo_relative(destination, REPO_ROOT),
            "inventory_sha256": inventory["inventory_sha256"],
            "siesta_runtime": inventory["siesta_runtime"]["status"],
            "orbital_contract": inventory["orbital_contract"]["status"],
            "summary": inventory["summary"],
            "summary_by_kind": inventory["summary_by_kind"],
            "missing": [item["label"] for item in inventory["missing"]],
            "unverified": [item["label"] for item in inventory["unverified"]],
            "reuse_sample": [item["label"] for item in inventory["reuse"][:10]],
            "invalidated_sample": [item["label"] for item in inventory["invalidated"][:10]],
        }, indent=2))
        return 0

    if args.siesta_runtime is None:
        print(json.dumps(collect_run_inventory(), indent=2))
        return 0

    payload = collect_siesta_runtime(
        args.siesta_runtime, verify_upstream=args.verify_upstream
    )
    ref = write_siesta_runtime_record(payload, args.output_dir)
    print(json.dumps({
        "ref": ref,
        "reasons": payload["classification"]["reasons"],
        "version_token": payload["source_candidate"]["version_token"],
        "upstream_status": payload["source_link"]["upstream"]["status"],
        "libraries": len(payload["libraries"]["libraries"]),
        "required_followup": payload["required_followup"],
    }, indent=2))
    return 0 if payload["classification"]["status"] != "UNVERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
