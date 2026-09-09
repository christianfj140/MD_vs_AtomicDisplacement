#!/usr/bin/env python3
"""Run and adjudicate the preregistered PROD-SZ C1:x numerical ladder."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"Comparison/scripts"), str(ROOT/"shared")]
from artifact_signature import file_sha256, input_signature_sha256
from certify_epc_energy_zero import vacuum_level
from certify_support_exact_connection import _hermitian_solve
from onlys_semantic_preflight import _dense_k_points
from run_displaced_projection_sentinel import H_ANG, STEPS, _label, _render_fdf
from run_hamiltonian_derivative_siesta_references import run_siesta
from run_nested_basis_ladder import _rms
from run_nested_basis_sentinel import _matrix_data, _production_run

OUT = ROOT/"Comparison/results/epc/pao_numerical_convergence_v2"
SIESTA = Path("/home/christian/bin/siesta")
WEIGHTS = {-2: 1., -1: -8., 1: 8., 2: -1.}
LIMITS = {"rms": .005, "relative": .01, "maximum": .015}


def _set(text, key, value):
    text, count = re.subn(rf"(?mi)^{re.escape(key)}\s+\S+(?:\s+Ry)?\s*$", f"{key} {value}", text)
    if count != 1: raise ValueError(f"expected one {key}, found {count}")
    return text


def _render(source, label, xyz, mesh, dm):
    text = _render_fdf(source, label, xyz, large_basis=False)
    text = _set(text, "MeshCutoff", f"{mesh} Ry")
    return _set(text, "DM.Tolerance", f"{dm:.1e}")


def _run(root, label, xyz, mesh, dm, source, siesta):
    root.mkdir(parents=True, exist_ok=True)
    fdf = _render(source, label, xyz, mesh, dm)
    (root/"RUN.fdf").write_text(fdf, encoding="utf-8")
    parent = _production_run("C1_x", 0)
    for name in ("C.psf", "Ghost-H.psf"):
        shutil.copy2(parent/name, root/name)
    signature = input_signature_sha256({"fdf": file_sha256(root/"RUN.fdf"), "xyz": xyz.tolist(),
        "mesh_ry": mesh, "dm_tolerance": dm, "siesta": file_sha256(siesta),
        "pseudo": {n: file_sha256(root/n) for n in ("C.psf", "Ghost-H.psf")}})
    manifest = root/"run_manifest.json"
    tshs = root/f"{label}.TSHS"
    if tshs.is_file():
        old = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if old.get("input_signature_sha256") != signature:
            raise RuntimeError(f"unverified cached run at {root}")
        return root
    record = run_siesta(root, command=str(siesta), use_shell=False)
    payload = {"schema": "pao_numerical_run_v2", "input_signature_sha256": signature,
        "mesh_ry": mesh, "dm_tolerance": dm, "geometry_ang": xyz.tolist(),
        "siesta_sha256": file_sha256(siesta), "run_record": record}
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    if record["returncode"] or not tshs.is_file():
        raise RuntimeError(f"SIESTA failed at {root}")
    payload["outputs"] = {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    return root


def _runs(name, mesh, dm, xyz0, source, siesta, rigid_shift=0.):
    if name == "scf_1e-6":
        return {j: _production_run("C1_x", j) for j in STEPS}
    if name == "mesh_600":
        return {j: OUT/"runs/scf_1e-8"/("central" if j == 0 else f"{'m' if j < 0 else 'p'}{abs(j)}") for j in STEPS}
    if name == "egg_box_unshifted":
        return {j: OUT/"runs/mesh_1000"/("central" if j == 0 else f"{'m' if j < 0 else 'p'}{abs(j)}") for j in STEPS}
    result = {}
    for step in STEPS:
        xyz = xyz0.copy(); xyz[:, 0] += rigid_shift; xyz[0, 0] += step*H_ANG
        tag = "central" if step == 0 else f"{'m' if step < 0 else 'p'}{abs(step)}"
        result[step] = _run(OUT/"runs"/name/tag, f"{name}_{tag}", xyz, mesh, dm, source, siesta)
    return result


def _connection(k):
    import sisl
    matrices = []
    no = _matrix_data(_production_run("C1_x", 0))[1].geometry.no
    for step in (-1, 1):
        tag = f"{'m' if step < 0 else 'p'}1"
        path = ROOT/f"Comparison/results/epc/onlys_semantic_preflight/runs/C1_x/{tag}/onlys_c1_x_{tag}.onlyS"
        matrices.append(np.asarray(sisl.get_sile(str(path)).read_overlap().Sk(k=k, format="array"))[:no, no:])
    right = (matrices[1]-matrices[0])/(2*H_ANG)
    return right.conj().T, right


def _response(runs):
    loaded = {j: _matrix_data(path) for j, path in runs.items()}
    output = []
    for k in _dense_k_points():
        matrices = {}
        for j, (_, h, ef, vac, _) in loaded.items():
            s = np.asarray(h.Sk(k=k, format="array"))
            matrices[j] = np.asarray(h.Hk(k=k, format="array")) + (ef-vac)*s
        s0 = np.asarray(loaded[0][1].Sk(k=k, format="array"))
        dk = sum(WEIGHTS[j]*matrices[j] for j in WEIGHTS)/(12*H_ANG)
        left, right = _connection(k)
        x, rx = _hermitian_solve(s0, matrices[0]); y, ry = _hermitian_solve(s0, right)
        output.append((dk-left@x-matrices[0]@y, s0, max(rx, ry)))
    return output


def _compare(coarse, fine):
    rows = []
    for k, (a, b) in zip(_dense_k_points(), zip(coarse, fine)):
        s, solve = b[1], b[2]
        rms, maximum = _rms(b[0]-a[0], s); signal = _rms(b[0], s)[0]
        rows.append({"k_reduced": k.tolist(), "rms_ev_per_ang": rms,
            "relative": rms/max(signal, np.finfo(float).tiny), "maximum_ev_per_ang": maximum,
            "solve_residual": max(a[2], solve)})
    worst = {key: max(row[key] for row in rows) for key in
             ("rms_ev_per_ang", "relative", "maximum_ev_per_ang", "solve_residual")}
    passed = worst["rms_ev_per_ang"] < LIMITS["rms"] and worst["relative"] < LIMITS["relative"] \
        and worst["maximum_ev_per_ang"] < LIMITS["maximum"] and worst["solve_residual"] < 1e-11
    return {"verdict": "PASS" if passed else "NO_GO", "worst": worst, "k_checks": rows}


def _stage(name, settings, xyz0, source, siesta, rigid_shift=0.):
    generated = {key: _runs(f"{name}_{key}", mesh, dm, xyz0, source, siesta,
                            rigid_shift if key == "shifted" else 0.)
                 for key, mesh, dm in settings}
    keys = list(generated)
    comparison = _compare(_response(generated[keys[-2]]), _response(generated[keys[-1]]))
    return {"name": name, "settings": [list(x) for x in settings],
            "decisive_pair": keys[-2:], "comparison": comparison,
            "runs": {key: {str(j): str(path) for j, path in runs.items()} for key, runs in generated.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siesta", type=Path, default=SIESTA)
    args = parser.parse_args(argv)
    import sisl
    parent = _production_run("C1_x", 0)
    source = (parent/"RUN.fdf").read_text()
    xyz0 = sisl.get_sile(str(parent/f"{_label(parent)}.TSHS")).read_geometry().xyz
    report = {"schema": "pao_numerical_convergence_v2", "gate": "numerical_PAO_convergence",
        "generated_at": datetime.now(timezone.utc).isoformat(), "verdict": "BLOCKED",
        "observable": "Delta_PAO_cov in PROD-SZ", "limits": LIMITS, "stages": []}
    plan = [("scf", [("1e-6",600,1e-6),("1e-7",600,1e-7),("1e-8",600,1e-8)], 0.),
            ("mesh", [("600",600,1e-8),("800",800,1e-8),("1000",1000,1e-8)], 0.)]
    for name, settings, shift in plan:
        stage = _stage(name, settings, xyz0, source, args.siesta, shift); report["stages"].append(stage)
        if stage["comparison"]["verdict"] != "PASS": break
    else:
        shape = sisl.get_sile(str(OUT/"runs/mesh_1000/central/mesh_1000_central.VT")).read_grid().shape
        grid_step = np.linalg.norm(sisl.get_sile(str(OUT/"runs/mesh_1000/central/mesh_1000_central.VT")).read_grid().lattice.cell[0])/shape[0]
        stage = _stage("egg_box", [("unshifted",1000,1e-8),("shifted",1000,1e-8)], xyz0, source, args.siesta, grid_step/2)
        stage["rigid_shift_ang"] = grid_step/2; report["stages"].append(stage)
    complete = len(report["stages"]) == 3 and all(s["comparison"]["verdict"] == "PASS" for s in report["stages"])
    report["verdict"] = "PASS" if complete else "NO_GO"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"pao_numerical_convergence_v2.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(report["verdict"], [(s["name"],s["comparison"]["verdict"],s["comparison"]["worst"]) for s in report["stages"]])
    return 0 if complete else 1


if __name__ == "__main__": raise SystemExit(main())
