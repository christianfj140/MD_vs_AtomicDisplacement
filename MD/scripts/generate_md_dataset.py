#!/usr/bin/env python3
"""Automatiza la generación del dataset de dinámica molecular (fase 1).

Este script replica de forma fiel los pasos de `MD/command_history.txt` para
la parte de entrenamiento relacionada con la generación del dataset MD:

1) mkdir dataset
2) cd dataset
3) graph2mat siesta md setup-store
4) crear RUN.fdf
5) source /home/christian/graph2mat-env/bin/activate
6) siesta < RUN.fdf | tee RUN.out

Notas:
- Se usan rutas y valores hardcodeados intencionalmente para esta primera fase.
- Se incluye una estructura en funciones para facilitar parametrización futura.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# -----------------------
# Hardcoded configuration
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "MD" / "dataset"
RUN_FDF_PATH = DATASET_DIR / "RUN.fdf"
RUN_OUT_PATH = DATASET_DIR / "RUN.out"
VENV_ACTIVATE = Path("/home/christian/graph2mat-env/bin/activate")

RUN_FDF_CONTENT = """# Run 50 steps of a verlet MD
MD.TypeOfRun verlet
MD.Steps 50

# Use the default Double Zeta Polarized basis.
PAO.BasisSize DZP

# Save all matrices
TS.HS.Save t
TS.DE.Save t

# Specify that we want to use our lua script
Lua.Script md_store.lua

# ForceAuxCell is not really needed here, but you will need it if you are
# computing a periodic system only at the Gamma point.
ForceAuxCell t

# And then the information about the structure

# The lattice is just a box big enough so that periodic images don't interact.
LatticeConstant 1.0 Ang
%block LatticeVectors
10.00000000 0.00000000 0.00000000
0.00000000 10.00000000 0.00000000
0.00000000 0.00000000 10.00000000
%endblock LatticeVectors

# Two species, Oxygen and Hydrogen
NumberOfSpecies 2
%block ChemicalSpeciesLabel
1 8 O
2 1 H
%endblock ChemicalSpeciesLabel

# The coordinates of the water molecule
NumberOfAtoms 3
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
5.00000000  5.00000000  0.11926200 1 # 1: O
5.00000000  5.76323900 -0.47704700 2 # 2: H
5.00000000  4.33683900 -0.47704700 2 # 3: H
%endblock AtomicCoordinatesAndAtomicSpecies
"""


def require_command(command_name: str) -> None:
    """Verifica que un ejecutable exista en PATH."""
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontró '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def run_command(cmd: list[str], cwd: Path) -> None:
    """Ejecuta un comando y falla con mensaje claro si retorna código != 0."""
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El comando falló con código {result.returncode}: {' '.join(cmd)}"
        )


def ensure_dataset_dir() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def setup_store() -> None:
    # Asumimos que `graph2mat siesta md setup-store` puede re-ejecutarse sobre el
    # mismo directorio. Si el contenido ya existe, el comportamiento depende de
    # graph2mat y se respeta tal cual.
    run_command(["graph2mat", "siesta", "md", "setup-store"], cwd=DATASET_DIR)


def write_run_fdf() -> None:
    # Asumimos que queremos un RUN.fdf determinista: lo sobreescribimos siempre.
    RUN_FDF_PATH.write_text(RUN_FDF_CONTENT, encoding="utf-8")
    print(f"[OK] RUN.fdf escrito en {RUN_FDF_PATH}")


def run_siesta_with_venv() -> None:
    if not VENV_ACTIVATE.exists():
        raise RuntimeError(
            "No se encontró el script de activación del entorno virtual esperado en "
            f"{VENV_ACTIVATE}."
        )

    # Mantiene la fidelidad con el flujo original:
    #   source /home/christian/graph2mat-env/bin/activate
    #   siesta < RUN.fdf | tee RUN.out
    bash_cmd = (
        f"source '{VENV_ACTIVATE}' "
        "&& siesta < RUN.fdf"
    )

    print(f"\n[RUN] bash -lc \"{bash_cmd}\"")
    with RUN_OUT_PATH.open("w", encoding="utf-8") as run_out:
        process = subprocess.Popen(
            ["bash", "-lc", bash_cmd],
            cwd=DATASET_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            run_out.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"siesta terminó con código {return_code}.")

    print(f"[OK] Salida guardada en {RUN_OUT_PATH}")


def main() -> int:
    print("=== Pipeline MD (fase 1): generación de dataset ===")
    print(f"Repositorio: {REPO_ROOT}")
    print(f"Dataset dir: {DATASET_DIR}")

    require_command("graph2mat")
    require_command("bash")

    ensure_dataset_dir()
    setup_store()
    write_run_fdf()
    run_siesta_with_venv()

    print("\n=== Pipeline completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
