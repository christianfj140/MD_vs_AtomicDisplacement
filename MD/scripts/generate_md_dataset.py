#!/usr/bin/env python3
"""Generate the molecular-dynamics dataset from pipeline_config.yaml."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from md_pipeline_config import command, load_pipeline_config, paths, render_run_fdf


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


def setup_store(config: dict) -> None:
    pipeline_paths = paths(config)
    # Asumimos que `graph2mat siesta md setup-store` puede re-ejecutarse sobre el
    # mismo directorio. Si el contenido ya existe, el comportamiento depende de
    # graph2mat y se respeta tal cual.
    run_command(
        [command(config, "graph2mat"), "siesta", "md", "setup-store"],
        cwd=pipeline_paths["dataset_dir"],
    )


def write_run_fdf(config: dict) -> None:
    pipeline_paths = paths(config)
    # Asumimos que queremos un RUN.fdf determinista: lo sobreescribimos siempre.
    pipeline_paths["run_fdf_path"].write_text(render_run_fdf(config), encoding="utf-8")
    print(f"[OK] RUN.fdf escrito en {pipeline_paths['run_fdf_path']}")


def run_siesta_with_venv(config: dict) -> None:
    pipeline_paths = paths(config)
    venv_activate = pipeline_paths["venv_activate"]
    if not venv_activate.exists():
        raise RuntimeError(
            "No se encontró el script de activación del entorno virtual esperado en "
            f"{venv_activate}."
        )

    bash_cmd = (
        f"source '{venv_activate}' "
        f"&& {command(config, 'siesta')} < {pipeline_paths['run_fdf_path'].name}"
    )

    print(f"\n[RUN] bash -lc \"{bash_cmd}\"")
    with pipeline_paths["run_out_path"].open("w", encoding="utf-8") as run_out:
        process = subprocess.Popen(
            [command(config, "shell"), "-lc", bash_cmd],
            cwd=pipeline_paths["dataset_dir"],
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

    print(f"[OK] Salida guardada en {pipeline_paths['run_out_path']}")


def main() -> int:
    config = load_pipeline_config()
    pipeline_paths = paths(config)

    print("=== Pipeline MD (fase 1): generación de dataset ===")
    print(f"Repositorio: {pipeline_paths['dataset_dir'].parent}")
    print(f"Dataset dir: {pipeline_paths['dataset_dir']}")

    require_command(command(config, "graph2mat"))
    require_command(command(config, "shell"))

    pipeline_paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
    setup_store(config)
    write_run_fdf(config)
    run_siesta_with_venv(config)

    print("\n=== Pipeline completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
