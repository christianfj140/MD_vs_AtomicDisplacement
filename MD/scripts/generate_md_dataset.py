#!/usr/bin/env python3
"""Generate the molecular-dynamics dataset from pipeline_config.yaml."""

from __future__ import annotations

import os
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


def _split_counts(total: int) -> tuple[int, int, int]:
    train = int(total * 0.8)
    validation = int(total * 0.1)
    test = total - train - validation
    if total >= 3 and test == 0:
        test = 1
        train = max(1, train - 1)
    return train, validation, test


def _select_spread(items: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)

    used: set[int] = set()
    selected: list[int] = []
    for index in range(count):
        target = min(len(items) - 1, int((index + 0.5) * len(items) / count))
        if target in used:
            target = min(
                (candidate for candidate in range(len(items)) if candidate not in used),
                key=lambda candidate: abs(candidate - target),
            )
        used.add(target)
        selected.append(target)
    return [items[index] for index in sorted(selected)]


def _split_spread(items: list[Path], train_count: int, validation_count: int, test_count: int) -> dict[str, list[Path]]:
    test = _select_spread(items, test_count)
    remaining = [item for item in items if item not in set(test)]
    validation = _select_spread(remaining, validation_count)
    train = [item for item in remaining if item not in set(validation)]
    if len(train) > train_count:
        train = _select_spread(train, train_count)
    return {"train": train, "validation": validation, "test": test}


def _sample_names(samples: list[Path]) -> str:
    return ", ".join(path.name for path in samples) if samples else "-"


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_split_sample(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file():
            _link_or_copy_file(src, dst_dir / src.name)


def prepare_dataset_splits(config: dict) -> None:
    split_config = config.get("splits", {})
    if not bool(split_config.get("enabled", False)):
        return

    pipeline_paths = paths(config)
    steps_dir = pipeline_paths["dataset_dir"] / "MD_steps"
    split_root = pipeline_paths["dataset_dir"] / "splits"
    step_dirs = sorted(
        (path for path in steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    total = int(config["md"]["steps"])
    if len(step_dirs) < total:
        raise RuntimeError(
            f"Se esperaban {total} muestras MD, pero solo hay {len(step_dirs)} en {steps_dir}."
        )

    default_train, default_validation, default_test = _split_counts(total)
    train_count = int(split_config.get("train", default_train))
    validation_count = int(split_config.get("validation", default_validation))
    test_count = int(split_config.get("test", default_test))
    requested = train_count + validation_count + test_count
    if requested > len(step_dirs):
        raise RuntimeError(
            "El split MD pide mas muestras de las disponibles: "
            f"{requested} > {len(step_dirs)}."
        )

    if split_root.exists():
        shutil.rmtree(split_root)

    selected = _select_spread(step_dirs, requested)
    split_ranges = _split_spread(selected, train_count, validation_count, test_count)
    for split_name, samples in split_ranges.items():
        for sample_dir in samples:
            _prepare_split_sample(sample_dir, split_root / split_name / sample_dir.name)

    print(
        "[OK] Split MD preparado: "
        f"{train_count} train, {test_count} test, {validation_count} validation "
        f"en {split_root}"
    )
    print(f"[INFO] MD train samples: {_sample_names(split_ranges['train'])}")
    print(f"[INFO] MD test samples: {_sample_names(split_ranges['test'])}")
    print(f"[INFO] MD validation samples: {_sample_names(split_ranges['validation'])}")


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
    prepare_dataset_splits(config)

    print("\n=== Pipeline completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
