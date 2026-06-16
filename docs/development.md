# Development

## Environment Setup

The repository does not include `pyproject.toml`, `setup.py`, `tox.ini`, or
`noxfile.py` at the root, so the local environment is created directly with the
helper script and the pinned requirements file:

```bash
./scripts/create_graph2mat_venv.sh
source .venv/bin/activate
```

The helper script creates `.venv/`, upgrades `pip`, `setuptools`, and `wheel`,
then installs `requirements-graph2mat.txt`.

If `graph2mat` cannot be installed from the Git URL in
`requirements-graph2mat.txt`, install a local checkout into the same virtual
environment:

```bash
python -m pip install -e /path/to/graph2mat
```

## Runtime Dependencies

The documented workflows expect:

- Python 3.12 or a compatible local Python 3 interpreter
- `siesta` in `PATH`
- `graph2mat`
- the DeepH command set when running the Graph2Mat-vs-DeepH benchmark

## Validation Commands

The repository documents these validation commands in `README.md`:

```bash
python3 -m unittest tests/test_comparison_workflow.py
python3 -m unittest tests/test_analyze_winners_three_methods.py
python3 -m unittest tests/test_method_provenance_fairness.py
python3 -m unittest tests/test_material_agnostic_smoke.py
python3 -m unittest tests/test_three_method_scientific_smoke.py
python3 -m unittest tests/test_metrics_material_compatibility.py
python3 -m py_compile Comparison/scripts/pipeline_ui.py Comparison/scripts/evaluate_hamiltonian_metrics.py Comparison/scripts/cleanup_generated_datasets.py
node --check Comparison/ui/app.js
```

The test suite is written with `unittest` rather than `pytest`.

## Coding Style Observed In The Repository

- Most scripts use `#!/usr/bin/env python3` and `from __future__ import annotations`.
- The codebase prefers `pathlib.Path` for filesystem handling.
- Many workflows are implemented as small command-line scripts that print
  machine-readable JSON manifests.
- Tests are organized as plain Python modules under `tests/` and usually end in
  `unittest.main()`.

## Safe Change Patterns

When extending the repository, prefer these patterns:

1. Add or update validation in the existing helper modules instead of
   bypassing them.
2. Keep canonical method IDs in `Comparison/scripts/method_registry.py`.
3. Reuse the existing material-bundle and joint-artifact validators instead of
   hard-coding paths in new scripts.
4. Update the README and the relevant docs page when a new public entry point
   or output artifact is introduced.
5. Add focused `unittest` coverage for any new workflow branch or manifest
   field.

## Practical Notes

- The comparison workflows are stateful and file-driven, so accidental reuse of
  an old workspace can affect results.
- Most debug and benchmark scripts assume they are run from the repository
  root.
- If you change a file path, config key, or manifest field, search the tests and
  docs together; many assertions are string-based and intentionally strict.
