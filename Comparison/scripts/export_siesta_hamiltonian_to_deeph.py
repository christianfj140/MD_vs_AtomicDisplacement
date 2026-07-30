#!/usr/bin/env python3
"""Export a SIESTA/Graph2Mat Hamiltonian to DeepH block HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_siesta_overlap_only import (
    _deeph_blocks,
    _orbital_rows,
    file_sha256,
    open_block_h5,
)


def export(hsx: Path, orb_indx: Path, output: Path) -> dict:
    import sisl

    hamiltonian = sisl.get_sile(str(hsx)).read_hamiltonian()
    rows = _orbital_rows(orb_indx)
    blocks, _orbital_types, adjustment = _deeph_blocks(hamiltonian, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_block_h5(output) as handle:
        for key, block in sorted(blocks.items()):
            handle.create_dataset(str(list(key)), data=block)
    manifest = {
        "status": "completed",
        "source": str(hsx.resolve()),
        "source_sha256": file_sha256(hsx),
        "orb_indx": str(orb_indx.resolve()),
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "n_orbitals": len(rows),
        "n_blocks": len(blocks),
        "energy_unit": "eV",
        "basis_transform": "siesta_real_orbitals_to_deeph_m_zeta_l_order_with_phase",
        "canonical_hermitization_relative_frobenius_adjustment": adjustment,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsx", type=Path, required=True)
    parser.add_argument("--orb-indx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.hsx.resolve(), args.orb_indx.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
