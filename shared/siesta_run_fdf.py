"""Layered SIESTA RUN.fdf rendering shared by dataset generators."""

from __future__ import annotations

from typing import Any


GENERATED_HEADER = "# Generated from pipeline_config.yaml using shared RUN.fdf layers"


def fdf_bool(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"t", "true", ".true."}:
            return "T"
        if text.lower() in {"f", "false", ".false."}:
            return "F"
        return text
    return "T" if bool(value) else "F"


def format_float(value: Any) -> str:
    return f"{float(value):.8f}"


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return fdf_bool(value)
    return str(value)


def species_map(species: list[dict[str, Any]]) -> dict[int, tuple[int, str]]:
    return {
        int(item["index"]): (int(item["atomic_number"]), str(item["symbol"]))
        for item in species
    }


def md_common_settings(md: dict[str, Any]) -> dict[str, Any]:
    return {
        "ForceAuxCell": fdf_bool(md.get("force_aux_cell", True)),
        "MeshCutoff": md.get("mesh_cutoff", "200 Ry"),
        "PAO.BasisType": md.get("basis_type", "split"),
        "PAO.BasisSize": md.get("basis_size", "DZP"),
        "PAO.EnergyShift": md.get("energy_shift", "0.03 eV"),
        "XC.functional": md.get("xc_functional", "GGA"),
        "XC.authors": md.get("xc_authors", "PBE"),
        "MaxSCFIterations": md.get("max_scf_iterations", 200),
        "SolutionMethod": md.get("solution_method", "diagon"),
        "DM.MixingWeight": md.get("dm_mixing_weight", 0.02),
        "DM.NumberPulay": md.get("dm_number_pulay", 3),
        "DM.Tolerance": md.get("dm_tolerance", "1.d-5"),
        "DM.Require.Energy.Convergence": md.get("dm_require_energy_convergence", "T"),
        "DM.Energy.Tolerance": md.get("dm_energy_tolerance", "1.e-5 eV"),
        "SpinPolarized": md.get("spin_polarized", "F"),
        "FixSpin": md.get("fix_spin", "F"),
        "NonCollinearSpin": md.get("non_collinear_spin", "F"),
        "SaveHS": "true" if bool(md.get("save_hs_file", True)) else "false",
        "Save.HS": fdf_bool(md.get("save_hs_file", True)),
        "TS.HS.Save": fdf_bool(md.get("save_hs", True)),
        "TS.DE.Save": fdf_bool(md.get("save_de", True)),
        "XML.Write": fdf_bool(md.get("xml_write", True)),
        "Write.OrbitalIndex": "T",
    }


def render_common_run_fdf(
    *,
    system_name: str,
    system_label: str,
    lattice_constant: dict[str, Any],
    lattice_vectors: list[list[float]],
    species: list[dict[str, Any]],
    coordinates_format: str,
    atoms: list[dict[str, Any]] | None = None,
    positions: list[list[float]] | None = None,
    atom_species: list[int] | None = None,
    kgrid_monkhorst_pack: list[list[Any]] | None = None,
    siesta_settings: dict[str, Any],
    header: str,
) -> str:
    species_by_index = species_map(species)
    if atoms is not None:
        positions = [atom["position"] for atom in atoms]
        atom_species = [int(atom["species_index"]) for atom in atoms]
    if positions is None or atom_species is None:
        raise RuntimeError("render_common_run_fdf requires atoms or positions+atom_species.")

    lines = [
        GENERATED_HEADER,
        f"# Common base: {header}",
        "",
        f"SystemName   {system_name}",
        f"SystemLabel  {system_label}",
        "",
        f"NumberOfSpecies  {len(species_by_index)}",
        f"NumberOfAtoms    {len(atom_species)}",
        "",
        "%block ChemicalSpeciesLabel",
    ]
    for index, (atomic_number, symbol) in sorted(species_by_index.items()):
        lines.append(f" {index:>1}  {atomic_number:>2}  {symbol}")
    lines.extend(
        [
            "%endblock ChemicalSpeciesLabel",
            "",
            f"LatticeConstant  {lattice_constant['value']} {lattice_constant['unit']}",
            "%block LatticeVectors",
        ]
    )
    for vector in lattice_vectors:
        lines.append(f" {format_float(vector[0])}   {format_float(vector[1])}   {format_float(vector[2])}")
    lines.extend(
        [
            "%endblock LatticeVectors",
            "",
            f"AtomicCoordinatesFormat {coordinates_format}",
            "%block AtomicCoordinatesAndAtomicSpecies",
        ]
    )
    for position, species_index in zip(positions, atom_species):
        symbol = species_by_index[int(species_index)][1]
        lines.append(
            f" {format_float(position[0])}  {format_float(position[1])}  "
            f"{format_float(position[2])}  {int(species_index)}  # {symbol}"
        )
    lines.extend(["%endblock AtomicCoordinatesAndAtomicSpecies", ""])
    if kgrid_monkhorst_pack:
        lines.append("%block kgrid_Monkhorst_Pack")
        for row in kgrid_monkhorst_pack:
            lines.append(f" {row[0]}  {row[1]}  {row[2]}  {row[3]}")
        lines.extend(["%endblock kgrid_Monkhorst_Pack", ""])
    for key, value in siesta_settings.items():
        lines.append(f"{key:<32} {format_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def render_md_layer(md: dict[str, Any], block: dict[str, Any] | None = None) -> str:
    block = dict(block or {})
    steps = int(block.get("n_snapshots") or block.get("steps") or md.get("steps") or 0)
    if steps <= 0:
        raise RuntimeError("MD layer requires a positive number of snapshots/steps.")
    temperature = block.get("temperature_K", md.get("temperature_K", md.get("initial_temperature_K", 300.0)))
    timestep_fs = block.get("timestep_fs", md.get("timestep_fs", 1.0))
    ensemble = str(block.get("ensemble", md.get("ensemble", "nve"))).strip().lower()
    thermostat = str(block.get("thermostat", md.get("thermostat", ""))).strip().lower()
    type_of_run = str(block.get("type_of_run", md.get("type_of_run", "Verlet"))).strip()
    if ensemble == "nvt" or thermostat == "nose":
        type_of_run = "Nose"
    elif type_of_run.lower() == "verlet":
        type_of_run = "Verlet"
    lines = [
        "",
        "# MD layer.",
        f"{'MD.TypeOfRun':<32} {type_of_run}",
        f"{'MD.Steps':<32} {steps}",
        f"{'MD.InitialTimeStep':<32} 1",
        f"{'MD.FinalTimeStep':<32} {steps}",
        f"{'MD.LengthTimeStep':<32} {float(timestep_fs):g} fs",
        f"{'MD.InitialTemperature':<32} {float(temperature):g} K",
    ]
    if type_of_run.lower() in {"nose", "noseparrinellorahman"}:
        lines.append(f"{'MD.TargetTemperature':<32} {float(temperature):g} K")
        lines.append(f"{'MD.NoseMass':<32} {block.get('nose_mass', md.get('nose_mass', '100.0 Ry*fs**2'))}")
    lines.append(f"{'WriteMDHistory':<32} {fdf_bool(md.get('write_md_history', True))}")
    lua_script = block.get("lua_script", md.get("lua_script"))
    if lua_script:
        lines.extend(["", "# Store Hamiltonians for each MD step.", f"{'Lua.Script':<32} {lua_script}"])
    return "\n".join(lines).rstrip() + "\n"


def render_fc_layer(force_constants: dict[str, Any], atom_count: int) -> str:
    first_atom = int(force_constants.get("first_atom", 1))
    last_atom = force_constants.get("last_atom")
    if last_atom is None:
        last_atom = atom_count
    lines = ["", "# SIESTA force-constants layer."]
    if "lua_script" in force_constants:
        lines.append(f"{'Lua.Script':<32} {force_constants['lua_script']}")
    lines.append(f"{'TS.HS.Save':<32} {fdf_bool(force_constants.get('save_tshs', True))}")
    lines.append(f"{'TS.DE.Save':<32} {fdf_bool(force_constants.get('save_tsde', True))}")
    lines.append(f"{'MD.TypeOfRun':<32} FC")
    lines.append(f"{'FC.Displacement':<32} {force_constants['displacement']}")
    lines.append(f"{'FC.First':<32} {first_atom}")
    lines.append(f"{'FC.Last':<32} {int(last_atom)}")
    lines.append(f"{'FC.Save.dHS':<32} {fdf_bool(force_constants.get('save_dhs', True))}")
    if "dHdR_tolerance" in force_constants:
        lines.append(f"{'FC.dHdR.Tolerance':<32} {force_constants['dHdR_tolerance']}")
    if "dSdR_tolerance" in force_constants:
        lines.append(f"{'FC.dSdR.Tolerance':<32} {force_constants['dSdR_tolerance']}")
    return "\n".join(lines).rstrip() + "\n"
