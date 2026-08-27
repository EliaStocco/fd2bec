from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from warnings import warn

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

ESPRESSO_GEOMETRY_FORMAT = "espresso-in"


def inferred_output_format(output: Path):
    """Return special output routes that cannot be selected by ASE alone."""
    return "cif" if output.suffix.lower() == ".cif" else None


def write(*argv, **kwargs):
    return ase_write(*argv, **kwargs)


def read_numeric_data(filename: Union[str, Path]) -> np.ndarray:
    """Read numeric whitespace- or comma-delimited data from ``filename``.

    The returned array always has at least one dimension, so a file containing
    one number can be used as a structure-level scalar property.
    """
    filename = Path(filename)
    delimiter = "," if filename.suffix.lower() == ".csv" else None
    return np.loadtxt(filename, delimiter=delimiter, ndmin=1)


def add_extxyz_data(
    structures: Iterable[Atoms],
    data,
    name: str,
    what: str,
    *,
    replicate: bool = False,
) -> List[Atoms]:
    """Return copies of structures with numeric data added under ``name``.

    ``what`` selects ``"info"`` for one value per structure or ``"arrays"``
    for one value per atom. Per-atom data is reshaped to
    ``(n_structures, n_atoms, -1)``; therefore every structure must contain
    the same number of atoms.
    """
    structures = list(structures)
    if not structures:
        raise ValueError("The input extxyz contains no structures.")
    if not name:
        raise ValueError("The data name cannot be empty.")

    location = {
        "i": "info",
        "info": "info",
        "a": "arrays",
        "array": "arrays",
        "arrays": "arrays",
    }
    try:
        what = location[what.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("'what' must be 'i'/'info' or 'a'/'array'/'arrays'.") from error

    data = np.asarray(data)
    if replicate:
        data = np.repeat(data[np.newaxis, ...], len(structures), axis=0)

    if what == "info":
        if data.ndim == 0:
            if len(structures) != 1:
                raise ValueError(
                    "Scalar data can only be assigned to one structure; use replicate=True "
                    "to assign it to every structure."
                )
            values = data.reshape(1)
        else:
            if data.shape[0] != len(structures):
                raise ValueError(
                    f"Info data has {data.shape[0]} entries but the input has "
                    f"{len(structures)} structures."
                )
            values = data
    else:
        atom_counts = {len(atoms) for atoms in structures}
        if len(atom_counts) != 1:
            raise ValueError(
                "Per-atom data requires every input structure to have the same atom count."
            )
        n_atoms = atom_counts.pop()
        expected = len(structures) * n_atoms
        if data.size % expected:
            raise ValueError(
                f"Per-atom data contains {data.size} values, which cannot be reshaped "
                f"for {len(structures)} structures with {n_atoms} atoms each."
            )
        values = data.reshape((len(structures), n_atoms, -1))

    output = []
    for index, atoms in enumerate(structures):
        updated = atoms.copy()
        if what == "info":
            updated.info[name] = values[index]
        else:
            updated.set_array(name, values[index])
        output.append(updated)
    return output


def add_born_effective_charges(
    structures: Iterable[Atoms], data, *, key: str = "REF_BEC", replicate: bool = False
) -> List[Atoms]:
    """Attach flattened 3-by-3 Born effective charge tensors to structures.

    ``data`` must contain nine components for every atom in every structure.
    Set ``replicate`` to use one structure's BECs for every structure in a
    trajectory.
    """
    structures = list(structures)
    if not structures:
        raise ValueError("The input extxyz contains no structures.")
    atom_counts = {len(atoms) for atoms in structures}
    if len(atom_counts) != 1:
        raise ValueError("BEC data requires every input structure to have the same atom count.")
    n_atoms = atom_counts.pop()
    per_structure = n_atoms * 9
    values = np.asarray(data, dtype=float)
    expected = per_structure if replicate else len(structures) * per_structure
    if values.size != expected:
        raise ValueError(
            f"BEC data has {values.size} values; expected {expected} "
            f"({n_atoms} atoms × 9 components" + (")." if replicate else " per structure).")
        )
    values = values.reshape((1 if replicate else len(structures), n_atoms, 9))
    if replicate:
        values = np.repeat(values, len(structures), axis=0)

    output = []
    for index, atoms in enumerate(structures):
        updated = atoms.copy()
        updated.set_array(key, values[index])
        output.append(updated)
    return output


def add_proper_piezoelectric_tensors(
    structures: Iterable[Atoms],
    data,
    *,
    key: str = "REF_piezoelectric",
    replicate: bool = False,
) -> List[Atoms]:
    """Attach 3-by-6 proper piezoelectric tensors to structure metadata.

    The input uses the project's Voigt order: ``xx, yy, zz, yz, xz, xy``.
    Set ``replicate`` to apply one tensor to every structure in a trajectory.
    """
    structures = list(structures)
    if not structures:
        raise ValueError("The input extxyz contains no structures.")
    values = np.asarray(data, dtype=float)
    per_structure = 18
    expected = per_structure if replicate else len(structures) * per_structure
    if values.size != expected:
        raise ValueError(
            f"Proper piezoelectric data has {values.size} values; expected {expected} "
            "(3 × 6 components per structure)."
        )
    values = values.reshape((1 if replicate else len(structures), 3, 6))
    if replicate:
        values = np.repeat(values, len(structures), axis=0)

    output = []
    for index, atoms in enumerate(structures):
        updated = atoms.copy()
        updated.info[key] = values[index]
        output.append(updated)
    return output


def write_tensor_extxyz(output: Path, atoms: Atoms, data, keyword: str, *, per_atom: bool) -> None:
    """Write an extxyz structure with a tensor stored under ``keyword``.

    Per-atom tensors are flattened into extxyz vector properties; global
    tensors are stored in the structure's ``info`` dictionary.
    """
    structure = atoms.copy()
    tensor = np.asarray(data, dtype=float)
    if per_atom:
        if tensor.ndim < 2 or tensor.shape[0] != len(structure):
            raise ValueError(
                "A per-atom tensor must have one leading entry for each atom; "
                f"got shape {tensor.shape} for {len(structure)} atoms."
            )
        structure.new_array(keyword, tensor.reshape((len(structure), -1)))
    else:
        structure.info[keyword] = tensor
    write(output, structure, format="extxyz")


def espresso_geometry(atoms: Atoms) -> str:
    """Return QE ``CELL_PARAMETERS`` and fractional ``ATOMIC_POSITIONS`` cards."""
    if not np.all(atoms.get_pbc()):
        raise ValueError("espresso-in geometry requires a fully periodic structure.")
    if abs(np.linalg.det(atoms.cell.array)) < 1e-14:
        raise ValueError("espresso-in geometry requires a non-singular cell.")

    lines = ["CELL_PARAMETERS angstrom"]
    for vector in atoms.cell.array:
        lines.append("  " + "  ".join(f"{value:.12f}" for value in vector))

    lines.extend(("", "ATOMIC_POSITIONS crystal"))
    scaled_positions = atoms.get_scaled_positions(wrap=False)
    for symbol, position in zip(atoms.get_chemical_symbols(), scaled_positions):
        coordinates = "  ".join(f"{value:.12f}" for value in position)
        lines.append(f"{symbol:<3s}  {coordinates}")

    return "\n".join(lines) + "\n"


def write_espresso_geometry(filename: Path, atoms: Atoms) -> None:
    """Write the geometry portion of a Quantum Espresso input file."""
    filename.write_text(espresso_geometry(atoms), encoding="utf-8")


def _cif_number(value: float, precision: int = 8) -> str:
    """Format a CIF number, mapping periodic round-off at one back to zero."""
    value %= 1.0
    if np.isclose(value, 0.0, atol=1e-8) or np.isclose(value, 1.0, atol=1e-8):
        value = 0.0
    return f"{value:.{precision}f}"


def _cif_symmetry_expression(rotation: np.ndarray, translation: float) -> str:
    """Convert one fractional affine-coordinate row to CIF ``x,y,z`` syntax."""
    coordinates = ("x", "y", "z")
    terms = []
    for coefficient, coordinate in zip(rotation, coordinates):
        coefficient = int(coefficient)
        if coefficient == 1:
            terms.append(coordinate)
        elif coefficient == -1:
            terms.append(f"-{coordinate}")
        elif coefficient:
            terms.append(f"{coefficient}{coordinate}")

    translation %= 1.0
    if not np.isclose(translation, 0.0, atol=1e-8) and not np.isclose(translation, 1.0, atol=1e-8):
        fraction = Fraction(float(translation)).limit_denominator(96)
        if not np.isclose(float(fraction), translation, atol=1e-8):
            raise ValueError(f"Cannot represent CIF symmetry translation {translation!r} exactly.")
        shift = str(fraction)
        terms.append(f"+{shift}" if terms else shift)
    return "".join(terms) or "0"


def write_input_symmetry_cif(output: Path, atoms: Atoms, *, symprec: float) -> None:
    """Write a symmetry-expanding CIF without standardizing the input cell.

    The CIF contains the symmetry operations expressed in the supplied
    fractional basis and one representative per crystallographic orbit.  It
    therefore expands back to the original number of atoms while preserving
    the input cell, including for non-primitive supercells.
    """
    if not np.all(atoms.pbc):
        raise ValueError("Symmetry-aware CIF output requires a fully periodic structure.")

    import spglib

    fractional_positions = atoms.get_scaled_positions(wrap=True)
    dataset = spglib.get_symmetry_dataset(
        (atoms.cell.array, fractional_positions, atoms.numbers), symprec=symprec
    )
    if dataset is None:
        raise ValueError("spglib could not determine the structure symmetry.")

    symbols = atoms.get_chemical_symbols()
    equivalent_atoms = np.asarray(dataset.equivalent_atoms, dtype=int)
    representative_indices = []
    seen_orbits = set()
    for index, orbit in enumerate(equivalent_atoms):
        if orbit not in seen_orbits:
            representative_indices.append(index)
            seen_orbits.add(orbit)

    cell_parameters = atoms.cell.cellpar()
    counts = [symbols.count(symbol) for symbol in sorted(set(symbols))]
    formula_units = 0
    for count in counts:
        formula_units = gcd(formula_units, count)
    formula = atoms.get_chemical_formula(mode="reduce")
    international = (
        dataset.international.decode()
        if isinstance(dataset.international, bytes)
        else dataset.international
    )

    lines = [
        "# Input-cell symmetry CIF generated by fd2bec.",
        f"# Detected space group: {international} (No. {dataset.number}).",
        "data_" + formula,
        f"_space_group_name_H-M_alt   '{international}'",
        f"_space_group_IT_number   {dataset.number}",
        f"_chemical_formula_structural   {formula}",
        f"_chemical_formula_sum   '{atoms.get_chemical_formula()}'",
        f"_cell_formula_units_Z   {formula_units}",
        f"_cell_length_a   {cell_parameters[0]:.8f}",
        f"_cell_length_b   {cell_parameters[1]:.8f}",
        f"_cell_length_c   {cell_parameters[2]:.8f}",
        f"_cell_angle_alpha   {cell_parameters[3]:.8f}",
        f"_cell_angle_beta   {cell_parameters[4]:.8f}",
        f"_cell_angle_gamma   {cell_parameters[5]:.8f}",
        f"_cell_volume   {atoms.get_volume():.8f}",
        "loop_",
        " _symmetry_equiv_pos_site_id",
        " _symmetry_equiv_pos_as_xyz",
    ]
    for index, (rotation, translation) in enumerate(
        zip(dataset.rotations, dataset.translations), start=1
    ):
        operation = ", ".join(
            _cif_symmetry_expression(row, shift) for row, shift in zip(rotation, translation)
        )
        lines.append(f"  {index}  '{operation}'")

    lines.extend(
        (
            "loop_",
            " _atom_site_type_symbol",
            " _atom_site_label",
            " _atom_site_symmetry_multiplicity",
            " _atom_site_fract_x",
            " _atom_site_fract_y",
            " _atom_site_fract_z",
            " _atom_site_occupancy",
        )
    )
    label_counts = {}
    for index in representative_indices:
        symbol = symbols[index]
        label_index = label_counts.get(symbol, 0)
        label_counts[symbol] = label_index + 1
        multiplicity = int(np.count_nonzero(equivalent_atoms == equivalent_atoms[index]))
        coordinates = "  ".join(_cif_number(value) for value in fractional_positions[index])
        lines.append(f"  {symbol}  {symbol}{label_index}  {multiplicity}  {coordinates}  1")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_primitive_symmetry_cif(output: Path, atoms: Atoms, *, symprec: float) -> None:
    """Write a symmetry-expanding CIF for a primitive input cell.

    Kept as a public compatibility wrapper.  The writer itself works for any
    periodic input cell and never standardizes it.
    """
    write_input_symmetry_cif(output, atoms, symprec=symprec)


def write_symmetry_cif(
    output: Path,
    atoms: Atoms,
    *,
    symprec: float,
    conventional: bool,
    primitive: bool = False,
) -> None:
    """Write a CIF containing detected space-group metadata and operations."""
    if not np.all(atoms.pbc):
        raise ValueError("Symmetry-aware CIF output requires a fully periodic structure.")
    if primitive:
        if conventional:
            raise ValueError("Choose either primitive or conventional CIF output, not both.")
        write_primitive_symmetry_cif(output, atoms, symprec=symprec)
        return
    if not conventional:
        write_input_symmetry_cif(output, atoms, symprec=symprec)
        return
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = AseAtomsAdaptor.get_structure(atoms)
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    structure = analyzer.get_conventional_standard_structure()
    CifWriter(structure, symprec=symprec).write_file(str(output))


def write_structure(
    output: Path,
    atoms: Atoms,
    output_format: Optional[str],
    *,
    symprec: float,
    conventional: bool,
    primitive: bool = False,
) -> None:
    """Write one structure through the appropriate output-format implementation."""
    if output_format == "cif":
        write_symmetry_cif(
            output,
            atoms,
            symprec=symprec,
            conventional=conventional,
            primitive=primitive,
        )
    elif output_format == ESPRESSO_GEOMETRY_FORMAT:
        write_espresso_geometry(output, atoms)
    elif output_format is None:
        write(output, atoms)
    else:
        write(output, atoms, format=output_format)


def read(*argv, rename: bool = False, **kwargs):
    structures = ase_read(*argv, **kwargs)
    if isinstance(structures, Atoms):
        return format_atoms(structures, rename)
    elif isinstance(structures, list):
        return [format_atoms(structure, rename) for structure in structures]
    else:
        raise TypeError(f"type not supported: {type(structures)}.")


def format_atoms(atom: Atoms, rename: bool) -> Atoms:
    if atom.calc is not None:
        results: Dict[str, Union[float, np.ndarray]] = atom.calc.results
        for key, value in results.items():
            if key in ["energy", "free_energy", "dipole", "stress"]:
                if rename:
                    atom.info[f"REF_{key}"] = value
                    warn(f"Renaming '{key}' to 'REF_{key}.'")
                else:
                    atom.info[key] = value
                    # warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
            elif key in ["forces"]:
                if rename:
                    atom.arrays[f"REF_{key}"] = value
                    warn(f"Renaming '{key}' to 'REF_{key}.'")
                else:
                    atom.arrays[key] = value
                    # warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
            else:
                if rename:
                    atom.info[f"REF_{key}"] = value
                    warn(f"Renaming '{key}' to 'REF_{key}.'")
                else:
                    atom.info[key] = value
                    # warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
    atom.calc = None
    return atom
