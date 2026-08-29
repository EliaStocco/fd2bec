"""Standardized terminal presentation for fd2bec."""

import argparse
from itertools import product
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike
from spglib import SpglibDataset

from fd2bec import ATOL, BEC_NORM_THRESHOLD
from fd2bec.atomic import AtomicStructure
from fd2bec.tensor_components import (
    CARTESIAN_LABELS,
    VOIGT_LABELS,
    forbidden_component_indices,
    format_numeric_components,
    parameter_name,
)

Axis = Mapping[str, Any]
Structure = Union[Atoms, AtomicStructure]
ScriptGroups = Mapping[str, Sequence[Tuple[str, Optional[str]]]]


def print_input_arguments(args: argparse.Namespace) -> None:
    """Print every parsed CLI argument, including parser defaults."""
    print("\t" + "-" * 40)
    print("\tInput arguments:")
    for name, value in vars(args).items():
        print(f"\t {name:>20s}: {value}")
    print("\t" + "-" * 40)
    print()


def print_scripts(
    scripts: ScriptGroups,
    show_folders: bool = False,
    descriptions: bool = False,
    color: bool = True,
) -> None:
    """Print grouped CLI script names and, optionally, their descriptions."""
    blue = "\033[1;34m" if color else ""
    green = "\033[0;32m" if color else ""
    reset = "\033[0m" if color else ""

    for folder, entries in scripts.items():
        print(f"\t{blue}{folder}:{reset}")
        if show_folders:
            continue

        width = max(len(filename) for filename, _ in entries)
        for filename, description in entries:
            if descriptions:
                text = description or "No description found"
                print(f"\t - {green}{filename:<{width}}{reset}: {text}")
            else:
                print(f"\t - {green}{filename}{reset}")
        print()


def print_matrix(matrix: ArrayLike, precision: int = 8) -> None:
    """Print a two-dimensional numeric matrix with aligned columns."""
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError(f"Expected a two-dimensional matrix, got shape {values.shape}.")
    for row in values:
        print("    " + "  ".join(f"{value:14.{precision}f}" for value in row))


def print_cell(atoms: Atoms, precision: int = 8) -> None:
    """Print cell vectors, lattice parameters, and volume."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    print("Cell vectors [Angstrom]:")
    for index, vector in enumerate(cell, start=1):
        print(f"  a{index}: " + "  ".join(f"{value:14.{precision}f}" for value in vector))

    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    print("\nLattice parameters:")
    print(f"  a, b, c [Angstrom]       = {a:.{precision}f}  {b:.{precision}f}  {c:.{precision}f}")
    print(
        f"  alpha, beta, gamma [deg] = {alpha:.{precision}f} "
        f"{beta:.{precision}f} {gamma:.{precision}f}"
    )
    print(f"  volume [Angstrom^3]      = {atoms.get_volume():.{precision}f}")


def print_positions(atoms: Atoms, precision: int = 6) -> None:
    """Print Cartesian and fractional coordinates for every atom."""
    cartesian = np.asarray(atoms.get_positions(), dtype=float)
    fractional = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    width = precision + 5
    print("Positions (Cartesian [Angstrom] and fractional):")
    print(
        "  index  atom"
        + "".join(f"{label:>{width + 2}}" for label in ("Rx", "Ry", "Rz", "fx", "fy", "fz"))
    )
    for index, (symbol, position, scaled) in enumerate(
        zip(atoms.get_chemical_symbols(), cartesian, fractional), start=1
    ):
        values = "  ".join(f"{value:{width}.{precision}f}" for value in (*position, *scaled))
        print(f"  {index:5d}  {symbol:>4s}  {values}")


def print_structure(atoms: Atoms, title: str = "Structure information") -> None:
    """Print a common structure summary followed by coordinates and, when periodic, its cell."""
    print(f"{title} ({atoms.get_chemical_formula()}):")
    print(f"  atoms = {len(atoms)}")
    print(f"  periodic boundary conditions = {atoms.get_pbc().tolist()}")
    print(f"  total mass [amu] = {atoms.get_masses().sum():.8f}")
    if np.all(atoms.get_pbc()):
        print()
        print_cell(atoms)
    print()
    print_positions(atoms)


def print_displacement_input_structure(atoms: Atoms) -> None:
    """Print the periodic input structure used for finite displacements."""
    if not np.all(atoms.get_pbc()):
        raise ValueError("This command requires a fully periodic input structure.")
    print_structure(atoms, title="Input structure")


def print_reference_structure(reference: Atoms) -> None:
    """Print a reference structure using the common structure layout."""
    print("-" * 20)
    print_structure(reference, title="Reference structure")
    print("-" * 20)
    print()


def _spglib_text(value: object) -> str:
    """Convert spglib strings, including byte strings, to text."""
    return value.decode() if isinstance(value, bytes) else str(value)


def print_space_group(dataset: SpglibDataset, atoms: Atoms, symprec: float) -> None:
    """Print a standardized summary of a spglib space-group dataset."""
    try:
        bravais_type = atoms.cell.get_bravais_lattice(eps=symprec).longname
    except (AttributeError, ValueError):
        bravais_type = "undetermined"
    centrosymmetric = any(
        np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in dataset.rotations
    )
    fields = (
        ("International symbol", _spglib_text(dataset.international)),
        ("Number", dataset.number),
        ("Crystal class", _spglib_text(dataset.pointgroup)),
        ("Bravais lattice type", bravais_type),
        ("Number of symmetry operations", len(dataset.rotations)),
        ("Centrosymmetric", "yes" if centrosymmetric else "no"),
    )
    print("Space-group summary:")
    for label, value in fields:
        print(f"  {label:<27}: {value}")


def print_symmetry_operations(dataset: SpglibDataset, precision: int = 8) -> None:
    """Print spglib symmetry operations in fractional coordinates."""
    print("Symmetry operations (fractional coordinates x' = R x + t):")
    for index, (rotation, translation) in enumerate(
        zip(dataset.rotations, dataset.translations), start=1
    ):
        print(f"  #{index}")
        print("    rotation:")
        print_matrix(rotation, precision=0)
        print("    translation: " + "  ".join(f"{value: .{precision}f}" for value in translation))


def print_symmetry_selection(
    unit_cell: AtomicStructure, displacements: np.ndarray, atomic: bool
) -> None:
    """Print space-group and selected finite-displacement information."""
    dataset = unit_cell._spglib_dataset  # pylint: disable=protected-access
    symbol = _spglib_text(getattr(dataset, "international", "unknown"))
    print(
        f"Space group: {dataset.number} ({symbol}); {len(dataset.rotations)} symmetry operations."
    )
    kind = "atomic" if atomic else "cell"
    number_of_displacements = np.count_nonzero(np.linalg.norm(displacements, axis=1) > 1e-14)
    print(
        f"Symmetry-selected {kind} displacements: {number_of_displacements}; "
        f"{len(displacements)} structures including the reference."
    )
    shape = (len(unit_cell), 3) if atomic else (3, 3)
    cartesian_axes = "xyz"
    cell_vectors = "abc"
    for index, displacement in enumerate(displacements):
        matrix = displacement.reshape(shape)
        nonzero = np.argwhere(np.abs(matrix) > 1e-14)
        if len(nonzero) == 0:
            formatted = "reference (zero)"
        elif atomic:
            formatted = ", ".join(
                f"{unit_cell.symbols[row]}[{row}].{cartesian_axes[column]}="
                f"{matrix[row, column]:.6g}"
                for row, column in nonzero
            )
        else:
            formatted = ", ".join(
                f"{cell_vectors[row]}.{cartesian_axes[column]}={matrix[row, column]:.6g}"
                for row, column in nonzero
            )
        print(f"  [{index}] {formatted}")


def print_born_charges(reference: Structure, bec: ArrayLike) -> None:
    """Print per-atom Born effective charge tensors and warn about large norms."""
    values = np.asarray(bec, dtype=float)
    if values.shape != (len(reference.symbols), 3, 3):
        raise ValueError(
            f"Born effective charges must have shape ({len(reference.symbols)}, 3, 3), "
            f"got {values.shape}."
        )
    symbols = (
        reference.get_chemical_symbols() if isinstance(reference, Atoms) else reference.symbols
    )
    positions = reference.get_positions() if isinstance(reference, Atoms) else reference.positions
    for index, (symbol, position, tensor) in enumerate(zip(symbols, positions, values)):
        norm = np.linalg.norm(tensor, "fro") / np.sqrt(tensor.shape[0])
        print(f"Atom {index:3d}, species {symbol}")
        print(f"Position: [{position[0]:10.6f} {position[1]:10.6f} {position[2]:10.6f}]")
        print(f"||Z*|| = {norm:10.5f}")
        print("Born effective charge tensor (Z*):")
        print_matrix(tensor, precision=5)
        print()
        if BEC_NORM_THRESHOLD is not None and norm > BEC_NORM_THRESHOLD:
            warn(
                f"Large Born effective charge detected for atom {index} ({symbol}): "
                f"||Z*|| = {norm:.3f} > {BEC_NORM_THRESHOLD:.3f}"
            )


def _display_number(value: float, precision: int, zero_tolerance: float = 5e-10) -> str:
    """Format a fixed-width-friendly number without negative numerical zero."""
    value = 0.0 if abs(value) < zero_tolerance else value
    return f"{value:.{precision}f}"


def print_voigt_tensor(tensor: ArrayLike, precision: int = 6) -> None:
    """Print a numerical 3-by-6 tensor with Cartesian and Voigt labels."""
    values = np.asarray(tensor, dtype=float)
    if values.shape != (3, 6):
        raise ValueError(f"Expected a 3x6 tensor, got {values.shape}.")
    width = precision + 8
    print(" " * 6 + "".join(f"{label:>{width}}" for label in VOIGT_LABELS))
    for axis, row in zip(CARTESIAN_LABELS, values):
        formatted = "".join(f"{_display_number(value, precision):>{width}}" for value in row)
        print(f"P_{axis:<4s}{formatted}")


def print_lattice_tensor(tensor: ArrayLike, precision: int = 6) -> None:
    """Print a rank-3 lattice-basis tensor as three labeled 3-by-3 slices."""
    values = np.asarray(tensor, dtype=float)
    if values.shape != (3, 3, 3):
        raise ValueError(f"Expected a 3x3x3 tensor, got {values.shape}.")
    width = precision + 8
    for component, block in zip(CARTESIAN_LABELS, values):
        print(f"P_{component} component:")
        print(" " * 7 + "".join(f"{label:>{width}}" for label in CARTESIAN_LABELS))
        for axis, row in zip(CARTESIAN_LABELS, block):
            formatted = "".join(f"{_display_number(value, precision):>{width}}" for value in row)
            print(f"  {axis:<5s}{formatted}")


def _axis_labels(axis: Axis, size: int) -> list[str]:
    labels = axis.get("labels")
    if labels is not None:
        if len(labels) != size:
            raise ValueError("Explicit axis labels must match the component axis length.")
        return list(labels)
    if axis.get("type") == "cartesian":
        return list(CARTESIAN_LABELS[:size])
    if axis.get("type") == "voigt":
        return list(VOIGT_LABELS)
    return [str(index) for index in range(size)]


def _coordinate_label(axis: Axis, coordinate: int) -> str:
    return CARTESIAN_LABELS[coordinate] if axis.get("type") == "cartesian" else str(coordinate)


def _prefix_label(prefixes: Sequence[Tuple[int, ...]], axes: Sequence[Axis]) -> str:
    """Format one or more equal tensor blocks compactly."""
    if len(prefixes) == 1:
        return ", ".join(
            f"{axis.get('name', f'axis_{index}')}={_coordinate_label(axis, value)}"
            for index, (axis, value) in enumerate(zip(axes, prefixes[0]))
        )
    varying = [
        index for index in range(len(axes)) if len({prefix[index] for prefix in prefixes}) > 1
    ]
    if len(varying) == 1 and axes[varying[0]].get("type") == "atomic":
        varying_index = varying[0]
        values = ", ".join(str(prefix[varying_index]) for prefix in prefixes)
        fixed = [
            f"{axis.get('name', f'axis_{index}')}={_coordinate_label(axis, prefixes[0][index])}"
            for index, axis in enumerate(axes)
            if index != varying_index
        ]
        return ", ".join([f"{axes[varying_index].get('name', 'atom')}={{{values}}}", *fixed])
    indices = "; ".join(
        "(" + ", ".join(_coordinate_label(axis, value) for axis, value in zip(axes, prefix)) + ")"
        for prefix in prefixes
    )
    return f"indices={{{indices}}}"


def print_components(
    components: ArrayLike, axes: Sequence[Axis], title: Optional[str] = None
) -> None:
    """Print numeric or symbolic tensor components using their axis definitions."""
    values = np.asarray(components, dtype=object)
    if values.ndim != len(axes):
        raise ValueError("The component array dimensions must match the tensor axes.")
    if title:
        print(title)
    if values.ndim == 0:
        print(f"  {values.item()}")
        return
    if values.ndim == 1:
        labels = _axis_labels(axes[0], values.shape[0])
        width = max(
            1,
            *(len(str(value)) for value in values.flat),
            *(len(label) for label in labels),
        )
        print("    " + "  ".join(f"{label:>{width}}" for label in labels))
        print("  [ " + "  ".join(f"{str(value):>{width}}" for value in values) + " ]")
        return

    row_labels = _axis_labels(axes[-2], values.shape[-2])
    column_labels = _axis_labels(axes[-1], values.shape[-1])
    column_width = max(
        1,
        *(len(str(value)) for value in values.flat),
        *(len(label) for label in column_labels),
    )
    row_width = max(len(label) for label in row_labels)
    row_prefix = "  " + " " * row_width + "  "
    prefixes = list(product(*(range(size) for size in values.shape[:-2])))
    prefix_axes = axes[: values.ndim - 2]
    groups = {}
    for prefix in prefixes:
        key = tuple(str(value) for value in values[prefix].flat)
        groups.setdefault(key, []).append(prefix)
    prefix_groups = (
        groups.values()
        if any(axis.get("type") == "atomic" for axis in prefix_axes)
        else ([prefix] for prefix in prefixes)
    )

    for group in prefix_groups:
        group = list(group)
        if prefix_axes:
            print(f"  [{_prefix_label(group, prefix_axes)}]")
        block = values[group[0]]
        print(row_prefix + "  ".join(f"{label:>{column_width}}" for label in column_labels))
        for label, row in zip(row_labels, block):
            print(
                f"  {label:>{row_width}}  "
                + "  ".join(f"{str(value):>{column_width}}" for value in row)
            )


def _component_label(index: int, shape: Tuple[int, ...], axes: Sequence[Axis]) -> str:
    coordinates = np.unravel_index(index, shape)
    labels = []
    for axis, coordinate in zip(axes, coordinates):
        value = _coordinate_label(axis, coordinate)
        labels.append(f"{axis.get('name', 'axis')}={value}")
    return ", ".join(labels)


def print_independent_components(
    pivots: Sequence[int], shape: Tuple[int, ...], axes: Sequence[Axis]
) -> None:
    """Print tensor entries selected as independent parameters."""
    if not pivots:
        return
    print("\nSymmetry-inequivalent components:")
    for index, pivot in enumerate(pivots):
        print(f"  {parameter_name(index)} = {_component_label(pivot, shape, axes)}")


def print_numeric_tensor(
    tensor: Any,
    keyword: str,
    location: str,
    pivots: Sequence[int],
    symbolic: ArrayLike,
    *,
    frame_label: str,
    atol: float = ATOL,
    parameter_values: Optional[ArrayLike] = None,
    precision: Optional[int] = None,
) -> None:
    """Print a numeric tensor, its independent values, and its symmetry-zero check."""
    print(
        f"\nNumeric tensor from {location}[{keyword!r}] ({tensor.basis} basis, {frame_label} axes):"
    )
    tensor.print_components(format_numeric_components(tensor.data, precision))

    print("\nSymmetry-inequivalent component values:")
    flat = tensor.data.reshape(-1)
    if pivots:
        values = flat[pivots] if parameter_values is None else np.asarray(parameter_values)
        if values.shape != (len(pivots),):
            raise ValueError("There must be one value per symmetry-inequivalent component.")
        formatted = format_numeric_components(values, precision)
        for index, value in enumerate(formatted):
            print(f"  {parameter_name(index)}: {value}")
    else:
        print("  none")

    violations, forbidden_count = forbidden_component_indices(tensor.data, symbolic, atol=atol)
    if not forbidden_count:
        print("\nZero-component check: no components are constrained to zero by symmetry.")
        return
    if not len(violations):
        print(
            f"\nZero-component check: PASS; all {forbidden_count} symmetry-forbidden "
            f"components are zero within absolute tolerance {atol:.1e}."
        )
        return

    maximum = np.max(np.abs(flat[violations]))
    print(
        f"\nZero-component check: FAIL; {len(violations)} of {forbidden_count} "
        f"symmetry-forbidden components exceed absolute tolerance {atol:.1e}."
    )
    raise ValueError(
        "Symmetry-forbidden tensor components are non-zero; "
        f"maximum absolute value is {maximum:.6e}."
    )
