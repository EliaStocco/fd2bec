import numpy as np
from ase import Atoms


def print_displacement_input_structure(atoms: Atoms) -> None:
    """Print the periodic input structure used for finite displacements."""
    if not np.all(atoms.get_pbc()):
        raise ValueError("This command requires a fully periodic input structure.")

    print("Input cell [Angstrom]:")
    print(np.array2string(atoms.cell.array, precision=8, suppress_small=True))
    print("Fractional coordinates:")
    for index, (symbol, position) in enumerate(
        zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=False))
    ):
        coordinates = " ".join(f"{value: .8f}" for value in position)
        print(f"  {index:4d} {symbol:>2s}  {coordinates}")


def print_symmetry_selection(unit_cell, displacements: np.ndarray, atomic: bool) -> None:
    """Print space-group and selected finite-displacement information."""
    dataset = unit_cell._spglib_dataset  # pylint: disable=protected-access
    symbol = getattr(dataset, "international", "unknown")
    if isinstance(symbol, bytes):
        symbol = symbol.decode()
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


def print_reference_structure(reference: Atoms):
    """Print the reference cell, lattice parameters, and fractional positions."""
    cell = np.asarray(reference.cell.array, dtype=float)
    a, b, c, alpha, beta, gamma = reference.cell.cellpar()
    scaled_positions = reference.get_scaled_positions(wrap=False)

    print("-" * 20)
    print("Reference structure:")
    print("\nCell vectors [Angstrom]:")
    for i, vec in enumerate(cell, start=1):
        formatted_vec = "  ".join(f"{val:14.8f}" for val in vec)
        print(f"\ta{i}:  {formatted_vec}")

    print(
        "\nLattice parameters:\n"
        f"\ta,b,c (Å) = {a:14.8f} {b:14.8f} {c:14.8f}\n"
        f"\tα,β,γ (°) = {alpha:14.8f} {beta:14.8f} {gamma:14.8f}"
    )
    print("\nFractional coordinates:")
    print("\tatom              x              y              z")
    for symbol, position in zip(reference.get_chemical_symbols(), scaled_positions):
        print(f"\t{symbol:<4s}  {position[0]:14.8f}  {position[1]:14.8f}  {position[2]:14.8f}")
    print("-" * 20)
    print()
