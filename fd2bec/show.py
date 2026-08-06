import numpy as np
from ase import Atoms


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
