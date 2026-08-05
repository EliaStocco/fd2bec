import argparse
from typing import Tuple
from warnings import warn

import numpy as np
from ase import Atoms

from fd2bec import float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read, write
from fd2bec.tensor import Tensor

description = "Generate Cartesian atomic or cell displacements and displaced structures."
CELL_COMPONENTS = (
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
)


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        required=False,
        help="Cartesian displacement amplitude in Angstrom (default: %(default)s)",
        default=1e-3,
    )
    parser.add_argument(
        "-w",
        "--what",
        **argv,
        type=str,
        required=False,
        help="target quantity (default: %(default)s)",
        default="bec",
        choices=["bec", "piezo"],
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--no-symmetry",
        action="store_true",
        help="disable symmetry reduction and use every Cartesian displacement",
    )
    selection.add_argument(
        "-n",
        "--number",
        **argv,
        type=int,
        help="number of normally distributed random displacements",
    )
    parser.add_argument(
        "--seed",
        **argv,
        type=int,
        help="random seed used with --number",
    )
    parser.add_argument(
        "-d",
        "--displacements-output",
        "--displacements",
        **argv,
        type=str,
        required=False,
        help="optional path to a flattened txt displacement table",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to the multi-frame extxyz output",
    )
    return parser


def tensor2perturbation_shape(tensor: Tensor) -> Tuple[int, ...]:
    """Return the atomic/covariant shape controlled by a tensor derivative."""
    if tensor.data is None:
        raise ValueError("A tensor template with explicit data is required.")

    shape = (tensor.data.shape[0],) if tensor.is_atomic else ()
    tensor_offset = 1 if tensor.is_atomic else 0
    shape += tuple(
        tensor.data.shape[tensor_offset + axis]
        for axis, is_covariant in enumerate(tensor.axes)
        if is_covariant
    )
    return shape


def all_cartesian_displacements(number_of_components: int) -> np.ndarray:
    """Return the reference and positive/negative Cartesian basis directions."""
    if number_of_components <= 0:
        raise ValueError("The number of displacement components must be positive.")

    directions = np.eye(number_of_components)
    return np.concatenate([np.zeros((1, number_of_components)), directions, -directions], axis=0)


def atomic_structure2all_displacements(unit_cell: AtomicStructure, amplitude: float) -> np.ndarray:
    """Return all signed Cartesian atomic displacements and the reference."""
    if amplitude <= 0:
        raise ValueError("The displacement amplitude must be positive.")
    return all_cartesian_displacements(3 * len(unit_cell)) * amplitude


def cell_components2displacements(components: np.ndarray) -> np.ndarray:
    """Expand six components into flattened lower-triangular 3x3 matrices."""
    components = np.asarray(components, dtype=float)
    if components.ndim != 2 or components.shape[1] != len(CELL_COMPONENTS):
        raise ValueError("Cell components must have shape (N, 6).")

    displacements = np.zeros((len(components), 3, 3))
    for column, (row, component) in enumerate(CELL_COMPONENTS):
        displacements[:, row, component] = components[:, column]
    return displacements.reshape((len(components), 9))


def project_cell_displacements(displacements: np.ndarray) -> np.ndarray:
    """Project flattened cell displacements onto six lower-triangular components."""
    matrices = np.asarray(displacements, dtype=float).reshape((-1, 3, 3))
    matrices = np.tril(matrices)
    return matrices.reshape((-1, 9))


def all_cell_displacements() -> np.ndarray:
    """Return the reference and positive/negative basis for six cell components."""
    directions = cell_components2displacements(np.eye(len(CELL_COMPONENTS)))
    return np.concatenate([np.zeros((1, 9)), directions, -directions], axis=0)


def random_cartesian_displacements(
    number: int, number_of_components: int, atomic: bool, seed: int = None
) -> np.ndarray:
    """Generate normally distributed atomic or lower-triangular cell displacements."""
    if number <= 0:
        raise ValueError("The number of random displacements must be positive.")

    generator = np.random.default_rng(seed)
    if atomic:
        return generator.normal(size=(number, number_of_components))
    if number_of_components != 9:
        raise ValueError("Cell displacements must have nine flattened components.")
    components = generator.normal(size=(number, len(CELL_COMPONENTS)))
    return cell_components2displacements(components)


def atomic_structure2unique_displacements(
    unit_cell: AtomicStructure, tensor: Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate normalized symmetry-inequivalent perturbation directions.

    For atomic tensors, the directions are Cartesian atomic displacements.
    For global tensors, they span the tensor's covariant input indices; for
    example, an improper piezoelectric tensor produces flattened strain
    directions with nine components.
    """
    _, theta, theta_real = unit_cell.get_symmetrizer(tensor=tensor)
    tensor_axis_offset = 2 if tensor.is_atomic else 1
    output_axes = tuple(
        tensor_axis_offset + axis
        for axis, is_covariant in enumerate(tensor.axes)
        if not is_covariant
    )
    input_shape = tensor2perturbation_shape(tensor)
    number_of_components = int(np.prod(input_shape))

    if len(theta_real) == 0:
        warn("The provided tensor has no symmetry-allowed components.")
        directions = np.empty((0, number_of_components))
    else:
        modes = theta_real.reshape((-1, *tensor.data.shape))
        active_components = np.abs(modes) > 1e-10
        if output_axes:
            directions = np.sum(active_components, axis=output_axes, dtype=float)
        else:
            directions = active_components.astype(float)
        directions = directions.reshape((len(theta), -1))
        if not tensor.is_atomic and input_shape == (3, 3):
            directions = project_cell_displacements(directions)

        norms = np.linalg.norm(directions, axis=1)
        directions = directions[norms > 1e-10]
        directions /= norms[norms > 1e-10, None]

    displacements = np.concatenate(
        [np.zeros((1, number_of_components)), directions, -directions], axis=0
    )

    _, first_indices = np.unique(displacements, axis=0, return_index=True)
    u = displacements[np.sort(first_indices)]

    return u, displacements


def proper_piezoelectric_cell_displacements(unit_cell: AtomicStructure, tensor):
    """Select lower-triangular cell directions spanning all allowed proper modes."""
    symmetrizer, _, _ = unit_cell.get_symmetrizer(tensor=tensor)
    modes = symmetrizer.reshape((3, 3, 3, -1))
    modes = 0.5 * (modes + modes.swapaxes(1, 2))
    modes = modes.reshape((27, -1))
    if modes.shape[1]:
        left, singular_values, _ = np.linalg.svd(modes, full_matrices=False)
        threshold = 1e-10 * max(modes.shape) * singular_values[0]
        modes = left[:, singular_values > threshold]

    candidates = cell_components2displacements(np.eye(len(CELL_COMPONENTS)))
    inverse_cell = np.linalg.inv(np.asarray(unit_cell.cell))
    selected = []
    response = np.empty((0, modes.shape[1]))
    rank = 0
    for candidate in candidates:
        displacement = candidate.reshape((3, 3))
        gradient = (inverse_cell @ displacement).T
        strain = 0.5 * (gradient + gradient.T)
        block = np.zeros((3, 27))
        for component in range(3):
            block[component, 9 * component : 9 * component + 9] = strain.reshape(9)
        trial = np.vstack((response, block @ modes))
        trial_rank = np.linalg.matrix_rank(trial, tol=1e-10)
        if trial_rank > rank:
            selected.append(candidate)
            response = trial
            rank = trial_rank
        if rank == modes.shape[1]:
            break
    if rank != modes.shape[1]:
        raise ValueError(
            "The six lower-triangular cell components do not span all "
            f"symmetry-allowed proper piezoelectric modes ({rank}/{modes.shape[1]})."
        )

    directions = np.asarray(selected).reshape((-1, 9))
    zero = np.zeros((1, 9))
    chosen = np.concatenate((zero, directions, -directions), axis=0)
    all_candidates = np.concatenate((zero, candidates, -candidates), axis=0)
    return chosen, all_candidates


def displacements2structures(atoms: Atoms, displacements: np.ndarray, atomic: bool) -> list[Atoms]:
    """Apply flattened atomic or cell displacements to copies of ``atoms``."""
    displacements = np.asarray(displacements, dtype=float)
    if displacements.ndim != 2:
        raise ValueError("Displacements must be a two-dimensional array.")

    expected = 3 * len(atoms) if atomic else 9
    if displacements.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} displacement components, got {displacements.shape[1]}."
        )
    if not atomic and not np.all(atoms.get_pbc()):
        raise ValueError("Cell displacements require a fully periodic structure.")

    structures = []
    reference_cell = atoms.cell.array
    reference_handedness = np.sign(np.linalg.det(reference_cell))
    for displacement in displacements:
        displaced = atoms.copy()
        if atomic:
            atomic_displacement = displacement.reshape((len(atoms), 3))
            displaced.set_positions(atoms.get_positions() + atomic_displacement)
            displaced.set_array("displacements", atomic_displacement.copy())
        else:
            cell_displacement = displacement.reshape((3, 3))
            displaced_cell = reference_cell + cell_displacement
            if np.sign(np.linalg.det(displaced_cell)) != reference_handedness:
                raise ValueError("A cell displacement produced a singular or inverted cell.")
            displaced.set_cell(displaced_cell, scale_atoms=True)
            displaced.info["cell_displacement"] = cell_displacement.copy()
        structures.append(displaced)

    return structures


def print_input_structure(atoms: Atoms) -> None:
    """Print a compact representation of the input cell and fractional positions."""
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


def print_symmetry_selection(
    unit_cell: AtomicStructure, displacements: np.ndarray, atomic: bool
) -> None:
    """Print concise space-group and selected-displacement information."""
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


@cli(prepare_args, description)
def main(args):
    """Generate and save the selected displaced structures."""
    if args.amplitude <= 0:
        raise ValueError("The displacement amplitude must be positive.")
    if args.number is not None and args.number <= 0:
        raise ValueError("The number of random displacements must be positive.")
    if args.seed is not None and args.number is None:
        raise ValueError("--seed can only be used together with --number.")

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")
    print_input_structure(atoms)

    unit_cell = AtomicStructure.from_ase(atoms)
    number_of_atoms = len(unit_cell)

    if args.what == "bec":
        print("Constructing Born Effective Charges ... ", end="")
        from fd2bec.tensor import BornCharges

        tensor = BornCharges(data=np.zeros((number_of_atoms, 3, 3)))
        print("done")
    elif args.what == "piezo":
        from fd2bec.tensor import ProperPiezoelectricTensor

        print("Constructing proper piezoelectric tensor ... ", end="")
        tensor = ProperPiezoelectricTensor.template()
        print("done")

    number_of_components = int(np.prod(tensor2perturbation_shape(tensor)))
    if args.number is not None:
        selected = random_cartesian_displacements(
            number=args.number,
            number_of_components=number_of_components,
            atomic=tensor.is_atomic,
            seed=args.seed,
        )
        print(
            f"Generated {len(selected)} normally distributed random "
            f"{'atomic' if tensor.is_atomic else 'lower-triangular cell'} displacements."
        )
    elif args.no_symmetry:
        if tensor.is_atomic:
            selected = all_cartesian_displacements(number_of_components)
        else:
            selected = all_cell_displacements()
        candidates = selected
        print(
            f"Symmetry disabled: selected all {len(selected) - 1} signed Cartesian "
            f"basis displacements; {len(selected)} structures including the reference."
        )
    else:
        if args.what == "piezo":
            selected, candidates = proper_piezoelectric_cell_displacements(unit_cell, tensor)
        else:
            selected, candidates = atomic_structure2unique_displacements(unit_cell, tensor=tensor)

    selected = selected * args.amplitude

    if args.number is None and not args.no_symmetry:
        print(
            f"Found {len(selected) - 1} unique signed displacements from "
            f"{len(candidates)} symmetry-mode candidates."
        )
        print_symmetry_selection(unit_cell, selected, atomic=tensor.is_atomic)

    structures = displacements2structures(atoms, selected, atomic=tensor.is_atomic)

    if args.displacements_output is not None:
        print(f"Writing displacements to {args.displacements_output} ... ", end="")
        np.savetxt(args.displacements_output, selected, fmt=float_format)
        print("done")

    print(f"Writing {len(structures)} displaced structures to {args.output} ... ", end="")
    write(args.output, structures, format="extxyz")
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
