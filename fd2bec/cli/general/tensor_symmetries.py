"""Show the symmetry-allowed components of a tensor."""

# Tested by pytest: tests/test_tensor_symmetries.py

import argparse

import numpy as np

from fd2bec import ATOL
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.show import print_reference_structure
from fd2bec.tensor import MAPPING
from fd2bec.tensor_components import print_independent_components, symbolic_components

description = "Show the symmetry-allowed components of a tensor."
choices = list(MAPPING.keys())


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure (e.g. supercell.extxyz)",
    )
    parser.add_argument(
        "-n",
        "--name",
        **argv,
        type=str,
        required=True,
        help=f"name of the tensor, choices: {choices}",
        choices=choices,
    )
    parser.add_argument(
        "--conventional-axes",
        action="store_true",
        help="rotate Cartesian tensor components into spglib's conventional axes",
    )
    return parser


def _rotated_modes(tensor, modes, coordinate_rotation):
    """Rotate each tensor mode into a new Cartesian coordinate frame."""
    if not len(modes):
        return modes.copy()
    return np.asarray(
        [tensor.copy_with(data=mode).rotate(coordinate_rotation).data for mode in modes]
    )


@cli(prepare_args, description)
def main(args):
    print(f"Reading input structure from {args.input} ... ", end="")
    reference = read(args.input, index=0)
    print("done")

    print_reference_structure(reference)
    unit_cell = AtomicStructure.from_ase(reference)
    tensor = MAPPING[args.name].template(len(unit_cell))
    shape = tensor.core_shape()
    print(f"Constructed {tensor.definition['name']} tensor with shape {shape}.")

    print("\nComputing symmetry-allowed components ... ", end="")
    _, _, theta_real = unit_cell.get_symmetrizer(tensor=tensor)
    print("done")
    modes = theta_real.reshape((theta_real.shape[0], *shape))
    if not np.allclose(np.linalg.norm(modes.reshape((len(modes), -1)), axis=1), 1, atol=ATOL):
        raise ValueError("Symmetry modes must be normalized.")
    print("n. symmetry-inequivalent component(s): ", len(modes))

    frame_label = "input"
    if args.conventional_axes:
        coordinate_rotation = np.asarray(
            unit_cell._spglib_dataset.std_rotation_matrix, dtype=float  # pylint: disable=protected-access
        )
        modes = _rotated_modes(tensor, modes, coordinate_rotation)
        frame_label = "conventional"
        print("Rotating Cartesian components into conventional crystallographic axes.")
        print("Cartesian coordinate rotation (conventional <- input):")
        print(np.array2string(coordinate_rotation, precision=8, suppress_small=True))

    symbolic, pivots = symbolic_components(modes, axes=tensor.axes)
    print_independent_components(pivots, shape, tensor.axes)
    tensor.print_components(
        symbolic,
        title=f"\nSymmetry-allowed tensor components ({frame_label} axes):",
        voigt=True,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
