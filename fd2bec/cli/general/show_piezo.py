import argparse

import numpy as np

from fd2bec import float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.cli.dPdS.dPdS2piezo import print_voigt_tensor
from fd2bec.io import read
from fd2bec.mathematics import rotate_rank3
from fd2bec.piezoelectric import (
    piezoelectric_symbolic_matrix,
    piezoelectric_to_voigt,
    proper_piezoelectric_symmetry_basis,
)
from fd2bec.show import print_reference_structure

description = "Extract BEC from a extxyz file and convert it to a txt file."


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
        required=False,
        help="name of the piezoelectric tensor (default: %(default)s)",
        default="piezoelectric",
    )
    parser.add_argument(
        "--conventional_axes",
        action="store_true",
        help=(
            "rotate reported and saved Cartesian tensors into spglib's "
            "conventional crystallographic axes"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to txt output file (e.g. piezoelectric.txt)",
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    reference = read(args.input, index=0)
    print("done")

    print_reference_structure(reference)

    unit_cell = AtomicStructure.from_ase(reference)
    direct_basis = proper_piezoelectric_symmetry_basis(unit_cell)
    dataset = unit_cell._spglib_dataset  # pylint: disable=protected-access
    space_group_symbol = dataset.international
    if isinstance(space_group_symbol, bytes):
        space_group_symbol = space_group_symbol.decode()
    coordinate_rotation = (
        np.asarray(dataset.std_rotation_matrix, dtype=float)
        if args.conventional_axes
        else np.eye(3)
    )
    reported_basis = (
        np.column_stack(
            [
                np.einsum(
                    "ai,bj,ck,ijk->abc",
                    coordinate_rotation,
                    coordinate_rotation,
                    coordinate_rotation,
                    mode.reshape((3, 3, 3)),
                ).reshape(-1)
                for mode in direct_basis.T
            ]
        )
        if direct_basis.shape[1]
        else direct_basis.copy()
    )

    if args.conventional_axes:
        print("Rotating reported and saved tensors into conventional crystallographic axes.")
        print("Cartesian coordinate rotation (conventional <- input):")
        print(np.array2string(coordinate_rotation, precision=8, suppress_small=True))

    print("Voigt order: xx, yy, zz, yz, xz, xy.")
    print("Engineering strain uses [exx, eyy, ezz, 2eyz, 2exz, 2exy].")
    print("\nSymmetry-allowed proper piezoelectric matrix [3x6]:")
    frame_label = "conventional" if args.conventional_axes else "input"
    print(f"(letters are independent parameters in the {frame_label} Cartesian axes)")
    symbolic_matrix = piezoelectric_symbolic_matrix(reported_basis)
    symbolic_width = max(1, max(len(value) for value in symbolic_matrix.flat))
    for row in symbolic_matrix:
        print("[ " + "  ".join(f"{value:>{symbolic_width}}" for value in row) + " ]")

    print(f"Extracting '{args.name}' from the structure ... ", end="")
    infos = reference.info
    if args.name not in infos:
        raise ValueError(
            f"'{args.name}' not found in the 'info' of the structure\n"
            f"Available keys: {list(infos.keys())}"
        )
    piezo = reference.info[args.name]
    print("done")
    print("piezo.shape:", piezo.shape)

    if args.conventional_axes:
        print("Rotating piezoelectric tensor ... ", end="")
        piezo = rotate_rank3(piezo, coordinate_rotation)
        print("done")

    print("Converting to Voigt notation ... ", end="")
    piezo = piezoelectric_to_voigt(piezo)
    print("done")
    print("piezo.shape:", piezo.shape)

    print("\nPiezoelectric Tensor [e/Angstrom^2]:")
    print_voigt_tensor(piezo)

    print(f"Saving piezoelectric tensor to file {args.output} ... ", end="")
    np.savetxt(args.output, piezo, fmt=float_format)
    print("done")

    pass


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
