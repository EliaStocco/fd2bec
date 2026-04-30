import argparse
import json

import numpy as np
from ase.io import read

from fd2bec import SYMPREC
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli, str2bool

description = "Prepare the file to solve the linear system to get Born Effective Charges."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-uc",
        "--unit_cell",
        **argv,
        type=str,
        required=True,
        help="path to unit cell structure (e.g. unitell.extxyz)",
    )
    parser.add_argument(
        "-b",
        "--coefficients",
        **argv,
        type=str,
        required=False,
        help="path to coefficients (e.g. dipole.txt, default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "-A",
        "--matrix",
        **argv,
        type=str,
        required=False,
        help="path to displacement matrix (e.g. displacement.txt, efault: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "-is_delta",
        "--is_delta_dipole",
        **argv,
        type=str2bool,
        required=False,
        help="wheter the coefficients are delta dipole (default: %(default)s)",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--symprec",
        **argv,
        type=float,
        required=False,
        help="symmetry precision for spglib (default: %(default)s)",
        default=SYMPREC,
    )
    parser.add_argument("-o", "--output", **argv, type=str, required=True, help="JSON output file")
    return parser


@cli(prepare_args, description)
def main(args):

    # if args.space_group:
    # assert args.acoustic_sum_rule == -1, "ASR and space groups are incompatible."

    min_displ = -1

    # ----------------------#
    # Structures
    # ----------------------#

    # unitcell
    print(f"Reading unit cell structure from {args.unit_cell} ... ", end="")
    unit_cell = read(args.unit_cell, index=0)
    print("done")
    Na = unit_cell.get_global_number_of_atoms()
    n_unknowns = Na * 9
    print(f"Number of atoms in the unit cell: {Na}")
    unit_cell = AtomicStructure.from_ase(unit_cell)

    spg_uc = unit_cell.to_spglib_cell(symprec=args.symprec)

    # ----------------------#
    # Coefficients
    # ----------------------#

    # b
    if args.coefficients is not None:
        print(f"Reading the coefficients b from file {args.coefficients} ... ", end="")
        b = np.loadtxt(args.coefficients)
        print("done")
        print("b.shape:", b.shape)
        assert b.shape[1] == 3, f"'b' must have 3 columns but it has shape {b.shape}"
    else:
        b = np.zeros((Na * 3, 3))

    # A
    if args.matrix is not None:
        print(f"Reading the matrix A from file {args.matrix} ... ", end="")
        A = np.loadtxt(args.matrix)
        print("done")
        print("A.shape:", A.shape)
    else:
        # A = np.zeros((Na*3,len(super_cell)*3))
        # assert A.shape[0] == A.shape[1], "error"
        A = np.eye(Na * 3)

    # ----------------------#
    # Fractional coordinates
    # ----------------------#
    # assert A.shape[1] == len(unit_cell)*3, "error"
    nr = len(A)
    A = unit_cell.to_fractional(A.reshape((nr, len(unit_cell), 3)), rank=1).reshape((nr, -1))
    b = unit_cell.to_fractional(b, rank=1)

    # ----------------------#
    # Symmetries
    # ----------------------#

    x = np.zeros((Na * 3, 3), dtype=object)

    S = None
    theta = None
    # if args.space_group:
    S, theta, _ = unit_cell.get_symmetrizer(rank=2, atomic=True, affine=False)
    n_unknowns = len(theta)
    # S = S.reshape((-1,3,n_unknowns))
    x = np.asarray([f"theta_{n}" for n in range(len(theta))])

    # ----------------------#
    # Linear system
    # ----------------------#

    b_coeff = b.copy()
    A_coeff = A.copy()

    b_coeff = b_coeff.flatten()
    A_coeff = np.kron(A_coeff, np.eye(3))
    A_coeff = A_coeff @ S  # np.einsum("ij,jkl->ikl",A_coeff,S)
    # A_coeff = A_coeff.reshape((-1,n_unknowns))

    min_displ = A_coeff.shape[1]

    if not args.is_delta_dipole:
        # this could be improved by using the space group
        # to understand which are the independent components of the dipole
        x = np.concatenate([np.asarray(["mu_x", "mu_y", "mu_z"], dtype=object), x.astype(object)])
        nr = int(len(b_coeff) / 3)
        tmp = np.tile(-np.eye(3), nr).T
        A_coeff = np.hstack([tmp, A_coeff])
        # S = np.vstack([np.full((1,S.shape[1],S.shape[2]),-1),S])

        min_displ += 1

    assert b_coeff.shape[0] == A_coeff.shape[0], (
        "'b_coeff' and 'A_coeff' must have the same number of rows"
        + f" but they have shapes {b_coeff.shape} and {A_coeff.shape}"
    )

    system_type = (
        "overdetermined"
        if A_coeff.shape[0] > x.shape[0]
        else "underdetermined"
        if A_coeff.shape[0] < x.shape[0]
        else "determined"
    )
    print(f"System type: {system_type}")

    print("Minimum number of necessary configurations: ", x.shape[0])
    # rank = np.linalg.matrix_rank(A_coeff)
    print("Rank of displacement matrix: ", min_displ)

    # ----------------------#
    # Save data
    # ----------------------#

    data = {
        "is_delta_dipole": args.is_delta_dipole,
        "unitcell": unit_cell.to_json(),
        "symmetry": {
            "space_group_number": spg_uc.number,
            "symmetrizer": S.tolist(),
            "n_theta": len(theta),
        },
        "linear_system": {
            "type": system_type,
            "n_rows": b_coeff.shape[0],
            "n_cols": A_coeff.shape[1],
            "n_unknowns": n_unknowns,
            "x": x.tolist() if x is not None else None,
            "b": b_coeff.tolist(),
            "A": A_coeff.tolist(),
        },
    }

    print(f"Saving data to {args.output} ... ", end="")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
