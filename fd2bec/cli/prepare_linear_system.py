import argparse
import json

import numpy as np
from ase.io import read

from fd2bec import SYMPREC
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli, str2bool

description = (
    "Prepare the file to solve the linear system to get Born Effective Charges."
)


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
    # parser.add_argument("-sc", "--super_cell"             , **argv, type=str     , required=True , help="path to unit super structure (e.g. supercell.extxyz)")
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
    # parser.add_argument("-asr"     , "--acoustic_sum_rule", **argv, type=float   , required=False, help="weight for the acoustic sum rule, -1: not used, positive number otherwise (default: %(default)s)", default=-1)
    parser.add_argument(
        "-is_delta",
        "--is_delta_dipole",
        **argv,
        type=str2bool,
        required=False,
        help="wheter the coefficients are delta dipole (default: %(default)s)",
        default=False,
    )
    # parser.add_argument("-tran", "--translations"         , **argv, type=str2bool, required=False, help="apply translational symmetries (default: %(default)s)", default=True)
    # parser.add_argument("-spg", "--space_group"           , **argv, type=str2bool, required=False, help="apply space group symmetries (default: %(default)s)", default=False)
    parser.add_argument(
        "-s",
        "--symprec",
        **argv,
        type=float,
        required=False,
        help="symmetry precision for spglib (default: %(default)s)",
        default=SYMPREC,
    )
    parser.add_argument(
        "-o", "--output", **argv, type=str, required=True, help="JSON output file"
    )
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

    # # supercell
    # print(f"Reading unit cell structure from {args.super_cell} ... ",end="")
    # super_cell = read(args.super_cell, index=0)
    # print("done")
    # Nas = super_cell.get_global_number_of_atoms()
    # print(f"Number of atoms in the super cell: {Nas}")
    # if Nas % Na != 0:
    #     raise ValueError(f"The number of atoms in the super cell ({Nas}) must be a multiple of the number of atoms in the unit cell ({Na}).")
    # if Nas < Na:
    #     raise ValueError(f"The number of atoms in the super cell ({Nas}) must be greater than or equal to the number of atoms in the unit cell ({Na}).")
    # elif Nas > Na:
    #     use_supercell = True
    # else:
    #     use_supercell = False
    # super_cell = AtomicStructure.from_ase(super_cell)

    spg_uc = unit_cell.to_spglib_cell(symprec=args.symprec)
    # spg_sc = super_cell.to_spglib_cell(symprec=args.symprec)

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
    # Sanity checks
    # ----------------------#

    # if A.shape[1] > n_unknowns:
    #     # if not use_supercell:
    #     raise ValueError(f"The number of columns in A ({A.shape[1]}) must be less than or equal to the number of unknowns ({n_unknowns}). If you want to use a supercell, please provide a super cell structure with more atoms than the unit cell.")
    #     # print("It seems that you are using a supercell.")

    # if use_supercell:
    #     print("Using translations symmetries for supercells.")
    #     args.translations = True

    # ----------------------#
    # Fractional coordinates
    # ----------------------#
    # assert A.shape[1] == len(unit_cell)*3, "error"
    nr = len(A)
    A = unit_cell.to_fractional(A.reshape((nr, len(unit_cell), 3)), rank=1).reshape(
        (nr, -1)
    )
    b = unit_cell.to_fractional(b, rank=1)
    pass

    # ----------------------#
    # Symmetries
    # ----------------------#

    x = np.zeros((Na * 3, 3), dtype=object)

    # if args.translations:

    #     for i in range(Na):
    #         for k,kstr in enumerate(["x","y","z"]):
    #             for j,jstr in enumerate(["x","y","z"]):
    #                 x[i*3+k,j] = f"d mu_{jstr} / d R^{i}_{kstr}"
    #     print("x.shape:", x.shape)

    #     rot_uc = np.unique(spg_uc.rotations,axis=0)
    #     rot_sc = np.unique(spg_sc.rotations,axis=0)
    #     assert np.allclose(rot_uc,rot_sc), \
    #         "The unit cell and the super cell must have the same space group symmetries but they have different rotation matrices."

    #     assert spg_uc.number == spg_sc.number, \
    #         f"The unit cell and the super cell must have the same space group but they have space groups {spg_uc.number} and {spg_sc.number}."

    #     mapping = spg_sc.mapping_to_primitive
    #     s = set(mapping)
    #     if not len(s) == Na:
    #         raise ValueError(f"The mapping from the super cell to the primitive cell must have {Na} unique values but has {len(s)}." +
    #                          "Maybe you are using a symmetric supercell. Consider using a unitcell")
    #     print(f"Mapping from super cell to primitive cell: {mapping}")

    #     map2sc = invert_mapping_to_list(mapping)
    #     print(f"Mapping from super cell to primitive cell:")
    #     for p, sc_list in enumerate(map2sc):
    #         print(f" - Primitive atom {p} corresponds to super cell atoms {sc_list}")
    #     supercell_size = len(map2sc[0])
    #     assert all(len(sc_list) == supercell_size for sc_list in map2sc), \
    #         "All primitive atoms must correspond to the same number of super cell atoms"
    #     assert supercell_size == Nas // Na, \
    #         f"The number of super cell atoms corresponding to each primitive atom ({supercell_size}) must be equal to the ratio of the number of atoms in the super cell and the unit cell ({Nas // Na})."

    #     map2sc = np.asarray(map2sc)
    #     translational_symmetries = np.zeros((Nas,Na),dtype=int)
    #     for col, sc_list in enumerate(map2sc):
    #         for row in sc_list:
    #             translational_symmetries[row,col] = 1

    #     assert np.allclose(mapping,translational_symmetries @ np.arange(Na)), \
    #         "The mapping from the super cell to the primitive cell must be consistent with the translational symmetries."

    #     translational_symmetries = np.kron(translational_symmetries,np.eye(3)) # / supercell_size

    S = None
    theta = None
    # if args.space_group:
    S, theta, theta_real, shape = unit_cell.get_symmetrizer(
        rank=2, atomic=True, affine=False
    )
    n_unknowns = len(theta)
    # S = S.reshape((-1,3,n_unknowns))
    x = np.asarray([f"theta_{n}" for n in range(len(theta))])

    # ----------------------#
    # Linear system
    # ----------------------#

    b_coeff = b.copy()
    A_coeff = A.copy()

    # if args.acoustic_sum_rule >= 0:
    #     b_coeff = np.vstack([np.zeros((3,3)), b_coeff])
    #     id = np.eye(3)
    #     A_coeff = np.vstack([args.acoustic_sum_rule*np.tile(id, A_coeff.shape[1]//3), A_coeff])

    # if args.space_group:

    b_coeff = b_coeff.flatten()
    A_coeff = np.kron(A_coeff, np.eye(3))
    A_coeff = A_coeff @ S  # np.einsum("ij,jkl->ikl",A_coeff,S)
    # A_coeff = A_coeff.reshape((-1,n_unknowns))

    min_displ = A_coeff.shape[1]

    if not args.is_delta_dipole:
        # ToDo
        # this could be improved by using the space group
        # to understand which are the independent components of the dipole
        x = np.concat(
            [np.asarray(["mu_x", "mu_y", "mu_z"], dtype=object), x.astype(object)]
        )
        nr = int(len(b_coeff) / 3)
        tmp = np.tile(-np.eye(3), nr).T
        A_coeff = np.hstack([tmp, A_coeff])
        # S = np.vstack([np.full((1,S.shape[1],S.shape[2]),-1),S])

        min_displ += 1

    # elif args.translations:
    #     A_coeff = A_coeff @ translational_symmetries
    #     x = np.zeros((A_coeff.shape[1],3),dtype=object)

    #     min_displ = A_coeff.shape[1]

    #     if not args.is_delta_dipole:
    #         x = np.vstack([x,np.asarray(["mu_x","mu_y","mu_z"],dtype=object)])
    #         A_coeff = np.hstack([A_coeff,np.full((A_coeff.shape[0],1),-1)])

    #         min_displ += 1

    assert b_coeff.shape[0] == A_coeff.shape[0], (
        f"'b_coeff' and 'A_coeff' must have the same number of rows but they have shapes {b_coeff.shape} and {A_coeff.shape}"
    )

    # if not args.space_group:
    #     assert A_coeff.shape[1] == x.shape[0], \
    #         f"The number of columns in A ({A.shape[1]}) must be equal to the number of unknowns ({x.shape[0]})."

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
        # "asr_weight" : float(args.acoustic_sum_rule),
        # "apply_translations" : args.translations,
        # "apply_space_group" : args.space_group,
        "is_delta_dipole": args.is_delta_dipole,
        "unitcell": unit_cell.to_json(),
        # "supercell" : super_cell.to_json(),
        # "input" : {
        #     "b" : b.tolist(),
        #     "A" : A.tolist(),
        # },
        "symmetry": {
            "space_group_number": spg_uc.number,
            # "transformation_matrix" : spg_sc.transformation_matrix.round(4).tolist(),
            # "translational_symmetries" : translational_symmetries.tolist() if args.translations else None,
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
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)
    print("done")


if __name__ == "__main__":
    main()
