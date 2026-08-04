import argparse
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
from ase import Atom, Atoms

from fd2bec import ATOL, SYMPREC, float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS, cli
from fd2bec.cli.tools import print_born_charges
from fd2bec.io import read
from fd2bec.linear_system import LinearSystem, StackedLinearSystem
from fd2bec.tensor import Forces
from fd2bec.tools import symmetrize_bec

description = (
    "Compute the Born Effective Charges as derivative of the forces w.r.t. applied electric field."
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
        help="path to extxyz file with all structures produced by 'build_dataset4dFdE' (e.g. structures.extxyz)",
    )
    parser.add_argument(
        "-sp",
        "--symprec",
        **argv,
        type=float,
        required=False,
        help="symmetry precision for spglib (default: %(default)s)",
        default=SYMPREC,
    )
    parser.add_argument(
        "-c",
        "--clean_forces",
        type=str,
        required=False,
        help="clean forces (default: %(default)s)",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--symmetrize",
        type=str,
        required=False,
        help="symmetrize forces, bec, both, or none (default: %(default)s)",
        default="none",
        choices=["forces", "bec", "none", "both"],
    )
    parser.add_argument(
        "-emin",
        "--efiled_min",
        type=float,
        required=False,
        help="minimum electric field (absolute value) to consider (default: %(default)s)",
        default=0,
    )
    parser.add_argument(
        "-emax",
        "--efiled_max",
        type=float,
        required=False,
        help="maximum electric field (absolute value) to consider (default: %(default)s)",
        default=np.inf,
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=False,
        help="folder for the output files (default: %(default)s)",
        default=".",
    )
    return parser


def symmetrize_forces(
    structures: List[Atoms], efield: np.ndarray, forces: np.ndarray
) -> np.ndarray:
    N = len(structures)
    broken_symmetry = [None] * N
    for n, (s, e) in enumerate(zip(structures, efield)):
        if not np.allclose(e, 0):
            atom = Atom("Og", position=100 * e / np.linalg.norm(e))
            s.append(atom)
        # pos = s.get_positions()
        # pos -= np.mean(pos,axis=0)
        # s.set_positions(pos)
        broken_symmetry[n] = s.copy()
    atomic_structures = [AtomicStructure.from_ase(s) for s in broken_symmetry]
    _forces = [None] * N
    for n, (s, e) in enumerate(zip(atomic_structures, efield)):
        s._test_symmetry()
        # f = forces[n]
        if not np.allclose(e, 0):
            f = np.vstack((forces[n], np.zeros(3)))
        else:
            f = forces[n]
        _forces[n] = Forces(data=f)
    for n in range(N):
        sym_forces = atomic_structures[n].symmetrize(tensor=_forces[n]).data
        if not np.allclose(efield[n], 0):
            _forces[n] = sym_forces[:-1, :]
        else:
            _forces[n] = sym_forces
    return np.asarray(_forces)


@cli(prepare_args, description)
def main(args):

    assert Path(args.input).suffix == ".extxyz", f"'{args.input}' must be an extxyz file."

    print(f"Reading input structures from '{args.input}' ... ", end="")
    structures: List[Atoms] = read(args.input, format="extxyz", index=":")
    print("done")

    Ns = len(structures)
    Na = structures[0].get_global_number_of_atoms()
    print("n. structures: ", Ns)
    print("n. atoms: ", Na)

    pos = np.asarray([atoms.get_positions() for atoms in structures])
    assert np.all([np.allclose(pos_i, pos[0]) for pos_i in pos]), (
        "You have provided different geometries."
    )

    key = KEYWORDS["efield"]
    print(f"Extracting electric field from '{key}' ... ", end="")
    efield = np.asarray([atoms.info[key] for atoms in structures])
    print("done")
    print("efield.shape: ", efield.shape)

    key = KEYWORDS["forces"]
    print(f"Extracting forces from '{key}' ... ", end="")
    forces = np.asarray([atoms.arrays[key] for atoms in structures])
    print("done")
    print("forces.shape: ", forces.shape)

    # --- filter electric fields and corresponding data ---
    print("Filtering electric fields ... ", end="")

    efield = np.asarray(efield)
    forces = np.asarray(forces)

    efield_norm = np.linalg.norm(efield, axis=1)

    mask = (efield_norm >= args.efiled_min) & (efield_norm <= args.efiled_max)

    if not np.any(mask):
        raise ValueError("No structures left after electric field filtering.")

    structures = [s for s, m in zip(structures, mask) if m]
    efield = efield[mask]
    forces = forces[mask]

    print(f"done ({mask.sum()} / {len(mask)})")

    Ns = len(structures)
    print("efield.shape: ", efield.shape)
    print("forces.shape: ", forces.shape)

    if not args.clean_forces:
        print("Not cleaning forces")
    else:
        print("Cleaning forces ... ", end="")
        forces -= np.mean(forces, axis=1, keepdims=True)
        print("done")

    if args.symmetrize in ["forces", "both"]:
        print("Symmetrizing forces ... ", end="")
        forces = symmetrize_forces(structures, efield, forces)
        print("done")
    else:
        print("Not symmetrizing forces")
        # mask = forces.flatten() != 0
        # relative = np.abs(
        #     (forces.flatten()[mask] - sym_forces.flatten()[mask])
        #     / forces.flatten()[mask]
        # ).mean()

    for n, f in enumerate(forces):
        sum_forces = np.mean(f, axis=0)
        if np.any(sum_forces > ATOL):
            msg = f"Structure {n} has non-zero mean force: {sum_forces.tolist()}"
            warn(msg)

    print("Preparing linear systems ... ", end="")
    all_ls: List[LinearSystem] = []
    ones = np.full((Ns, 1), 1.0)
    A = np.hstack((ones, efield))
    for n in range(Na):
        b = forces[:, n, :]
        all_ls.append(LinearSystem(A=A, b=b))
    print("done")

    print("Solving linear systems  ... ", end="")
    LS = StackedLinearSystem(all_ls)
    LS.solve()  # saves solutions in  all_ls[:].x
    print("done")

    LS.summary()

    bec = np.zeros((Na, 3, 3))
    print("Extracting Born Charges  ... ", end="")
    for n in range(Na):
        bec[n, :, :] = all_ls[n].x[1:, :]
    print("done")

    print_born_charges(structures[0], bec)

    if args.symmetrize in ["bec", "both"]:
        print("Symmetrizing Born Charges ... ", end="")
        bec = symmetrize_bec(structures[0], bec)
        print("done")
        print_born_charges(structures[0], bec)
    else:
        print("Not symmetrizing Born Charges")

    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)

    file = folder / "bec-no-asr.txt"
    print(f"Writing Born Charges to {file} ... ", end="")
    np.savetxt(file, bec.reshape((Na, 9)), fmt=float_format)
    print("done")

    file = folder / "asr.txt"
    print(f"Writing sum of all Born Charges to {file} ... ", end="")
    asr = bec.mean(axis=0)
    np.savetxt(file, asr, fmt=float_format)
    print("done")

    file = folder / "bec.txt"
    print(f"Writing sum of Born Charges with ASR applied to {file} ... ", end="")
    np.savetxt(file, bec.reshape((Na, 9)) - asr.reshape((1, 9)), fmt=float_format)
    print("done")

    return


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
