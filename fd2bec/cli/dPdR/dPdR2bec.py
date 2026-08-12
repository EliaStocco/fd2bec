# Tested by pytest: tests/test_aims_workflow_wrappers.py

import argparse
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
from ase import Atoms

from fd2bec import ATOL, float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS, cli, str2bool
from fd2bec.cli.tools import matrix_norm, print_born_charges
from fd2bec.io import read
from fd2bec.linear_system import LinearSystem
from fd2bec.tensor import BornCharges

description = "Compute the Born Effective Charges as derivative of polarization/dipole w.r.t. nuclear displacements."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to extxyz file with all structures produced by 'build_dataset4dPdR.py' (e.g. structures.extxyz)",
    )
    parser.add_argument(
        "-r",
        "--retry",
        type=str2bool,
        required=False,
        help="retry linear fit without bad data (default: %(default)s)",
        default=True,
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

    key = KEYWORDS["dipole"]
    print(f"Extracting dipole from '{key}' ... ", end="")
    dipole = np.asarray([atoms.info[key] for atoms in structures])
    print("done")
    print("dipole.shape: ", dipole.shape)

    key = KEYWORDS["displacements"]
    print(f"Extracting displacements from '{key}' ... ", end="")
    displacements = np.asarray([atoms.arrays[key] for atoms in structures])
    print("done")
    print("displacements.shape: ", displacements.shape)

    print("Reconstructing reference structure ... ", end="")
    positions = np.asarray([atoms.get_positions() for atoms in structures])
    reference = positions - displacements
    print("done")
    var = np.var(reference, axis=0)
    var_norm = np.linalg.norm(var)
    if var_norm > ATOL:
        raise ValueError("There has been a problem while reconstrucing the reference structure.")

    ref_pos = np.mean(reference, axis=0)
    reference = Atoms(
        positions=ref_pos,
        cell=structures[0].get_cell(),
        pbc=structures[0].get_pbc(),
        symbols=structures[0].get_chemical_symbols(),
    )
    reference = AtomicStructure.from_ase(reference)

    print("Preparing Born Charges and symmetrization ... ", end="")
    bec = BornCharges(data=np.zeros((Na, 3, 3)), cell=reference.cell)
    S, theta, theta_real = reference.get_symmetrizer(bec)
    A = np.kron(displacements.reshape((Ns, -1)), np.eye(3))
    b = dipole.flatten()
    print("done")
    n_unknown = S.shape[1]

    print("\nMatrix shapes:")
    print(" - b.shape:", b.shape)
    print(" - A.shape:", A.shape)
    print(" - S.shape:", S.shape)

    print("\nMatrix shapes with symmetrization:")
    A = A @ S
    print(" - b.shape:", b.shape)
    print(" - A.shape:", A.shape)

    print("\nMatrix for total dipoles:")
    tmp = np.tile(-np.eye(3), Ns).T
    A = np.hstack([A, tmp])
    print(" - b.shape:", b.shape)
    print(" - A.shape:", A.shape)

    print("Preparing linear system ... ", end="")
    ls = LinearSystem(A=A, b=b)
    print("done")

    print("Solving linear systems  ... ", end="")
    ls.solve()
    print("done")

    info = ls.summary()

    if info["quality"] == "BAD" and args.retry:
        print("Removing bad data and retrying to solve the linear system")
        mask = info["good_mask"]
        A = A[mask, :]
        b = b[mask]

        print("Preparing linear system ... ", end="")
        ls = LinearSystem(A=A, b=b)
        print("done")

        print("Solving linear systems  ... ", end="")
        ls.solve()
        print("done")

        ls.summary()

    print("Extracting Born Charges  ... ", end="")
    bec = S @ ls.x[:n_unknown]
    bec = bec.reshape((Na, 3, 3))
    print("done\n")

    print_born_charges(reference, bec)

    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)

    print("Checking Acoustic Sum Rule:")
    asr = np.asarray(bec.mean(axis=0))
    norm = matrix_norm(asr)
    print(f"||sum_I Z*_I|| = {norm:10.5f} | this value should be almost zero")
    if norm > ATOL:
        warn(f"Acoustic Sum Rule is not satisfied: {norm:.5e} > {ATOL:.1e}")

    file = folder / "bec-no-asr.txt"
    print(f"Writing Born Charges to {file} ... ", end="")
    np.savetxt(file, bec.reshape((Na, 9)), fmt=float_format)
    print("done")

    file = folder / "asr.txt"
    print(f"Writing sum of all Born Charges to {file} ... ", end="")
    np.savetxt(file, asr, fmt=float_format)
    print("done")

    file = folder / "bec.txt"
    print(f"Writing sum of Born Charges with ASR applied to {file} ... ", end="")
    np.savetxt(file, bec.reshape((Na, 9)) - asr.reshape((1, 9)), fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
