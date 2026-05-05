import argparse
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
from ase import Atoms

from fd2bec import ATOL, SYMPREC, float_format

# from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.linear_system import LinearSystem, StackedLinearSystem

# from fd2bec.tensor import BornCharge

# from pymatgen.core import Molecule
# from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# # convert ASE -> pymatgen
# from ase.io import write
# from pymatgen.io.ase import AseAtomsAdaptor


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
        help="path to extxyz file with all structures (e.g. structures.extxyz)",
    )
    parser.add_argument(
        "-e",
        "--efield_keyword",
        **argv,
        type=str,
        required=False,
        help="keyword for the electric field [V/ang] (default: %(default)s)",
        default="efield",
    )
    parser.add_argument(
        "-f",
        "--forces_keyword",
        **argv,
        type=str,
        required=False,
        help="keyword for the forces [eV/ang] (default: %(default)s)",
        default="REF_forces",
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
    parser.add_argument(
        "-c",
        "--clean_forces",
        type=str,
        required=False,
        help="clean forces (default: %(default)s)",
        default=False,
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

    print(f"Reading input structures from '{args.input}' ... ", end="")
    structures: List[Atoms] = read(args.input, index=":")
    print("done")

    Ns = len(structures)
    Na = structures[0].get_global_number_of_atoms()
    print("n. structures: ", Ns)
    print("n. atoms: ", Na)

    pos = np.asarray([atoms.get_positions() for atoms in structures])
    assert np.all([np.allclose(pos_i, pos[0]) for pos_i in pos]), (
        "You have provided different geometries."
    )

    print(f"Extracting electric field from '{args.efield_keyword}' ... ", end="")
    efield = np.asarray([atoms.info["efield"] for atoms in structures])
    print("done")
    print("efield.shape: ", efield.shape)

    print(f"Extracting forces from '{args.forces_keyword}' ... ", end="")
    forces = np.asarray([atoms.arrays["REF_forces"] for atoms in structures])
    print("done")
    print("forces.shape: ", forces.shape)

    if not args.clean_forces:
        print("Not cleaning forces")
    else:
        print("Cleaning forces ... ", end="")
        forces -= np.mean(forces, axis=1, keepdims=True)
        print("done")

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

    bec = np.zeros((Na, 3, 3))
    print("Extracting Born Charges  ... ", end="")
    for n in range(Na):
        bec[n, :, :] = all_ls[n].x[1:, :]
    print("done")

    file = Path(args.output) / "bec.txt"
    print(f"Writing Born Charges to {file} ... ", end="")
    np.savetxt(file, bec.reshape((Na, 9)), fmt=float_format)
    print("done")

    file = Path(args.output) / "asr.txt"
    print(f"Writing sum of all Born Charges to {file} ... ", end="")
    asr = bec.sum(axis=0)
    np.savetxt(file, asr, fmt=float_format)
    print("done")

    file = Path(args.output) / "bec-asr.txt"
    print(f"Writing sum of Born Charges with ASR applied to {file} ... ", end="")
    np.savetxt(file, bec.reshape((Na, 9)) - asr.reshape((1, 9)), fmt=float_format)
    print("done")

    # print("Symmetrizing Born Charges  ... ", end="")
    # aperiodic = structures[0].copy()

    # pmg_mol = AseAtomsAdaptor.get_molecule(aperiodic)

    # analyzer = PointGroupAnalyzer(pmg_mol, tolerance=1e-3)

    # H = analyzer.get_symmetry_operations()
    # R = np.asarray([ h.rotation_matrix for h in H])
    # T = np.asarray([ h.translation_vector for h in H])

    # aperiodic.set_cell([100, 100, 100])

    # unit_cell = AtomicStructure.from_ase(aperiodic)
    # z = BornCharge(data=bec)
    # P = unit_cell.get_totally_symmetric_projection(tensor=z)
    # print("done")

    # p = Path(args.output)
    # new_filename = p.with_name(f"{p.stem}_sym{p.suffix}")

    return


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
