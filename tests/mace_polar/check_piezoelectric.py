"""Check the MACE-POLAR centrosymmetric BiFeO3 piezoelectric regression."""

import argparse

import numpy as np

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS
from fd2bec.io import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--proper", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    structures = read(args.dataset, format="extxyz", index=":")
    if len(structures) != 13:
        raise AssertionError(f"Expected 13 finite-strain structures, got {len(structures)}")
    reference = AtomicStructure.from_ase(structures[0])
    if reference.space_group != 167:
        raise AssertionError(f"Expected R-3c space group 167, got {reference.space_group}")

    polarizations = np.asarray(
        [atoms.info[KEYWORDS["polarization"]] for atoms in structures]
    )
    if not np.all(np.isfinite(polarizations)):
        raise AssertionError("MACE-POLAR produced non-finite polarization proxies")

    proper = np.loadtxt(args.proper)
    if proper.shape != (3, 6):
        raise AssertionError(f"Expected a 3x6 proper tensor, got {proper.shape}")
    maximum = float(np.max(np.abs(proper)))
    if maximum > args.tolerance:
        raise AssertionError(
            f"R-3c BiFeO3 proper piezoelectric tensor is not zero: "
            f"max |e_ij| = {maximum:.6e} > {args.tolerance:.6e}"
        )
    print(f"R-3c BiFeO3 proper piezoelectric regression passed: max |e_ij|={maximum:.3e}")


if __name__ == "__main__":
    main()
