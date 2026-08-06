import argparse
import re
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.units import Bohr

from fd2bec import ATOL
from fd2bec.cli import cli
from fd2bec.mathematics import wrap

description = "Post process Quantum ESPRESSO Berry-phase polarization calculations."


def parse_polarization_scalar(nscf_file: Path) -> float:
    """Extracts scalar P (in e/Omega * bohr) from a QE NSCF Berry phase output.

    Target line format:
        P =   1.6664509  (mod  15.1058715)  (e/Omega).bohr
    """
    pattern = re.compile(
        r"P\s*=\s*([-+]?\d*\.\d+|\d+)\s*\(mod\s*[-+]?\d*\.\d+|\d+\)\s*\(e/Omega\)\.bohr",
        re.IGNORECASE,
    )
    with open(nscf_file, "r", encoding="utf-8") as f:
        content = f.read()

    matches = pattern.findall(content)
    if not matches:
        raise ValueError(
            f"Could not parse polarization scalar 'P = ... (e/Omega).bohr' from '{nscf_file}'."
        )

    return float(matches[-1])


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="reference un-displaced structure file",
    )
    parser.add_argument(
        "-r",
        "--results",
        **argv,
        default="results",
        help="folder containing QE displacement subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="dataset.extxyz",
        help="assembled polarized dataset destination (default: %(default)s)",
    )
    return parser


@cli(prepare_args, description)
def main(args):
    """Read QE polarization subdirectories and assemble the extxyz dataset."""
    results = Path(args.results)
    if not results.is_dir():
        raise FileNotFoundError(f"QE results directory not found: '{results}'.")

    reference = read(args.input)

    dataset = []

    n = 0
    while True:
        geometries_dir = results / f"geometry.n={n}"
        if not geometries_dir.is_dir():
            break

        scf = geometries_dir / "scf.out"
        atoms = read(scf)

        # Validate that atomic positions match the reference structure within ATOL
        a = atoms.get_scaled_positions()
        b = reference.get_scaled_positions()
        diff = np.abs(wrap(a - b))
        assert np.all(diff < ATOL), f"Structure in geometry.n={n} exceeds tolerance."

        p_scalars = np.zeros(3)

        for xyz in range(1, 4):
            nscf = geometries_dir / f"nscf.g={xyz}.out"
            pol_atoms = read(nscf)

            # Ensure NSCF geometry consistency
            assert np.allclose(atoms.positions, pol_atoms.positions), (
                f"Mismatch in positions for geometry.n={n}, g={xyz}."
            )
            assert np.allclose(atoms.cell, pol_atoms.cell), (
                f"Mismatch in cell for geometry.n={n}, g={xyz}."
            )

            p_scalars[xyz - 1] = parse_polarization_scalar(nscf)

        volume = atoms.get_volume()
        real_cell = atoms.get_cell()  # rows are real-space cell vectors a1, a2, a3

        # Transform dimensionless reciprocal phase scalars P_i to 3D Cartesian polarization:
        # P_cart = (1 / Omega) * sum_i ( P_i * Bohr * a_i )  -->  [e / Angstrom^2]
        cart_pol = np.dot(p_scalars * Bohr, real_cell) / volume

        # Compute total Cartesian dipole moment: D = Volume * P_cart  -->  [e * Angstrom]
        atoms.info["REF_dipole"] = volume * cart_pol

        dataset.append(atoms)

        n += 1

    if not dataset:
        raise ValueError(
            f"No valid geometry directories ('geometry.n=0', etc.) found in '{results}'."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, dataset, format="extxyz")

    print(f"Processed {len(dataset)} structure(s).")
    print(f"Polarized dataset written to '{output_path}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
