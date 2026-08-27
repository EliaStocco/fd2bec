"""Convert, standardize, or rotate one structure between ASE formats.

The default ``cif`` output route writes spglib symmetry operations in the
input-cell basis, preserving both the cell and atom count.  The optional
``--conventional`` route uses pymatgen to produce a standardized conventional
CIF.  ``espresso-in`` writes the QE geometry cards already used by the
displacement workflow.
"""

import argparse
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io.formats import ioformats

from fd2bec import SYMPREC
from fd2bec.cli import cli
from fd2bec.io import ESPRESSO_GEOMETRY_FORMAT, inferred_output_format, read, write_structure
from fd2bec.structure_alignment import is_ase_standard_cell as is_ase_standard_cell

description = "Convert, standardize, or rotate one structure between ASE-supported formats."
ase_writable_formats = sorted(name for name, ioformat in ioformats.items() if ioformat.can_write)
output_formats = sorted(set(ase_writable_formats) | {ESPRESSO_GEOMETRY_FORMAT})


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="path to input structure")
    parser.add_argument("-o", "--output", **argv, required=True, help="path to converted structure")
    parser.add_argument(
        "-f",
        "--format",
        **argv,
        choices=output_formats,
        help="output format; inferred from the output extension when omitted",
    )
    parser.add_argument(
        "--input-format",
        **argv,
        help="ASE input format; inferred from the input filename when omitted",
    )
    parser.add_argument(
        "--index",
        **argv,
        type=int,
        default=0,
        help="index of the input structure to convert (default: %(default)s)",
    )
    parser.add_argument(
        "--symprec",
        **argv,
        type=float,
        default=SYMPREC,
        help="symmetry tolerance used for CIF output (default: %(default)s)",
    )
    cell_setting = parser.add_mutually_exclusive_group()
    cell_setting.add_argument(
        "--conventional",
        action="store_true",
        help="write CIF in pymatgen's conventional standardized cell",
    )
    cell_setting.add_argument(
        "--primitive",
        action="store_true",
        help="preserve the primitive input cell and write its explicit symmetry operations",
    )
    transforms = parser.add_argument_group("input-cell transformations")
    transforms.add_argument(
        "--standardize",
        **argv,
        choices=("primitive", "conventional"),
        help="standardize the input cell with spglib before writing",
    )
    transforms.add_argument(
        "--rotate-cell",
        action="store_true",
        help="rotate a periodic input cell into ASE's lower-triangular standard form",
    )
    return parser


def rotate_to_ase_standard_cell(atoms: Atoms) -> Atoms:
    """Return a copy of ``atoms`` with ASE's lower-triangular standard cell."""
    if not np.all(atoms.pbc):
        raise ValueError("Cell rotation requires a fully periodic structure.")

    rotated = atoms.copy()
    standard_cell, rotation = rotated.cell.standard_form()
    rotated.set_positions(rotated.get_positions() @ rotation.T)
    rotated.set_cell(standard_cell, scale_atoms=False)
    return rotated


def standardize_structure(atoms: Atoms, *, setting: str, symprec: float) -> Atoms:
    """Return a primitive or conventional standardized copy of ``atoms``."""
    if setting not in {"primitive", "conventional"}:
        raise ValueError(f"Unknown standardized cell setting {setting!r}.")
    if not np.all(atoms.pbc):
        raise ValueError("Structure standardization requires a fully periodic structure.")

    standardized = spglib.standardize_cell(
        (atoms.cell.array, atoms.get_scaled_positions(wrap=True), atoms.numbers),
        to_primitive=setting == "primitive",
        no_idealize=False,
        symprec=symprec,
    )
    if standardized is None:
        raise ValueError("spglib could not standardize the structure.")
    cell, fractional_positions, numbers = standardized
    return Atoms(
        numbers=numbers,
        cell=cell,
        scaled_positions=fractional_positions,
        pbc=True,
    )


@cli(prepare_args, description)
def main(args):
    """Convert the requested input structure."""
    if args.standardize == "primitive" and args.conventional:
        raise ValueError("--standardize primitive cannot be combined with --conventional.")
    if args.standardize == "conventional" and args.primitive:
        raise ValueError("--standardize conventional cannot be combined with --primitive.")

    print(f"Reading structure {args.index} from {args.input} ... ", end="")
    atoms = read(args.input, index=args.index, format=args.input_format)
    print("done")

    if args.standardize:
        print(f"Constructing {args.standardize} standardized cell ... ", end="")
        atoms = standardize_structure(atoms, setting=args.standardize, symprec=args.symprec)
        print(f"done ({len(atoms)} atoms)")
    if args.rotate_cell:
        print("Rotating cell into ASE standard form ... ", end="")
        atoms = rotate_to_ase_standard_cell(atoms)
        print("done")

    output = Path(args.output)
    output_format = args.format or inferred_output_format(output)
    print(f"Writing {output_format or 'ASE-inferred'} structure to {output} ... ", end="")
    write_structure(
        output,
        atoms,
        output_format,
        symprec=args.symprec,
        conventional=args.conventional or args.standardize == "conventional",
        primitive=args.primitive or args.standardize == "primitive",
    )
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
