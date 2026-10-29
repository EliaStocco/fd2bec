"""Export Born effective charges and directional response vectors for OVITO."""

import argparse

from fd2bec.born_charges_visualization import attach_born_charge_visualization_data
from fd2bec.cli import KEYWORDS, cli, read_input_structures
from fd2bec.io import write

description = (
    "Write an extxyz structure with each Born effective charge tensor and the induced "
    "response vector for a selected displacement direction.\nIn OVITO, display the "
    "resulting BEC_response property with the Vector Visualization modifier."
)


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="input extxyz structure")
    parser.add_argument(
        "-k",
        "--keyword",
        default=KEYWORDS["bec"],
        help="per-atom BEC array key (default: %(default)s)",
    )
    parser.add_argument(
        "--direction",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("NX", "NY", "NZ"),
        help="Cartesian displacement direction (default: 0 0 1)",
    )
    parser.add_argument(
        "--prefix",
        default="BEC_",
        help="prefix for exported per-atom properties (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", **argv, required=True, help="output extxyz path")
    return parser


@cli(prepare_args, description)
def main(args):
    """Attach BEC visualization properties and write the extended XYZ output."""
    atoms = read_input_structures(args.input, input_format="extxyz")
    print(f"Adding Born-charge visualization data from '{args.keyword}' ... ", end="")
    output, direction = attach_born_charge_visualization_data(
        atoms, args.keyword, args.direction, prefix=args.prefix
    )
    print("done")
    print(
        f"Selected displacement direction: [{direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f}]"
    )
    print(f"Writing OVITO-ready extxyz to {args.output} ... ", end="")
    write(args.output, output, format="extxyz")
    print("done")
    print(
        f"Use '{args.prefix}response' as the arrow vector property; the exported scalar "
        f"properties are '{args.prefix}longitudinal', '{args.prefix}magnitude', and "
        f"'{args.prefix}transverse'."
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
