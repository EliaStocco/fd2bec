import argparse

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.show import print_reference_structure
from fd2bec.tensor import MAPPING

description = "Show the symmetry inequivalent components of a tensor."


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
        required=True,
        help="name of the tensor",
        choices=MAPPING.keys(),
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    reference = read(args.input, index=0)
    print("done")

    print_reference_structure(reference)
    unit_cell = AtomicStructure.from_ase(reference)

    if args.name not in MAPPING:
        raise ValueError(f"{args.name} not supported.")

    print("Constructing tensor ... ", end="")
    cls = MAPPING[args.name]
    tensor = cls.template(len(unit_cell))
    print("done")
    print("tensor.shape:", tensor.shape)
    print("tensor summary: ", tensor)

    print("\nComputing symmetry inequivalent components ... ", end="")
    S, theta, theta_real = unit_cell.get_symmetrizer(tensor=tensor)
    print("done")
    print(f"Found {len(theta)} independent component(s)")

    pass


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
