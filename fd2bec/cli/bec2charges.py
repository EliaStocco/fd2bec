import numpy as np

from fd2bec import float_format
from fd2bec.cli import cli

description = "Convert BEC to atomic charges."


def prepare_args(descr):
    import argparse

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to txt file with BEC (e.g. bec.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to txt output file (e.g. charges.txt)",
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    bec = np.loadtxt(args.input)
    print("done")

    bec = bec.reshape((-1, 3, 3))
    charges = bec[:, 0, 0] + bec[:, 1, 1] + bec[:, 2, 2]
    charges /= 3.0

    print("Total sum: ", np.sum(charges))

    print(f"Writing charges to {args.output} ... ", end="")
    np.savetxt(args.output, charges, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()
