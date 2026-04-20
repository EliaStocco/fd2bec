import json

import numpy as np

from fd2bec import float_format
from fd2bec.cli import cli

description = "Extract the Born Effective Charges from the JSON solution file."

choices = ["pseudo-inverse", "lstsq"]


def prepare_args(description):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="JSON input file produced by 'solve_linear_system.py'",
    )
    # parser.add_argument("-m", "--method"       , **argv, type=str  , required=True, help=f"method: {choices}"+" (default: %(default)s)", default="pseudo-inverse", choices=choices)
    parser.add_argument(
        "-o", "--output", **argv, type=str, required=True, help="txt output file"
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading solitio system from {args.input} ... ", end="")
    with open(args.input, "r") as f:
        solution = json.load(f)
    print("done")

    bec = np.asarray(solution["results"]["bec"])
    print(f"bec.shape: {bec.shape}")

    Natoms = len(solution["equation"]["unitcell"]["symbols"])

    # if solution['equation']['asr_weight'] > 0:
    #     bec = bec[1:,:]
    # else:
    #     bec = bec

    assert bec.shape == (Natoms, 3, 3), (
        f"Expected shape {(Natoms, 3, 3)}, got {bec.shape}"
    )

    bec = bec.reshape((Natoms, 9))
    print(f"Writing cartesian displacements to {args.output} ... ", end="")
    np.savetxt(args.output, bec, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()
