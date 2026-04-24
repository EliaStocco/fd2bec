import argparse

import numpy as np
import pandas as pd
from ase.io import read

from fd2bec.cli import cli

description = "Overview of the mathematical problem to solve."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure (e.g. unitcell.extxyz)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to CSV output file (e.g. overview.csv)",
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    Nbec_uc = 9 * atoms.get_global_number_of_atoms() + 3  #
    n = 1
    rows = []
    while True:
        factor = n * n * n
        tran = np.ceil(Nbec_uc / factor)
        row = {
            "supercell": f"{n}x{n}x{n}",
            "n. unit cells": factor,
            "tran": int(tran),
        }

        rows.append(row)
        if tran == 1:
            break

        n += 1

    df = pd.DataFrame(rows, columns=["supercell", "n. unit cells", "tran"])
    df.to_csv(args.output, index=False)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
