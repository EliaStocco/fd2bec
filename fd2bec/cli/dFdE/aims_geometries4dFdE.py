import argparse
import tempfile
from pathlib import Path

import numpy as np

from fd2bec.cli import cli, flist
from fd2bec.io import read, write

description = "Generate the FHI-aims geometry.in files to compute the Born Effective Charges as derivative of the forces w.r.t. applied electric field."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i", "--input", **argv, type=str, required=True, help="file with the input geometry"
    )
    parser.add_argument(
        "-e",
        "--efields",
        **argv,
        type=flist,
        required=False,
        help="list of electric fields (V/ang) to apply (default: %(default)s)",
        default=[-0.1, 0.0, 0.1],
    )
    parser.add_argument(
        "-o", "--output", **argv, type=str, required=True, help="output folder with the geometries"
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input geometry from file '{args.input}' ... ", end="")
    atoms = read(args.input, index=0)
    print("done")
    print("n. atoms:", atoms.get_global_number_of_atoms())

    print("Using the following field intensities:", args.efields)
    efields = np.asarray(args.efields)
    efields = np.asarray([np.eye(3) * e for e in efields]).reshape((-1, 3))
    efields = np.unique(efields, axis=0)

    print("Using the following fields:")
    for n, e in enumerate(efields):
        print(f" - {n}) ", e)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Generate a reference geometry.in with ASE
    with tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".in",
        delete=True,
    ) as tmp:
        write(tmp.name, atoms, format="aims")
        tmp.seek(0)
        geometry_text = tmp.read().rstrip()

    print(f"Writing geometries to '{output}' ...")
    for n, e in enumerate(efields):
        field = " ".join(f"{x:g}" for x in e)
        outfile = output / f"geometry.n={n}.in"
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(geometry_text)
            f.write("\n")
            f.write(f"homogeneous_field {field}\n")
        print(f" - {n}) {outfile}: homogeneous_field {field}")
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
