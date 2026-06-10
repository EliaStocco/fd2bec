import argparse
import re
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
from ase import Atoms

from fd2bec.cli import KEYWORDS, cli
from fd2bec.io import read, write

# for file in raw/run-0/results/* ; do echo ${file} >> list.txt; done

REGEX = {
    "aims": r"homogeneous_field\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
}

description = "Build the dataset to compute the Born Effective Charges as derivative of the forces w.r.t. applied electric field."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="txt file with the list of files to read or folder with the files",
    )
    parser.add_argument(
        "-er",
        "--efield_regex",
        **argv,
        type=str,
        required=False,
        help="regex to read the electric field from file or file format (default: %(default)s)",
        default="aims",
    )
    parser.add_argument(
        "-ef",
        "--efield_factor",
        **argv,
        type=float,
        required=False,
        help="conversion factor to V/ang for the electric field from file (default: %(default)s)",
        default=1,
    )
    parser.add_argument(
        "-f",
        "--forces_keyword",
        **argv,
        type=str,
        required=False,
        help="keyword for the forces [eV/ang] (default: %(default)s)",
        default="forces",
    )
    parser.add_argument(
        "-o", "--output", **argv, type=str, required=True, help="extxyz output file"
    )
    return parser


@cli(prepare_args, description)
def main(args):

    assert Path(args.output).suffix == ".extxyz", f"'{args.output}' must be an extxyz file."

    print(f"Reading input structures from '{args.input}'")
    structures: List[Atoms] = []
    input_path = Path(args.input)
    filenames = []
    if input_path.is_dir():
        for filename in sorted(input_path.iterdir()):
            if not filename.is_file():
                continue
            try:
                atoms = read(filename)
                structures.append(atoms)
                filenames.append(filename)
            except Exception as e:
                warn(f"Exception while reading file '{filename}'. This file will be discarded.'")
                print(e)

    elif input_path.is_file():
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                filename = line.strip()
                # skip empty lines and comments
                if not filename or filename.startswith("#"):
                    continue
                try:
                    atoms = read(filename)
                    structures.append(atoms)
                    filenames.append(filename)
                except Exception as e:
                    warn(
                        f"Exception while reading file '{filename}'. This file will be discarded.'"
                    )
                    print(e)

    else:
        raise FileNotFoundError(f"Input '{args.input}' is neither a file nor a directory.")
    # print("done")
    print("n. read files: ", len(structures))
    print("List of all read files:")
    for n, file in enumerate(filenames):
        print(f"- {n}) {file}")

    info_keys = set()
    array_keys = set()

    for atoms in structures:
        info_keys.update(atoms.info.keys())
        array_keys.update(atoms.arrays.keys())

    print("info keys:", info_keys)
    print("arrays keys:", array_keys)

    key = KEYWORDS["forces"]
    assert args.forces_keyword in array_keys, "error"
    print(f"Using '{args.forces_keyword}' as forces and saving them to '{key}'")
    for atoms in structures:
        atoms.arrays[key] = atoms.arrays.pop(args.forces_keyword)

    if args.efield_regex in REGEX:
        args.efield_regex = REGEX[args.efield_regex]

    print("Using the following regex to read the electric field from file:")
    print(args.efield_regex)

    print("Using the following conversion factor to V/ang: ", args.efield_factor)

    key = KEYWORDS["efield"]
    print(f"Extracting electric field from files and saving it to '{key}':")
    efield_pattern = re.compile(args.efield_regex)
    for atoms, file in zip(structures, filenames):
        print(f" - {file}: ", end="")

        efield = None

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                match = efield_pattern.search(line)

                if match:
                    # extract all numeric groups from the match
                    # supports formats like: "electric_field 0.1 0.2 0.3"
                    try:
                        efield = np.array([float(x) for x in match.groups()])
                    except Exception:
                        # fallback: extract all floats from the line
                        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                        if len(nums) >= 3:
                            efield = np.array(list(map(float, nums[:3])))

                    break

        if efield is None:
            raise ValueError(f"Could not find electric field in file: {file}")

        atoms.info[key] = efield * args.efield_factor
        print(efield)

    print(f"Saving dataset to file '{args.output}'")
    write(args.output, structures)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
