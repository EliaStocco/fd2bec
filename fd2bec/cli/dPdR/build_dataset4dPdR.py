# Tested by pytest: tests/test_aims_workflow_wrappers.py

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
from ase import Atoms

from fd2bec.cli import KEYWORDS, cli, extract_n, read_input_structures
from fd2bec.geometry import cart2frac, frac2cart
from fd2bec.io import read, write

# ============================================================
# FORMAT REGISTRY
# ============================================================

FORMAT_REGISTRY = {
    "aims_polarization": {
        "type": "polarization",
        "regex": (
            r"Cartesian Polarization\s+"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        "factor": 0.06241517271464743,
        # FHI-aims can be pretty verbose when computing the polarization.
        # Here we can provide some pattern to discard.
        "drop_patterns": ["Berry phase for band", "TT:", "- k ="],
    },
    "aims_dipole": {
        "type": "dipole",
        "regex": (
            r"\|\s*Total dipole moment \[eAng\]\s*:\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        "factor": 1.0,
        "drop_patterns": [],
    },
}


# ============================================================
# CLI DESCRIPTION
# ============================================================

description = (
    "Build dataset to compute Born Effective Charges as derivative of polarization/dipole w.r.t. nuclear displacements.\n"
    "The script reads either polarization or dipole from DFT output files.\n\n"
    "Input parsing is controlled via '--format'. This can be:\n"
    "  (1) a predefined format key\n"
    "  (2) a JSON file defining a custom format\n\n"
    "Each format defines:\n"
    "  - type: polarization or dipole\n"
    "  - regex: extraction pattern\n"
    "  - factor: unit conversion factor\n"
    "  - drop_patterns: optional lines to ignore\n\n"
    "Available formats:\n"
    f"{', '.join(FORMAT_REGISTRY.keys())}\n"
)

# ============================================================
# ARGPARSE
# ============================================================


def prepare_args(descr):

    parser = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    argv = {"metavar": "\b"}

    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="file with list of files or folder containing outputs",
    )

    parser.add_argument(
        "-r",
        "--reference",
        **argv,
        type=str,
        required=True,
        help="file with the reference structure",
    )

    parser.add_argument(
        "-f",
        "--format",
        **argv,
        type=str,
        required=True,
        help=(
            "Format key or JSON file.\n"
            f"Available built-in formats:\n{', '.join(FORMAT_REGISTRY.keys())}"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="output extxyz file",
    )

    return parser


# ============================================================
# FORMAT LOADER
# ============================================================


def load_format(fmt: str):
    path = Path(fmt)

    # JSON override
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        required = {"type", "regex", "factor"}
        missing = required - cfg.keys()
        if missing:
            raise ValueError(f"Missing keys in format JSON: {missing}")

        cfg.setdefault("drop_patterns", [])
        return cfg

    # built-in format
    if fmt in FORMAT_REGISTRY:
        return FORMAT_REGISTRY[fmt]

    raise ValueError(
        f"Unknown format '{fmt}'. Available: {list(FORMAT_REGISTRY.keys())} or a .json file"
    )


def extract_vectors(path, regex):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = regex.search(line)
            if m:
                vectors = [float(x) for x in m.groups()]
                assert len(vectors) == 3, "wrong shape"
                break
    return np.asarray(vectors)


# ============================================================
# TEMP FILE FILTER
# ============================================================


def filtered_temp_file(path, fmt):
    drop_patterns = fmt.get("drop_patterns", [])
    compiled = [re.compile(p) for p in drop_patterns]

    tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if any(p.search(line) for p in compiled):
                continue
            tmp.write(line)

    tmp.flush()
    return tmp.name


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ============================================================
# MAIN
# ============================================================


@cli(prepare_args, description)
def main(args):

    # ==============================
    # Read files
    # ==============================

    fmt = load_format(args.format)
    regex = re.compile(fmt["regex"])
    factor = fmt["factor"]
    dtype = fmt["type"]

    reference = read_input_structures(args.reference, label="reference structure")
    print("n. atoms: ", reference.get_global_number_of_atoms())

    print(f"Reading input structures from '{args.input}'")
    structures: List[Atoms] = []
    filenames = []
    vectors = []  # polarization or dipole
    input_path = Path(args.input)

    if input_path.is_dir():
        files = sorted((f for f in input_path.iterdir() if f.is_file()), key=extract_n)

    elif input_path.is_file():
        with open(input_path, "r", encoding="utf-8") as f:
            files = [Path(line.strip()) for line in f if line.strip() and not line.startswith("#")]
    else:
        raise FileNotFoundError(f"Input '{args.input}' is invalid.")

    for filename in files:
        try:
            print(f" - reading file '{filename}'")

            tmp_file = filtered_temp_file(filename, fmt)

            vectors.append(extract_vectors(tmp_file, regex))

            # original_n = count_lines(filename)
            # filtered_n = count_lines(tmp_file)
            # print("original:", original_n)
            # print("filtered:", filtered_n)

            atoms = read(tmp_file)
            structures.append(atoms)
            filenames.append(filename)
            os.remove(tmp_file)

        except Exception as e:
            warn(f"Exception while reading '{filename}'. File skipped.")
            print(e)

    print("n. read files:", len(structures))
    print("\nList of all read files:")
    for i, f in enumerate(filenames):
        print(f" - {i:3d}) {f}")

    # ==============================
    # periodic boundary conditions
    # ==============================
    pbc_list = [tuple(a.get_pbc()) for a in structures]

    # --- enforce PBC consistency ---
    if len(set(pbc_list)) != 1:
        raise ValueError(f"Inconsistent PBCs across structures: {pbc_list}")

    pbc = pbc_list[0]
    is_periodic = all(pbc)

    # ==============================
    # polarization
    # ==============================
    # --- dtype logic check ---
    if dtype == "polarization" and not is_periodic:
        raise ValueError(
            "dtype='polarization' requires periodic structures (PBC = True, True, True)."
        )

    # --- volume handling ---
    if dtype == "polarization":
        volumes = np.array([a.get_volume() for a in structures])
        v0 = volumes[0]

        if not np.allclose(volumes, v0):
            raise ValueError("Structures do not have the same volume.")

        cells = np.array([a.cell.array for a in structures])
        cell0 = cells[0]
        if not np.allclose(cells, cell0):
            raise ValueError("Structures do not have the same cell.")

    vectors = np.asarray(vectors)

    print(f"\nList of read '{dtype}':")
    for n, v in enumerate(vectors):
        formatted = [f"{x: 12.6e}" for x in v.tolist()]
        print(f" - {n:3d}) [{', '.join(formatted)}]")
    print()

    # ==============================
    # conversion
    # ==============================
    if dtype == "polarization":
        print("Using the following conversion factor to e/ang^2:", factor)
        vectors = vectors * factor
        print(f"Converting polarization to dipole using volume {v0} Å^3")
        dipoles = np.asarray(vectors * v0)
    else:
        print("Using the following conversion factor to e·Å:", factor)
        dipoles = np.asarray(vectors * factor)

    print()

    # ==============================
    # wrap
    # ==============================
    if dtype == "polarization":
        print("Wrapping dipoles on the same branch:")
        old_dipoles = dipoles.copy()
        cell = structures[0].get_cell()
        quanta = cart2frac(cell=cell, v=old_dipoles)
        new_quanta = np.unwrap(quanta, axis=0, period=1)
        dipoles = frac2cart(cell=cell, v=new_quanta)
        all_jumps = new_quanta - quanta

        # ==============================
        # final
        # ==============================
        N = len(dipoles)
        for n in range(N):
            old = [f"{x: 12.6e}" for x in old_dipoles[n].tolist()]
            new = [f"{x: 12.6e}" for x in dipoles[n].tolist()]
            jump = [f"{int(x): 3d}" for x in all_jumps[n].tolist()]

            print(
                f" - {n:3d}) [{', '.join(old)}] --> [{', '.join(new)}] | jump of [{', '.join(jump)}]"
            )
        print()

    # ==============================
    # final
    # ==============================
    print("Final dipoles:")
    for n, v in enumerate(dipoles):
        formatted = [f"{x: 12.6e}" for x in v.tolist()]
        print(f" - {n:3d}) [{', '.join(formatted)}]")
    print()

    key = KEYWORDS["dipole"]
    print(f"Saving dipoles [e*ang] to atomic structures using the keyword '{key}'")
    for n, atoms in enumerate(structures):
        atoms.info[key] = dipoles[n]

    key = KEYWORDS["displacements"]
    print(f"Constructing cartesian displacements and saving them using keyword '{key}'")
    pos0 = reference.get_positions()
    displacements = [a.get_positions() - pos0 for a in structures]
    for n, atoms in enumerate(structures):
        atoms.arrays[key] = displacements[n]

    info_keys = set()
    array_keys = set()

    for atoms in structures:
        info_keys.update(atoms.info.keys())
        array_keys.update(atoms.arrays.keys())

    print("info keys:", info_keys)
    print("arrays keys:", array_keys)

    print(f"Saving dataset to file '{args.output}'")
    write(args.output, structures)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()
