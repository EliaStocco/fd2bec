"""Split a multi-frame extxyz file into one folder per structure."""

# Tested by pytest: tests/test_split_extxyz.py

import argparse
import os
import shlex
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, List
from zipfile import ZIP_DEFLATED, ZipFile

from ase import Atoms
from ase.io import iread as ase_iread

from fd2bec.cli import cli, read_input_structures
from fd2bec.io import format_atoms, write

description = "Split a multi-frame extxyz file into structure folders or a ZIP archive."

RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
output_dir=__OUTPUT_DIR__

for folder in "$output_dir"/structure-*/; do
    [[ -d "$folder" ]] || continue
    echo "Processing ${folder%/}"
    (
        cd "$folder"

        # Add commands to run for each structure here, for example:
        # prepare_aims -i start.extxyz --k-density 5.0
    )
done
"""


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        required=True,
        help="path to the multi-frame extxyz input",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="dataset",
        help="output folder or archive base name (default: %(default)s)",
    )
    parser.add_argument(
        "--zip",
        "--archive",
        dest="zip_output",
        action="store_true",
        help="write a streaming ZIP archive instead of creating folders",
    )
    return parser


def write_run_script(output: Path, script: Path = None) -> Path:
    """Write an editable script that processes every structure folder."""
    script = Path.cwd() / "run_all.sh" if script is None else Path(script)
    output_reference = os.path.relpath(output.resolve(), script.parent.resolve())
    text = RUN_SCRIPT.replace("__OUTPUT_DIR__", shlex.quote(output_reference))
    script.write_text(text, encoding="utf-8")
    script.chmod(0o755)
    return script


def write_dataset(structures: Iterable[Atoms], output: Path) -> List[Path]:
    """Write one extxyz file for each structure.

    Each structure is written as ``structure-<index>/start.extxyz``. Existing
    output directories are reused, which makes the operation safe to rerun
    without removing any files added by the user.
    """
    structures = list(structures)
    if not structures:
        raise ValueError("The input contains no structures.")

    output.mkdir(parents=True, exist_ok=True)
    filenames = []
    for index, atoms in enumerate(structures):
        structure_dir = output / f"structure-{index}"
        structure_dir.mkdir(exist_ok=True)
        filename = structure_dir / "start.extxyz"
        write(filename, atoms, format="extxyz")
        filenames.append(filename)

    return filenames


def _extxyz_text(atoms: Atoms) -> str:
    """Serialize one structure without creating a temporary file."""
    stream = StringIO()
    write(stream, atoms, format="extxyz")
    return stream.getvalue()


def write_dataset_archive(structures: Iterable[Atoms], output: Path) -> int:
    """Stream structures into a ZIP archive and return the structure count.

    Only the current structure is serialized in memory. The archive contains
    the same structure layout as folder mode. The runner script is written
    next to the archive by :func:`main`.
    """
    structures = iter(structures)
    try:
        first = next(structures)
    except StopIteration as error:
        raise ValueError("The input contains no structures.") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for count, atoms in enumerate(chain((first,), structures), start=1):
            archive.writestr(f"structure-{count - 1}/start.extxyz", _extxyz_text(atoms))
    return count


def iter_extxyz(filename: str) -> Iterator[Atoms]:
    """Yield extxyz structures one at a time."""
    for atoms in ase_iread(filename, index=":", format="extxyz"):
        yield format_atoms(atoms, rename=False)


@cli(prepare_args, description)
def main(args):
    """Split the input file and create the editable runner script."""
    output = Path(args.output)
    if args.zip_output:
        if output.suffix.lower() != ".zip":
            output = output.with_name(output.name + ".zip")
        print(f"Streaming structures from {args.input} to {output} ... ", end="")
        count = write_dataset_archive(iter_extxyz(args.input), output)
        print(f"done ({count} structures)")
        script = write_run_script(output.with_suffix(""))
        print(f"Wrote editable runner script to {script}")
        return

    structures = read_input_structures(args.input, index=":")

    print(f"Writing structures to {output} ... ", end="")
    filenames = write_dataset(structures, output)
    print(f"done ({len(filenames)} files)")
    script = write_run_script(output)
    print(f"Wrote editable runner script to {script}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
