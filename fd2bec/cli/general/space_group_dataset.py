"""Write space-group information for every structure in a multi-frame extxyz."""

# Tested by pytest: tests/test_space_group.py

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fd2bec import SYMPREC
from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.tools import ase2spglib_dataset

description = "Write lattice and space-group information for a multi-frame extxyz file."

CSV_COLUMNS = (
    "n. atoms",
    "volume",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "space group",
    "space group symbol",
    "Hall symbol",
    "crystal class",
    "Bravais lattice type",
    "centrosymmetric",
    "number of symmetry operations",
    "primitive cell atoms",
    "standardized conventional cell atoms",
    "input cell is primitive",
)


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to a multi-frame extxyz file",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to the output CSV file",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        **argv,
        type=float,
        default=SYMPREC,
        help="symmetry tolerance passed to spglib (default: %(default)s)",
    )
    parser.add_argument(
        "--plot-output",
        **argv,
        type=str,
        default=None,
        help="path to the statistics image (default: output CSV path with .png)",
    )
    return parser


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _bravais_type(atoms, threshold):
    try:
        return atoms.cell.get_bravais_lattice(eps=threshold).longname
    except (AttributeError, ValueError):
        return "undetermined"


def structure_record(atoms, threshold=SYMPREC):
    """Return one CSV row containing lattice and space-group information."""
    record = {column: None for column in CSV_COLUMNS}
    record["n. atoms"] = len(atoms)
    if not np.all(atoms.get_pbc()):
        return record

    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    record.update(
        {
            "volume": float(atoms.get_volume()),
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        }
    )

    dataset = ase2spglib_dataset(atoms, symprec=threshold)
    if dataset is None:
        return record

    primitive_count = len(np.unique(dataset.mapping_to_primitive))
    record.update(
        {
            "space group": int(dataset.number),
            "space group symbol": _text(dataset.international),
            "Hall symbol": _text(dataset.hall),
            "crystal class": _text(dataset.pointgroup),
            "Bravais lattice type": _bravais_type(atoms, threshold),
            "centrosymmetric": any(
                np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in dataset.rotations
            ),
            "number of symmetry operations": len(dataset.rotations),
            "primitive cell atoms": primitive_count,
            "standardized conventional cell atoms": len(dataset.std_positions),
            "input cell is primitive": len(atoms) == primitive_count,
        }
    )
    return record


def collect_space_group_information(structures, threshold=SYMPREC):
    """Return one ordered record for each structure."""
    return [structure_record(structure, threshold) for structure in structures]


def _plot_histogram(axis, dataframe, columns, labels, title, xlabel):
    plotted = False
    for column, label in zip(columns, labels):
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty:
            continue
        axis.hist(values, bins="auto", alpha=0.65, label=label)
        plotted = True
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.grid(alpha=0.25)
    if plotted and len(columns) > 1:
        axis.legend()
    if not plotted:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)


def plot_dataset_statistics(dataframe, output):
    """Create one image containing the main dataset statistics."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fd2bec-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    _plot_histogram(
        axes[0, 0],
        dataframe,
        ("a", "b", "c"),
        ("a", "b", "c"),
        "Lattice parameters",
        "Length [Angstrom]",
    )
    _plot_histogram(
        axes[0, 1], dataframe, ("volume",), ("volume",), "Cell volumes", "Volume [Angstrom^3]"
    )
    _plot_histogram(
        axes[1, 0],
        dataframe,
        ("number of symmetry operations",),
        ("operations",),
        "Symmetry operations",
        "Number of operations",
    )

    _plot_histogram(
        axes[1, 1], dataframe, ("n. atoms",), ("atoms",), "Number of atoms", "Number of atoms"
    )

    axis = axes[0, 2]
    valid = dataframe.dropna(subset=("space group",)).copy()
    if valid.empty:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        axis.set_title("Space-group frequencies")
    else:
        valid["space group"] = pd.to_numeric(valid["space group"], errors="coerce")
        valid = valid.dropna(subset=("space group",))
        counts = valid.groupby(["space group", "space group symbol"], dropna=False).size()
        counts = counts.sort_values(ascending=False)
        labels = [f"{int(number)}\n{symbol}" for number, symbol in counts.index]
        axis.bar(labels, counts.to_numpy())
        axis.set_title("Space-group frequencies")
        axis.set_xlabel("Number\nInternational symbol")
        axis.set_ylabel("Count")
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 2]
    correlation = (
        dataframe[["n. atoms", "number of symmetry operations"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if correlation.empty:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        axis.set_title("Atoms vs symmetry operations")
    else:
        x_values = correlation["n. atoms"]
        y_values = correlation["number of symmetry operations"]
        axis.scatter(x_values, y_values, alpha=0.75)
        if len(correlation) > 1 and x_values.nunique() > 1 and y_values.nunique() > 1:
            coefficient = np.corrcoef(x_values, y_values)[0, 1]
            title = f"Atoms vs symmetry operations (r = {coefficient:.3f})"
        else:
            title = "Atoms vs symmetry operations"
        axis.set_title(title)
        axis.set_xlabel("Number of atoms")
        axis.set_ylabel("Number of operations")
        axis.grid(alpha=0.25)

    figure.suptitle(f"Dataset statistics ({len(dataframe)} structures)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


@cli(prepare_args, description)
def main(args):
    print(f"Reading structures from {args.input} ... ", end="")
    structures = read(args.input, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    print(f"done ({len(structures)} structure(s))")

    print("Computing lattice and space-group information ... ", end="")
    records = collect_space_group_information(structures, args.threshold)
    print("done")

    print(f"Writing CSV file to {args.output} ... ", end="")
    dataframe = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)
    dataframe.to_csv(args.output, index=False)
    print("done")

    plot_output = args.plot_output or str(Path(args.output).with_suffix(".png"))
    print(f"Writing dataset statistics image to {plot_output} ... ", end="")
    plot_dataset_statistics(dataframe, plot_output)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
