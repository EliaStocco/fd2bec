import numpy as np
import pandas as pd
from ase import Atoms

from fd2bec.cli.structures.space_group_dataset import (
    CSV_COLUMNS,
    collect_space_group_information,
    plot_dataset_statistics,
)
from fd2bec.show import print_cell, print_positions, print_space_group, print_symmetry_operations
from fd2bec.tools import ase2spglib_dataset


def test_structure_information_helpers_print_cell_and_positions(capsys):
    atoms = Atoms(
        "Si2",
        cell=np.diag([5.43, 5.43, 5.43]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        pbc=True,
    )

    print_cell(atoms)
    print_positions(atoms)
    output = capsys.readouterr().out

    assert "Cell vectors [Angstrom]:" in output
    assert "volume [Angstrom^3]" in output
    assert "Positions (Cartesian [Angstrom] and fractional):" in output
    assert "0.250000" in output


def test_symmetry_operations_are_printed(capsys):
    atoms = Atoms(
        "Si2",
        cell=np.diag([5.43, 5.43, 5.43]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        pbc=True,
    )
    dataset = ase2spglib_dataset(atoms, symprec=1e-3)

    print_symmetry_operations(dataset)
    output = capsys.readouterr().out

    assert "Symmetry operations (fractional coordinates x' = R x + t):" in output
    assert "#1" in output
    assert "rotation:" in output
    assert "translation:" in output


def test_space_group_summary_has_readable_symmetry_fields(capsys):
    atoms = Atoms(
        "Si2",
        cell=np.diag([5.43, 5.43, 5.43]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        pbc=True,
    )
    dataset = ase2spglib_dataset(atoms, symprec=1e-3)

    print_space_group(dataset, atoms, 1e-3)
    output = capsys.readouterr().out

    assert "International symbol" in output
    assert "Crystal class" in output
    assert "Bravais lattice type" in output
    assert "Number of symmetry operations" in output
    assert "Centrosymmetric            : yes" in output


def test_multi_frame_records_contain_lattice_and_symmetry_columns():
    structures = [
        Atoms(
            "Si2",
            cell=np.diag([5.43, 5.43, 5.43]),
            scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            pbc=True,
        ),
        Atoms("H", positions=[[0.0, 0.0, 0.0]], pbc=False),
    ]

    records = collect_space_group_information(structures)

    assert len(records) == 2
    assert tuple(records[0]) == CSV_COLUMNS
    assert records[0]["n. atoms"] == 2
    assert records[0]["space group symbol"]
    assert records[0]["number of symmetry operations"] > 0
    assert records[1]["n. atoms"] == 1
    assert records[1]["space group"] is None


def test_dataset_statistics_plot_is_written(tmp_path):
    atoms = Atoms(
        "Si2",
        cell=np.diag([5.43, 5.43, 5.43]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        pbc=True,
    )
    dataframe = pd.DataFrame.from_records(
        collect_space_group_information([atoms]), columns=CSV_COLUMNS
    )
    output = tmp_path / "statistics.png"

    plot_dataset_statistics(dataframe, output)

    assert output.exists()
    assert output.stat().st_size > 0
