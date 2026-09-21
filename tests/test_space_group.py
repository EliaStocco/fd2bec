import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write

from fd2bec.cli.symmetry import space_group as space_group_cli
from fd2bec.cli.symmetry.space_group_dataset import (
    CSV_COLUMNS,
    collect_space_group_information,
    plot_dataset_statistics,
)
from fd2bec.tools import ase2spglib_dataset


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


def test_space_group_cli_prints_named_point_group_generators(tmp_path, capsys):
    input_path = tmp_path / "cubic.extxyz"
    atoms = Atoms(
        "BaTiO3",
        cell=np.eye(3) * 4.0,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        pbc=True,
    )
    write(input_path, atoms)
    args = space_group_cli.prepare_args(space_group_cli.description).parse_args(
        ["-i", str(input_path)]
    )

    space_group_cli.main.__wrapped__(args)

    output = capsys.readouterr().out
    assert "Point-group generators (" in output
    assert "1) 90-degree rotation about z axis" in output
    assert "2) mirror plane normal to [0 1 -1] direction" in output
    assert "    R:" in output
    assert "Character table:" in output
    assert "6 x C4" in output
    assert "6 x S4" in output
    table_output = output.split("Character table:", 1)[1].split(
        "Entries are characters", 1
    )[0]
    assert "..." not in table_output
