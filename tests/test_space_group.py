import numpy as np
import pandas as pd
from ase import Atoms

from fd2bec.cli.structures.space_group_dataset import (
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
