import numpy as np
from ase import Atoms

from fd2bec.cli.displacements.extxyz2folder import espresso_geometry, write_snapshots


def periodic_atoms(shift=0.0):
    return Atoms(
        symbols=["Ba", "Ti", "O"],
        cell=np.diag([3.977555743158] * 3),
        scaled_positions=[
            [0.5 + shift, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        pbc=True,
    )


def test_espresso_geometry_contains_cell_and_fractional_positions():
    text = espresso_geometry(periodic_atoms())

    assert text.startswith("CELL_PARAMETERS angstrom\n")
    assert "3.977555743158" in text
    assert "ATOMIC_POSITIONS crystal\n" in text
    assert "Ba   0.500000000000  0.500000000000  0.500000000000" in text


def test_write_snapshots_uses_requested_geometry_names(tmp_path):
    structures = [periodic_atoms(), periodic_atoms(shift=0.01)]

    filenames = write_snapshots(structures, tmp_path, "espresso-in")

    assert [filename.name for filename in filenames] == [
        "geometry.n=0.in",
        "geometry.n=1.in",
    ]
    assert all(filename.is_file() for filename in filenames)
    assert "0.510000000000" in filenames[1].read_text(encoding="utf-8")
