import numpy as np
from ase import Atoms

from fd2bec.cli.structures.convert_format import (
    is_ase_standard_cell,
    rotate_to_ase_standard_cell,
)


def test_rotate_to_ase_standard_cell_preserves_fractional_positions():
    standard_cell = np.array([[3.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.1, 0.3, 5.0]])
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
    )
    fractional_positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    atoms = Atoms(
        symbols=["Si", "O"],
        cell=standard_cell @ rotation,
        scaled_positions=fractional_positions,
        pbc=True,
    )

    assert not is_ase_standard_cell(atoms)

    rotated = rotate_to_ase_standard_cell(atoms)

    assert is_ase_standard_cell(rotated)
    np.testing.assert_allclose(rotated.cell.array, standard_cell)
    np.testing.assert_allclose(
        rotated.get_scaled_positions(wrap=False), fractional_positions
    )
