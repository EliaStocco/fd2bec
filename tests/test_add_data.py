import numpy as np
import pytest
from ase import Atoms

from fd2bec.io import add_extxyz_data, read_numeric_data


def test_add_extxyz_info_data():
    structures = [Atoms("H"), Atoms("H")]
    updated = add_extxyz_data(structures, np.array([[1.0, 2.0], [3.0, 4.0]]), "dipole", "info")

    np.testing.assert_allclose(updated[0].info["dipole"], [1.0, 2.0])
    np.testing.assert_allclose(updated[1].info["dipole"], [3.0, 4.0])
    assert "dipole" not in structures[0].info


def test_add_extxyz_array_data_and_replicate():
    structures = [Atoms("H2"), Atoms("H2")]
    updated = add_extxyz_data(structures, [1.0, 2.0], "weights", "a", replicate=True)

    for atoms in updated:
        np.testing.assert_allclose(atoms.arrays["weights"], [[1.0], [2.0]])


def test_add_extxyz_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="cannot be reshaped"):
        add_extxyz_data([Atoms("H2"), Atoms("H2")], [1.0, 2.0, 3.0], "weights", "arrays")

    with pytest.raises(ValueError, match="same atom count"):
        add_extxyz_data([Atoms("H"), Atoms("H2")], [1.0, 2.0, 3.0], "weights", "arrays")


def test_read_numeric_data_accepts_csv_and_scalar(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("1,2\n3,4\n", encoding="utf-8")
    np.testing.assert_allclose(read_numeric_data(csv_file), [[1.0, 2.0], [3.0, 4.0]])

    scalar_file = tmp_path / "scalar.txt"
    scalar_file.write_text("2.5\n", encoding="utf-8")
    np.testing.assert_allclose(read_numeric_data(scalar_file), [2.5])
