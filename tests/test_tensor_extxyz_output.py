import numpy as np
from ase import Atoms

from fd2bec.cli import KEYWORDS
from fd2bec.io import read, write_tensor_extxyz


def test_per_atom_tensor_extxyz_uses_the_default_bec_key(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[2, 2, 2], pbc=True)
    bec = np.arange(18.0).reshape((2, 3, 3))
    output = tmp_path / "bec.extxyz"

    write_tensor_extxyz(output, atoms, bec, KEYWORDS["bec"], per_atom=True)

    saved = read(output, format="extxyz")
    np.testing.assert_allclose(saved.arrays[KEYWORDS["bec"]], bec.reshape((2, 9)))


def test_global_tensor_extxyz_uses_the_default_piezoelectric_key(tmp_path):
    atoms = Atoms("H", cell=[2, 2, 2], pbc=True)
    piezoelectric = np.arange(27.0).reshape((3, 3, 3))
    output = tmp_path / "piezoelectric.extxyz"

    write_tensor_extxyz(
        output,
        atoms,
        piezoelectric,
        KEYWORDS["piezoelectric"],
        per_atom=False,
    )

    saved = read(output, format="extxyz")
    np.testing.assert_allclose(saved.info[KEYWORDS["piezoelectric"]], piezoelectric)
