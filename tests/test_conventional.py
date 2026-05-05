import numpy as np
import pytest
from fd2bec.io import read

from fd2bec.atomic import AtomicStructure


def test_conventional(sg_case):
    dataset, filepath, n = sg_case
    atoms = read(filepath)
    unit_cell = AtomicStructure.from_ase(atoms)
    conventional_cell = unit_cell.conventional

    assert unit_cell.space_group == conventional_cell.space_group, \
        f"Different space groups: {unit_cell.space_group} != {conventional_cell.space_group}."
    assert unit_cell.space_group == n , f"Wrong space group: {unit_cell.space_group} != {n}."

    # if np.allclose(unit_cell.spglib_dataset.origin_shift,0,atol=ATOL):
    #     pytest.skip("origin_shift is trivial.")
    # if np.allclose(unit_cell.spglib_dataset.transformation_matrix,np.eye(3),atol=ATOL):
    #     pytest.skip("transformation_matrix is trivial.")

    # if unit_cell.is_equal_to(conventional_cell):
    #     pytest.skip("Unit cell is already conventional.")

    P = conventional_cell.spglib_dataset.transformation_matrix
    O = conventional_cell.spglib_dataset.origin_shift
    assert np.allclose(P,np.eye(3))
    assert np.allclose(O,0)

    twice_conventional_cell = conventional_cell.conventional
    assert conventional_cell.is_equal_to(twice_conventional_cell), "Twice conventional is not equal to conventional cell."

    # conventional_cell._test_symmetry_pbc_fractional()


if __name__ == "__main__":
    pytest.main([__file__])
