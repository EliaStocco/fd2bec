from pathlib import Path

import numpy as np
import pytest
from fd2bec.io import read

from fd2bec import ATOL
from fd2bec.atomic import AtomicStructure

DATA_DIR = Path(__file__).parent / "MP/spacegroup_structures"
# DATA_DIR = Path(__file__).parent / "MP/problems"


@pytest.mark.parametrize("n", range(230))
def test_conventional(n):
    pattern = DATA_DIR / f"SG_{n}_mp-*.cif"
    files = list(pattern.parent.glob(pattern.name))

    if not files:
        pytest.skip(f"No files found for space group {n}")

    assert len(files) == 1, f"Expected exactly one file matching {pattern}, but found {len(files)}"
    filepath = files[0]
    atoms = read(filepath)
    unit_cell = AtomicStructure.from_ase(atoms,kwargs={"symprec":1e-1})
    conventional_cell = unit_cell.conventional
    
    assert unit_cell.space_group == conventional_cell.space_group, \
        f"Different space groups: {unit_cell.space_group} != {conventional_cell.space_group}."
    # assert unit_cell.space_group == n , f"Wrong space group: {unit_cell.space_group} != {n}."
        
    P = conventional_cell.spglib_dataset.transformation_matrix
    O = conventional_cell.spglib_dataset.origin_shift
    assert np.allclose(P,np.eye(3))
    assert np.allclose(O,0)
    
    twice_conventional_cell = conventional_cell.conventional
    assert conventional_cell.is_equal_to(twice_conventional_cell), "Twice conventional is not equal to conventional cell."
    
    # conventional_cell._test_symmetry()


if __name__ == "__main__":
    pytest.main([__file__])
