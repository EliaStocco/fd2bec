from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from fd2bec import ATOL
from fd2bec.atomic import AtomicStructure

DATA_DIR = Path(__file__).parent / "MP/spacegroup_structures"
# DATA_DIR = Path(__file__).parent / "MP/problems"


@pytest.mark.parametrize("n", range(230))
def test_symmetry_MP(n):
    pattern = DATA_DIR / f"SG_{n}_mp-*.cif"
    files = list(pattern.parent.glob(pattern.name))

    if not files:
        pytest.skip(f"No files found for space group {n}")

    assert len(files) == 1, f"Expected exactly one file matching {pattern}, but found {len(files)}"
    filepath = files[0]
    atoms = read(filepath)
    unit_cell = AtomicStructure.from_ase(atoms)
    if not np.allclose(unit_cell.cell.array, atoms.cell.array, atol=ATOL):
        raise ValueError("There is a problem with the cell.")
    try:
        unit_cell._test_symmetry(atol=ATOL * len(unit_cell))
    except Exception as e:
        unit_cell._test_symmetry(atol=ATOL * len(unit_cell))
        raise e


if __name__ == "__main__":
    pytest.main([__file__])
