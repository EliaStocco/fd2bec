import numpy as np
import pytest
from ase.io import read


def test_read(structure):
    n, file = structure
    atoms = read(file, index=0)
    assert atoms.get_global_number_of_atoms() == np.power(n, 3) * 5, "Wrong number of atoms"


if __name__ == "__main__":
    pytest.main([__file__])
