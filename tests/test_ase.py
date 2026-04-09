import numpy as np
from ase.io import read
import pytest
from fd2bec.conftest import structure

def test_read(structure):
    n, file = structure
    atoms = read(file,index=0)
    assert atoms.get_global_number_of_atoms() == np.power(n,3)*5, f"Wrong number of atoms"
        
if __name__ == "__main__":
    pytest.main([__file__])