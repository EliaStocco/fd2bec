import numpy as np
from ase.io import read
import pytest
from fd2bec.conftest import structures_dir

def test_read():
    for n in [1,2,4]:
        file = f"{structures_dir}/BaTiO3.{n}x{n}x{n}.extxyz"
        atoms = read(file,index=0)
        assert atoms.get_global_number_of_atoms() == np.power(n,3)*5, f"Wrong number of atoms"
        
if __name__ == "__main__":
    pytest.main([__file__])