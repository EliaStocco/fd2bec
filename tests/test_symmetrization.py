import pytest
import numpy as np
from ase.io import read
from fd2bec import ATOL
from fd2bec.conftest import structures_dir
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import homogeneous2affine

            
def test_structures_spacegroup_positions():
    """
    Test that BaTiO3 structures have the correct number of atoms,
    correct space group, and that atomic positions are symmetric.
    """
    for n in [1, 2, 4]:
        file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
        atoms = read(file_path, index=0)
        
        atoms = AtomicStructure.from_ase(atoms)
        assert atoms._test_symmetry(), "Error in AtomicStructure._test_symmetry() method"
        
        R,T = atoms.get_affine_symmetry_operations(debug=True)
        H = atoms.get_homogeneous_symmetry_operations(debug=True)
        Rtest, Ttest = homogeneous2affine(H)
        assert np.allclose(Rtest,R,atol=ATOL), "Different rotation matrices"
        assert np.allclose(Ttest,T,atol=ATOL), "Different rotation matrices"
        # atoms.get_symmetrizer(debug=True)


if __name__ == "__main__":
    pytest.main([__file__])