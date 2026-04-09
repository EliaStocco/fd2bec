import pytest
import numpy as np
from ase.io import read
from fd2bec import ATOL
from fd2bec.tools import allclose_chunked
from fd2bec.conftest import structure
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import homogeneous2affine

def test_structures_spacegroup_positions(structure):
    """
    Test that BaTiO3 structures have the correct number of atoms,
    correct space group, and that atomic positions are symmetric.
    """
    n, file_path = structure

    atoms = read(file_path, index=0)
    Natoms = atoms.get_global_number_of_atoms()
    
    atomic_structure = AtomicStructure.from_ase(atoms)
    assert atomic_structure._test_symmetry(), "Error in AtomicStructure._test_symmetry() method"
    
    R,T = atomic_structure.get_affine_symmetry_operations(debug=True)
    H = atomic_structure.get_homogeneous_symmetry_operations(debug=True)
    Rtest, Ttest = homogeneous2affine(H)
    assert allclose_chunked(Rtest,R,atol=ATOL), "Different rotation matrices"
    assert np.allclose(Ttest,T,atol=ATOL), "Different translation vectors"
    
    S, theta = atomic_structure.get_symmetrizer(debug=True)
    
    pass


if __name__ == "__main__":
    pytest.main([__file__])