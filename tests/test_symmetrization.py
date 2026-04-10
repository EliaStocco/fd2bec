import pytest
import numpy as np
from ase.io import read
from fd2bec import ATOL
from fd2bec.tools import allclose_chunked
from fd2bec.conftest import structure
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import append_one, remove_one

def test_symmetrizer(structure):
    """
    Test that the structures have the correct number of atoms,
    correct space group, and that atomic positions are symmetric.
    """
    n, file_path = structure

    atoms = read(file_path, index=0)
    Natoms = atoms.get_global_number_of_atoms()
    if Natoms > 200:
        return 
    
    atomic_structure = AtomicStructure.from_ase(atoms)
    assert atomic_structure._test_symmetry(), "Error in AtomicStructure._test_symmetry() method"
    pos = atomic_structure.frac_pos.flatten()
    
    #----------------------#
    # Positions
    #----------------------#
    # affine
    R,T = atomic_structure.get_affine_symmetry_operations(debug=True)
    new_pos = R @ pos + T 
    assert np.allclose(pos,new_pos,atol=ATOL), "Error with the affine symmetry operations"
    
    # homogeneous
    pos1 = append_one(pos)
    H = atomic_structure.get_homogeneous_symmetry_operations(debug=True)
    new_pos = H @ pos1
    assert np.allclose(pos1,new_pos,atol=ATOL), "Error with the homogeneous symmetry operations"
    
    # symmetrizer
    S, theta, theta_real = atomic_structure.get_symmetrizer(what='positions',debug=True)
    assert len(theta) == 5, "there is something wrong"
    assert np.allclose(remove_one(S @ theta), pos), "Error with the positions symmetrizer."

    #----------------------#
    # Forces
    #----------------------#   
    frac_forces = atomic_structure.get_fractional(atoms.arrays["REF_forces"])
    assert np.allclose(frac_forces @ atomic_structure.cell, atoms.arrays["REF_forces"],atol=ATOL), "Error with fractional forces"
    
    forces = frac_forces.flatten()
    R = atomic_structure.get_symmetry_operations(x=frac_forces,debug=True)
    assert np.allclose(R @ forces, forces,atol=ATOL), "Error with the forces symmetrizer."
    
    S, theta, theta_real = atomic_structure.get_symmetrizer(what='vector',x=frac_forces,debug=True)
    assert np.allclose(S @ theta, forces,atol=ATOL), "Error with the forces symmetrizer."
    
    pass


if __name__ == "__main__":
    pytest.main([__file__])