import pytest
import numpy as np
from ase.io import read
from fd2bec import ATOL
from fd2bec.tools import allclose_chunked
from fd2bec.conftest import structure
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import append_one, remove_one

def test_atomic_structure_core(structure):
    n, file_path = structure

    atoms = read(file_path, index=0)
    Natoms = atoms.get_global_number_of_atoms()
    if Natoms > 200:
        return

    atomic_structure = AtomicStructure.from_ase(atoms)

    assert atomic_structure._test_symmetry(), \
        "AtomicStructure symmetry check failed"
        
def test_positions_affine_and_homogeneous(structure):
    n, file_path = structure

    atoms = read(file_path, index=0)
    atomic_structure = AtomicStructure.from_ase(atoms)

    pos = atomic_structure.frac_pos.flatten()

    # ----------------------
    # affine symmetry
    # ----------------------
    R, T = atomic_structure.get_affine_symmetry_operations(debug=True)
    new_pos = R @ pos + T

    assert np.allclose(pos, new_pos, atol=ATOL*len(atomic_structure)), \
        "Affine symmetry failed for positions"

    # ----------------------
    # homogeneous symmetry
    # ----------------------
    pos1 = append_one(pos)
    H = atomic_structure.get_homogeneous_symmetry_operations(debug=True)

    new_pos = H @ pos1

    assert np.allclose(pos1, new_pos, atol=ATOL*len(atomic_structure)), \
        "Homogeneous symmetry failed for positions"
        
        
def test_position_symmetrizer(structure):
    n, file_path = structure

    atoms = read(file_path, index=0)
    atomic_structure = AtomicStructure.from_ase(atoms)

    pos = atomic_structure.frac_pos.flatten()

    S, theta, theta_real = atomic_structure.get_symmetrizer(
        what='positions',
        debug=True
    )

    assert np.allclose(remove_one(S @ theta), pos, atol=ATOL*len(atomic_structure)), \
        "Position symmetrizer reconstruction failed"
        
@pytest.mark.parametrize("name", ["positions"]) #  "REF_atomic-oxn-dipole"
def test_affine_fields(structure, name):
    n, file_path = structure

    atoms = read(file_path, index=0)
    atomic_structure = AtomicStructure.from_ase(atoms)

    frac = atomic_structure.get_fractional(atoms.arrays[name])

    assert np.allclose(
        frac @ atomic_structure.cell,
        atoms.arrays[name],
        atol=ATOL*len(atomic_structure)
    ), f"Fractional conversion failed for {name}"

    x = frac.flatten()

    pos1 = append_one(x)
    H = atomic_structure.get_homogeneous_symmetry_operations(
        x=frac,
        debug=True
    )

    new_x = H @ pos1

    assert np.allclose(pos1, new_x, atol=ATOL*len(atomic_structure)), \
        f"Homogeneous symmetry failed for {name}"

    S, theta, _ = atomic_structure.get_symmetrizer(
        what='positions',
        x=frac,
        debug=True
    )

    assert np.allclose(remove_one(S @ theta), x, atol=ATOL*len(atomic_structure)), \
        f"Symmetrizer failed for {name}"
        
# @pytest.mark.parametrize("name", ["REF_forces"])
# def test_vector_symmetry(structure, name):
#     n, file_path = structure

#     atoms = read(file_path, index=0)
#     atomic_structure = AtomicStructure.from_ase(atoms)

#     frac = atomic_structure.get_fractional(atoms.arrays[name])

#     assert np.allclose(
#         frac @ atomic_structure.cell,
#         atoms.arrays[name],
#         atol=ATOL*len(atomic_structure)
#     ), f"Fractional conversion failed for {name}"

#     x = frac.flatten()

#     # ----------------------
#     # symmetry operators
#     # ----------------------
#     R = atomic_structure.get_symmetry_operations(x=frac, debug=True)

#     assert np.allclose(R @ x, x, atol=ATOL*len(atomic_structure)), \
#         f"Vector symmetry failed for {name}"

#     # ----------------------
#     # symmetrizer
#     # ----------------------
#     S, theta, _ = atomic_structure.get_symmetrizer(
#         what='vector',
#         x=frac,
#         debug=True
#     )

#     assert np.allclose(S @ theta, x, atol=ATOL*len(atomic_structure)), \
#         f"Vector symmetrizer failed for {name}"
        
# def test_born_charges(structure):
#     n, file_path = structure

#     atoms = read(file_path, index=0)
#     atomic_structure = AtomicStructure.from_ase(atoms)

#     if "REF_BEC" not in atoms.arrays:
#         return

#     bec = atomic_structure.get_fractional(atoms.arrays["REF_BEC"])
#     x = bec.flatten()

#     R = atomic_structure.get_symmetry_operations(x=bec, debug=True)

#     assert np.allclose(R @ x, x, atol=ATOL*len(atomic_structure)), \
#         "Born charge symmetry failed"




if __name__ == "__main__":
    pytest.main([__file__])