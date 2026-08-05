import numpy as np
import pytest
from ase import Atoms

from fd2bec.cli import KEYWORDS
from fd2bec.cli.dPdS.dPdS2piezo import attach_polarizations
from fd2bec.piezoelectric import E_PER_ANGSTROM2_TO_C_PER_M2


def structure_with_dipole(cell, polarization):
    atoms = Atoms("H", cell=cell, pbc=True)
    atoms.info[KEYWORDS["dipole"]] = np.asarray(polarization) * atoms.get_volume()
    return atoms


def test_dipole_only_structures_are_converted_using_each_volume():
    expected = np.array([0.1, -0.2, 0.3])
    structures = [
        structure_with_dipole([2, 3, 4], expected),
        structure_with_dipole([3, 4, 5], expected),
    ]

    converted = attach_polarizations(structures)

    assert converted == 2
    for atoms in structures:
        np.testing.assert_allclose(atoms.info[KEYWORDS["polarization"]], expected)


def test_existing_polarization_is_preferred_in_auto_mode():
    atoms = structure_with_dipole([2, 3, 4], [1, 1, 1])
    atoms.info[KEYWORDS["polarization"]] = [4, 5, 6]

    assert attach_polarizations([atoms]) == 0
    np.testing.assert_allclose(atoms.info[KEYWORDS["polarization"]], [4, 5, 6])


def test_explicit_dipole_mode_and_si_conversion():
    atoms = structure_with_dipole([2, 3, 4], [0.1, 0.2, 0.3])
    atoms.info[KEYWORDS["polarization"]] = [9, 9, 9]

    attach_polarizations([atoms], quantity="dipole", polarization_unit="C/m^2")

    np.testing.assert_allclose(
        atoms.info[KEYWORDS["polarization"]],
        np.array([0.1, 0.2, 0.3]) * E_PER_ANGSTROM2_TO_C_PER_M2,
    )


def test_missing_requested_vector_is_reported():
    with pytest.raises(ValueError, match="REF_dipole"):
        attach_polarizations([Atoms("H", cell=[2, 2, 2], pbc=True)], quantity="dipole")
