from pathlib import Path

import numpy as np
import pytest

from fd2bec.atomic import AtomicStructure
from fd2bec.io import read
from fd2bec.tensor import Dipole, Displacement, Position, Vector


def test_centrosymmetric_periodic_dipole_needs_reference_configuration():
    """A Berry-phase polarization still needs one reference calculation."""
    path = Path(__file__).parent / "data/BiFeO3-R-3c.geometry.in"
    unit_cell = AtomicStructure.from_ase(read(path, format="aims"))

    assert unit_cell.space_group == 167  # R-3c
    rotations, _ = unit_cell.get_symmetry_operations(basis="fractional")
    inversion = -np.eye(3, dtype=int)
    assert any(np.array_equal(rotation, inversion) for rotation in rotations)

    vector_data = np.full(3, 0.5)
    vector = Vector(
        data=vector_data,
        basis="fractional",
        cell=unit_cell.cell,
    )
    vector_projection, vector_mode_coefficients, vector_modes = unit_cell.get_symmetry_modes(
        vector
    )

    dipole = Dipole(
        data=vector_data,
        basis="fractional",
        cell=unit_cell.cell,
    )
    dipole_projection, dipole_mode_coefficients, dipole_modes = unit_cell.get_symmetry_modes(
        dipole
    )

    # In reduced coordinates, (1/2, 1/2, 1/2) represents Q/2 along every
    # periodic direction. Inversion removes every invariant mode from an
    # ordinary global vector, whereas a global affine dipole keeps one
    # homogeneous mode.
    assert vector_projection.shape == (3, 3)
    assert len(vector_mode_coefficients) == 0
    assert vector_modes.shape == (0, 3)
    assert dipole_projection.shape == (4, 4)
    assert len(dipole_mode_coefficients) == 1
    assert dipole_modes.shape == (1, 3)


def test_position_modes_are_displacement_modes_about_the_reference_structure():
    path = Path(__file__).parent / "data/BiFeO3-R-3c.geometry.in"
    unit_cell = AtomicStructure.from_ase(read(path, format="aims"))
    positions = Position(data=unit_cell.frac_pos, basis="fractional")
    displacements = Displacement(data=np.zeros_like(unit_cell.frac_pos), basis="fractional")

    position_projection, _, position_modes = unit_cell.get_symmetry_modes(positions)
    displacement_projection, _, displacement_modes = unit_cell.get_symmetry_modes(
        displacements
    )

    np.testing.assert_allclose(position_projection, displacement_projection)
    np.testing.assert_allclose(position_modes, displacement_modes)

if __name__ == "__main__":
    pytest.main([__file__])
