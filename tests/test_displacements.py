from pathlib import Path

import numpy as np
import pytest

from fd2bec.atomic import AtomicStructure
from fd2bec.io import read
from fd2bec.tensor import Dipole, Vector


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
    vector_symmetrizer, vector_theta, _ = unit_cell.get_symmetrizer(vector)

    dipole = Dipole(
        data=vector_data,
        basis="fractional",
        cell=unit_cell.cell,
    )
    dipole_symmetrizer, dipole_theta, _ = unit_cell.get_symmetrizer(dipole)

    # In reduced coordinates, (1/2, 1/2, 1/2) represents Q/2 along every
    # periodic direction. Inversion removes every invariant mode from an
    # ordinary global vector, whereas a global affine dipole keeps one
    # homogeneous mode.
    assert vector_symmetrizer.shape == (3, 0)
    assert len(vector_theta) == 0
    assert dipole_symmetrizer.shape == (4, 1)
    assert len(dipole_theta) == 1

if __name__ == "__main__":
    pytest.main([__file__])
