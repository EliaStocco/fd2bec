from pathlib import Path

import numpy as np
import pytest

from fd2bec import ATOL
from fd2bec.atomic import AtomicStructure
from fd2bec.io import read
from fd2bec.tensor import Dipole, Forces, Position, Vector


@pytest.fixture(scope="module")
def bifeo3():
    path = Path(__file__).parent / "data/BiFeO3-R-3c.geometry.in"
    return AtomicStructure.from_ase(read(path, format="aims"))


@pytest.mark.parametrize(
    "kind", ["atomic-linear", "global-linear", "atomic-affine", "global-affine"]
)
def test_tensor_symmetry_operation_variants(bifeo3, kind):
    if kind == "atomic-linear":
        tensor = Forces(data=np.zeros((len(bifeo3), 3)), basis="fractional")
    elif kind == "global-linear":
        tensor = Vector(data=np.zeros(3), basis="fractional")
    elif kind == "atomic-affine":
        tensor = Position(data=bifeo3.frac_pos, basis="fractional")
    else:
        tensor = Dipole(data=np.full(3, 0.5), basis="fractional")

    rotations, translations = bifeo3.get_tensor_symmetry_operations(tensor)
    flattened = tensor.flatten_full()
    expected_shape = (len(rotations), len(flattened))

    assert rotations.shape == (*expected_shape, len(flattened))
    assert translations.shape == expected_shape

    if any(axis.get("affine", False) for axis in tensor.axes):
        transformed = rotations @ flattened + translations
        expected = np.broadcast_to(flattened, transformed.shape)
        np.testing.assert_allclose(transformed, expected, atol=ATOL)
    else:
        np.testing.assert_allclose(translations, 0)


def test_atomic_symmetry_mapping_uses_cartesian_symprec(monkeypatch):
    displacement = 5e-5

    def approximate_symmetry_operation(self, basis="cartesian"):
        del self, basis
        return np.eye(3)[None, ...], np.array([[displacement, 0.0, 0.0]])

    monkeypatch.setattr(AtomicStructure, "get_symmetry_operations", approximate_symmetry_operation)
    tensor = Forces.template(1, basis="cartesian")

    accepted = AtomicStructure(
        symbols=["H"],
        cell=np.eye(3),
        frac_pos=np.zeros((1, 3)),
        symprec=1e-4,
    )
    accepted.get_tensor_symmetry_operations(tensor)

    rejected = AtomicStructure(
        symbols=["H"],
        cell=np.eye(3),
        frac_pos=np.zeros((1, 3)),
        symprec=1e-5,
    )
    with pytest.raises(ValueError, match="Mapping failed for species H"):
        rejected.get_tensor_symmetry_operations(tensor)
