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
    flattened = tensor.flatten(full=True)
    expected_shape = (len(rotations), len(flattened))

    assert rotations.shape == (*expected_shape, len(flattened))
    assert translations.shape == expected_shape

    if tensor.is_affine:
        transformed = rotations @ flattened + translations
        expected = np.broadcast_to(flattened, transformed.shape)
        np.testing.assert_allclose(transformed, expected, atol=ATOL)
    else:
        np.testing.assert_allclose(translations, 0)
