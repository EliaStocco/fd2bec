import numpy as np
import pytest
from ase import Atoms

from fd2bec.cli import KEYWORDS
from fd2bec.io import add_born_effective_charges, add_proper_piezoelectric_tensors


def test_add_becs_uses_default_key_and_flat_extxyz_shape():
    becs = np.arange(18.0).reshape((2, 9))
    updated = add_born_effective_charges([Atoms("H2")], becs)

    np.testing.assert_allclose(updated[0].arrays[KEYWORDS["bec"]], becs)


def test_add_becs_can_replicate_one_tensor_over_trajectory():
    becs = np.arange(18.0).reshape((2, 9))
    updated = add_born_effective_charges([Atoms("H2"), Atoms("H2")], becs, replicate=True)

    for atoms in updated:
        np.testing.assert_allclose(atoms.arrays[KEYWORDS["bec"]], becs)


def test_add_proper_piezoelectric_uses_default_key():
    piezoelectric = np.arange(18.0).reshape((3, 6))
    updated = add_proper_piezoelectric_tensors([Atoms("H")], piezoelectric)

    np.testing.assert_allclose(updated[0].info[KEYWORDS["piezoelectric"]], piezoelectric)


def test_add_proper_piezoelectric_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected 18"):
        add_proper_piezoelectric_tensors([Atoms("H")], np.zeros((3, 3)))
