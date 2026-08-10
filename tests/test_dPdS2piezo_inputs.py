import warnings

import numpy as np
import pytest
from ase import Atoms

from fd2bec.cli import KEYWORDS
from fd2bec.cli.dPdS.dPdS2piezo import (
    dipoles_to_polarizations,
    prepare_args,
    validate_clamped_coordinates,
)


def structure_with_dipole(cell, dipole):
    atoms = Atoms("H", cell=cell, pbc=True)
    atoms.info[KEYWORDS["dipole"]] = np.asarray(dipole)
    return atoms


def test_dipoles_are_converted_using_each_cell_volume():
    dipole = np.array([0.1, -0.2, 0.3])
    structures = [
        structure_with_dipole([2, 3, 4], dipole),
        structure_with_dipole([3, 4, 5], dipole),
    ]

    polarizations = dipoles_to_polarizations(structures)

    np.testing.assert_allclose(polarizations[0], dipole / 24)
    np.testing.assert_allclose(polarizations[1], dipole / 60)


def test_missing_requested_vector_is_reported():
    with pytest.raises(ValueError, match="REF_dipole"):
        dipoles_to_polarizations([Atoms("H", cell=[2, 2, 2], pbc=True)])


def fractional_structure(position):
    return Atoms("H", scaled_positions=[position], cell=[2, 3, 4], pbc=True)


def test_clamped_flag_defaults_true_and_has_negative_form():
    parser = prepare_args("test")

    assert parser.parse_args(["-i", "dataset.extxyz"]).clamped is True
    assert parser.parse_args(["-i", "dataset.extxyz", "--no-clamped"]).clamped is False


def test_clamped_dataset_requires_identical_fractional_coordinates():
    reference = fractional_structure([0.1, 0.2, 0.3])
    structures = [reference.copy(), fractional_structure([0.1, 0.2, 0.31])]

    with pytest.raises(ValueError, match="--no-clamped"):
        validate_clamped_coordinates(structures, reference, clamped=True)


def test_relaxed_dataset_warns_when_fractional_coordinates_are_unchanged():
    reference = fractional_structure([0.1, 0.2, 0.3])

    with pytest.warns(UserWarning, match="same fractional coordinates"):
        validate_clamped_coordinates([reference.copy()], reference, clamped=False)


def test_relaxed_dataset_accepts_changed_fractional_coordinates_without_warning():
    reference = fractional_structure([0.1, 0.2, 0.3])
    structures = [fractional_structure([0.1, 0.2, 0.31])]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not validate_clamped_coordinates(structures, reference, clamped=False)
