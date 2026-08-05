import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from fd2bec.cli.ml.add_oxidation_numbers import (
    add_oxidation_numbers,
    read_oxidation_numbers,
)


def test_add_oxidation_numbers_round_trip_extxyz(tmp_path):
    structures = [Atoms("BaTiO3"), Atoms("BaTiO3")]
    charged = add_oxidation_numbers(structures, {"Ba": 2, "Ti": 4, "O": -2})
    output = tmp_path / "charged.extxyz"
    write(output, charged, format="extxyz")

    restored = read(output, index=":")
    assert len(restored) == 2
    for atoms in restored:
        np.testing.assert_array_equal(atoms.arrays["Qs"], [2, 4, -2, -2, -2])


def test_missing_species_is_reported():
    with pytest.raises(ValueError, match="absent.*O"):
        add_oxidation_numbers([Atoms("MgO")], {"Mg": 2})


def test_non_neutral_structure_requires_opt_in():
    with pytest.raises(ValueError, match="not oxidation-number neutral"):
        add_oxidation_numbers([Atoms("Na")], {"Na": 1})

    charged = add_oxidation_numbers(
        [Atoms("Na")], {"Na": 1}, name="oxidation_numbers", require_neutral=False
    )
    np.testing.assert_array_equal(charged[0].arrays["oxidation_numbers"], [1])


def test_json_mapping_validation_and_fractional_warning(tmp_path):
    mapping = tmp_path / "charges.json"
    mapping.write_text(json.dumps({"H": 0.5}), encoding="utf-8")

    with pytest.warns(UserWarning, match="not an integer"):
        assert read_oxidation_numbers(mapping) == {"H": 0.5}
