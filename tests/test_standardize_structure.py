import numpy as np
import pytest
from ase import Atoms

from fd2bec.cli.structures import convert_format


def test_standardize_structure_requests_the_selected_spglib_setting(monkeypatch):
    atoms = Atoms("Na", cell=np.eye(3), scaled_positions=[[0.0, 0.0, 0.0]], pbc=True)
    calls = []

    def fake_standardize_cell(cell, **kwargs):
        calls.append((cell, kwargs))
        return np.eye(3) * 2.0, np.array([[0.25, 0.25, 0.25]]), np.array([11])

    monkeypatch.setattr(convert_format.spglib, "standardize_cell", fake_standardize_cell)

    result = convert_format.standardize_structure(
        atoms, setting="primitive", symprec=1e-4
    )

    assert calls[0][1] == {"to_primitive": True, "no_idealize": False, "symprec": 1e-4}
    np.testing.assert_allclose(result.cell.array, np.eye(3) * 2.0)
    np.testing.assert_allclose(result.get_scaled_positions(), [[0.25, 0.25, 0.25]])


def test_standardize_structure_requires_a_periodic_input():
    with pytest.raises(ValueError, match="fully periodic"):
        convert_format.standardize_structure(
            Atoms("H"), setting="conventional", symprec=1e-4
        )
