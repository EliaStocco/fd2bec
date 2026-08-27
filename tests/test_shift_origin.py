import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from fd2bec.cli.structures import shift_origin
from fd2bec.tools import shift_first_atom_to_origin


def _minimum_image_displacements(atoms):
    fractional = atoms.get_scaled_positions(wrap=False)
    displacement = fractional[:, None, :] - fractional[None, :, :]
    return displacement - np.floor(displacement + 0.5)


def test_shift_first_atom_to_origin_preserves_periodic_structure_and_metadata():
    atoms = Atoms(
        "BaTiO",
        cell=[[4.0, 0.0, 0.0], [0.2, 4.1, 0.0], [0.1, 0.3, 4.2]],
        scaled_positions=[[0.25, 0.5, 0.75], [0.75, 0.25, 0.1], [0.1, 0.9, 0.8]],
        pbc=True,
    )
    atoms.info["source"] = "original"
    atoms.set_array("labels", np.asarray([10, 20, 30]))
    original_positions = atoms.get_positions().copy()

    shifted = shift_first_atom_to_origin(atoms)

    np.testing.assert_array_equal(atoms.get_positions(), original_positions)
    np.testing.assert_array_equal(shifted.cell.array, atoms.cell.array)
    np.testing.assert_array_equal(shifted.numbers, atoms.numbers)
    np.testing.assert_array_equal(shifted.arrays["labels"], atoms.arrays["labels"])
    assert shifted.info == atoms.info
    np.testing.assert_allclose(
        shifted.get_scaled_positions(wrap=False),
        [[0.0, 0.0, 0.0], [0.5, 0.75, 0.35], [0.85, 0.4, 0.05]],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        _minimum_image_displacements(shifted),
        _minimum_image_displacements(atoms),
        atol=1e-14,
    )


def test_shift_first_atom_to_origin_requires_a_nonempty_periodic_structure():
    with pytest.raises(ValueError, match="fully periodic"):
        shift_first_atom_to_origin(Atoms("H"))
    with pytest.raises(ValueError, match="empty"):
        shift_first_atom_to_origin(Atoms(cell=np.eye(3), pbc=True))


def test_shift_origin_cli_writes_the_shifted_copy(tmp_path):
    input_path = tmp_path / "input.extxyz"
    output_path = tmp_path / "shifted.extxyz"
    atoms = Atoms(
        "NaCl",
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        pbc=True,
    )
    write(input_path, atoms)
    args = shift_origin.prepare_args(shift_origin.description).parse_args(
        ["-i", str(input_path), "-o", str(output_path)]
    )

    shift_origin.main.__wrapped__(args)

    shifted = read(output_path)
    np.testing.assert_allclose(
        shifted.get_scaled_positions(wrap=False), [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
