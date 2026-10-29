import numpy as np
from ase import Atoms
from ase.io import read, write

from fd2bec.born_charges_visualization import (
    attach_born_charge_visualization_data,
    born_charge_visualization_data,
)
from fd2bec.cli.general import export_bec_visualization


def test_bec_visualization_data_separates_directional_components():
    tensors = np.asarray([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]])

    direction, data = born_charge_visualization_data(tensors, [1.0, 1.0, 0.0])

    np.testing.assert_allclose(direction, [2.0**-0.5, 2.0**-0.5, 0.0])
    np.testing.assert_allclose(data["response"], [[2.0**0.5, 2.0**-0.5, 0.0]])
    np.testing.assert_allclose(data["longitudinal"], [1.5])
    np.testing.assert_allclose(data["magnitude"], [(2.5) ** 0.5])
    np.testing.assert_allclose(data["transverse"], [0.5])


def test_bec_visualization_cli_writes_response_vector_properties(tmp_path, capsys):
    atoms = Atoms("HHe", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_array("REF_BEC", np.asarray([np.eye(3), np.diag([2.0, 1.0, 3.0])]).reshape((2, 9)))
    input_path = tmp_path / "input.extxyz"
    output_path = tmp_path / "visualization.extxyz"
    write(input_path, atoms, format="extxyz")

    args = export_bec_visualization.prepare_args(export_bec_visualization.description).parse_args(
        ["-i", str(input_path), "-o", str(output_path), "--direction", "1", "0", "0"]
    )
    export_bec_visualization.main.__wrapped__(args)
    exported = read(output_path, format="extxyz")

    np.testing.assert_allclose(exported.arrays["BEC_response"], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(exported.arrays["BEC_longitudinal"], [1.0, 2.0])
    np.testing.assert_allclose(exported.arrays["BEC_transverse"], [0.0, 0.0])
    assert "OVITO-ready extxyz" in capsys.readouterr().out


def test_attach_bec_visualization_keeps_original_structure_unchanged():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.set_array("REF_BEC", np.eye(3).reshape((1, 9)))

    exported, _ = attach_born_charge_visualization_data(atoms, "REF_BEC", [0.0, 0.0, 1.0])

    assert "BEC_response" not in atoms.arrays
    assert exported.arrays["BEC_tensor"].shape == (1, 9)
