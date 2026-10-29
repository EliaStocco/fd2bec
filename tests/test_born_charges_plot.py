import numpy as np
from ase import Atoms
from ase.io import write

from fd2bec.born_charges_plot import (
    born_charge_from_atoms,
    born_charges_from_atoms,
    directional_born_charge_response,
    induced_polarization_response,
    plot_born_charge_response_comparison,
    plot_born_charge_structure_comparison,
)
from fd2bec.cli.general import plot_bec


def test_born_charge_responses_separate_longitudinal_and_transverse_parts():
    born_charge = np.asarray([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    direction = np.asarray([[2.0**-0.5, 2.0**-0.5, 0.0]])

    polarization = induced_polarization_response(born_charge, direction)
    longitudinal = directional_born_charge_response(born_charge, direction, "longitudinal")
    magnitude = directional_born_charge_response(born_charge, direction, "magnitude")
    transverse = directional_born_charge_response(born_charge, direction, "transverse")

    np.testing.assert_allclose(polarization, [[2.0**0.5, 2.0**-0.5, 0.0]])
    np.testing.assert_allclose(longitudinal, [1.5])
    np.testing.assert_allclose(magnitude, [(2.5) ** 0.5])
    np.testing.assert_allclose(transverse, [0.5])


def test_born_charge_reader_and_cli_plot(tmp_path, capsys):
    atoms = Atoms("HHe", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tensors = np.asarray([np.eye(3), np.diag([2.0, 1.0, 3.0])])
    atoms.set_array("REF_BEC", tensors.reshape((2, 9)))
    input_path = tmp_path / "born-charges.extxyz"
    output_path = tmp_path / "born-charge-components.png"
    write(input_path, atoms, format="extxyz")

    np.testing.assert_allclose(born_charge_from_atoms(atoms, "REF_BEC", 1), tensors[1])
    np.testing.assert_allclose(born_charges_from_atoms(atoms, "REF_BEC"), tensors)
    args = plot_bec.prepare_args(plot_bec.description).parse_args(
        ["-i", str(input_path), "--all-atoms", "-o", str(output_path), "--resolution", "9"]
    )
    plot_bec.main.__wrapped__(args)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert "Transverse response range:" in capsys.readouterr().out


def test_born_charge_comparison_writes_all_surfaces(tmp_path):
    output = tmp_path / "responses.png"
    responses = plot_born_charge_response_comparison(np.eye(3), output, resolution=9)

    assert output.exists()
    assert set(responses) == {"longitudinal", "magnitude", "transverse"}


def test_born_charge_structure_comparison_writes_one_row_per_atom(tmp_path):
    output = tmp_path / "all-atoms.png"
    responses = plot_born_charge_structure_comparison(
        np.asarray([np.eye(3), np.diag([2.0, 1.0, 3.0])]),
        output,
        labels=["atom 0: H", "atom 1: He"],
        resolution=9,
    )

    assert output.exists()
    assert set(responses) == {"longitudinal", "magnitude", "transverse"}
    assert all(response.shape == (2, 9, 19) for response in responses.values())
