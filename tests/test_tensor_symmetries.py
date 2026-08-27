from types import SimpleNamespace

import numpy as np
import pytest

from fd2bec.cli import count_with_percentage
from fd2bec.cli.structures.tensor_symmetries import prepare_args
from fd2bec.tensor_components import (
    _symmetric_basis,
    affine_parameter_values,
    expand_voigt_data,
    flattened_nuclear_position_matrix,
    forbidden_component_indices,
    parameter_name,
    physical_modes,
    print_components,
    print_independent_components,
    print_numeric_tensor,
    selected_tensor_basis,
    selected_tensor_precision,
    symbolic_affine_components,
    symbolic_components,
    symmetric_pairs,
    voigt_components,
)
from fd2bec.tools import tensor_data_from_atoms


def test_count_with_percentage_reports_selected_and_total_counts():
    assert count_with_percentage(2, 9) == "2 out of 9 (22.2%)"


def test_tensor_keyword_is_optional():
    parser = prepare_args("test")

    without_keyword = parser.parse_args(["-i", "structure.xyz", "-n", "bec"])
    with_keyword = parser.parse_args(
        [
            "-i",
            "structure.xyz",
            "-n",
            "bec",
            "--keyword",
            "MACE_BEC",
            "--precision",
            "6",
        ]
    )

    assert without_keyword.keyword is None
    assert without_keyword.precision is None
    assert with_keyword.keyword == "MACE_BEC"
    assert with_keyword.precision == 6


def test_tensor_data_is_found_in_arrays_info_and_split_bec_fields():
    atoms = SimpleNamespace(
        arrays={
            "forces": np.ones((2, 3)),
            "becx": np.ones((2, 3)),
            "becy": np.ones((2, 3)) * 2,
            "becz": np.ones((2, 3)) * 3,
        },
        info={"dipole": [1.0, 2.0, 3.0]},
    )
    forces, forces_location = tensor_data_from_atoms(atoms, "forces", "forces")
    dipole, dipole_location = tensor_data_from_atoms(atoms, "dipole", "dipole")

    np.testing.assert_array_equal(forces, np.ones((2, 3)))
    assert forces_location == "atoms.arrays"
    np.testing.assert_array_equal(dipole, [1.0, 2.0, 3.0])
    assert dipole_location == "atoms.info"

    # SimpleNamespace special methods are looked up on the type, so use an
    # actual small fake for the split-array convention.
    class FakeAtoms:
        def __init__(self):
            self.arrays = atoms.arrays
            self.info = atoms.info

        def __len__(self):
            return 2

    bec, bec_location = tensor_data_from_atoms(FakeAtoms(), "bec", "bec")
    assert bec.shape == (2, 3, 3)
    np.testing.assert_array_equal(bec[:, :, 0], np.ones((2, 3)))
    np.testing.assert_array_equal(bec[:, :, 1], np.ones((2, 3)) * 2)
    np.testing.assert_array_equal(bec[:, :, 2], np.ones((2, 3)) * 3)
    assert "split" in bec_location


def test_standard_ase_stress_voigt_value_is_expanded():
    from fd2bec.tensor import Stress

    expanded = expand_voigt_data(np.arange(1.0, 7.0), Stress.template())

    np.testing.assert_array_equal(
        expanded,
        [[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]],
    )


def test_numeric_tensor_prints_independent_values_and_checks_zeros(capsys):
    from fd2bec.tensor import Vector

    tensor = Vector(data=np.asarray([2.54321, 1e-7, 0.0]))
    print_numeric_tensor(
        tensor,
        "prediction",
        "atoms.info",
        [0],
        np.asarray(["a", "0", "0"]),
        frame_label="input",
        precision=4,
    )

    output = capsys.readouterr().out
    assert "Symmetry-inequivalent component values:" in output
    assert "a: 2.543" in output
    assert "Zero-component check: PASS" in output


def test_nonzero_symmetry_forbidden_components_are_rejected():
    violations, forbidden_count = forbidden_component_indices(
        [2.5, 0.1, 0.0], ["a", "0", "0"]
    )

    assert violations.tolist() == [1]
    assert forbidden_count == 2

    from fd2bec.tensor import Vector

    with pytest.raises(ValueError, match="Symmetry-forbidden tensor components are non-zero"):
        print_numeric_tensor(
            Vector(data=np.asarray([2.5, 0.1, 0.0])),
            "prediction",
            "atoms.info",
            [0],
            np.asarray(["a", "0", "0"]),
            frame_label="input",
        )


def test_affine_numeric_values_are_reported_as_parameters_not_raw_pivots():
    reference = np.array(
        [
            [0.0, 0.0, 0.4999137],
            [0.5, 0.5, 0.01527403],
            [0.0, 0.5, 0.98200336],
            [0.5, 0.0, 0.98200336],
            [0.5, 0.5, 0.47140009],
        ]
    )
    modes = np.zeros((4, 5, 3))
    modes[0, 0, 2] = 1.0
    modes[1, 1, 2] = 1.0
    modes[2, 2:4, 2] = 1.0
    modes[3, 4, 2] = 1.0

    values = affine_parameter_values(reference, reference, modes, fractional=True)

    np.testing.assert_allclose(values, [0.0000863, 0.01527403, 0.01799664, 0.02859991])


def test_affine_fixed_zero_strings_are_checked():
    violations, forbidden_count = forbidden_component_indices(
        [1.0, 0.1, 0.5], ["a", "0.0", "0.5"]
    )

    assert violations.tolist() == [1]
    assert forbidden_count == 1


def test_symbolic_components_use_independent_letters():
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    symbolic, pivots = symbolic_components(basis.T)

    assert pivots == [0, 1]
    assert symbolic.tolist() == ["a", "b", "a + b"]


def test_symbolic_parameter_names_continue_with_numbered_alphabets():
    assert [parameter_name(index) for index in (0, 25, 26, 51, 52, 77)] == [
        "a",
        "z",
        "a1",
        "z1",
        "a2",
        "z2",
    ]


def test_symbolic_components_with_no_modes_returns_zero_tensor():
    symbolic, pivots = symbolic_components(np.empty((0, 2, 3)))

    assert pivots == []
    assert symbolic.tolist() == [["0", "0", "0"], ["0", "0", "0"]]


def test_symbolic_affine_components_show_ideal_fractional_coordinates_and_displacements():
    reference = np.array(
        [
            [0.0, 0.0, 0.4999137],
            [0.5, 0.5, 0.01527403],
            [0.0, 0.5, 0.98200336],
            [0.5, 0.0, 0.98200336],
            [0.5, 0.5, 0.47140009],
        ]
    )
    modes = np.zeros((4, 5, 3))
    modes[0, 0, 2] = 1.0
    modes[1, 1, 2] = 1.0
    modes[2, 2:4, 2] = 1.0
    modes[3, 4, 2] = 1.0

    components, _ = symbolic_affine_components(reference, modes, fractional=True)

    assert components.tolist() == [
        ["0.0", "0.0", "0.5 - a"],
        ["0.5", "0.5", "b"],
        ["0.0", "0.5", "-c"],
        ["0.5", "0.0", "-c"],
        ["0.5", "0.5", "0.5 - d"],
    ]


def test_physical_modes_discards_homogeneous_only_affine_mode():
    component_modes = np.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])

    modes = physical_modes(component_modes, (1, 3), affine=True)

    np.testing.assert_allclose(modes, [[[3.0, 4.0, 0.0]]])


def test_tensor_basis_defaults_to_fractional_only_for_positions():
    assert selected_tensor_basis("positions", None) == "fractional"
    assert selected_tensor_basis("forces", None) == "cartesian"
    assert selected_tensor_basis("positions", "cartesian") == "cartesian"


def test_tensor_precision_defaults_only_for_positions():
    assert selected_tensor_precision("positions", None) == 4
    assert selected_tensor_precision("bec", None) is None
    assert selected_tensor_precision("positions", 7) == 7


def test_tensor_print_components_accepts_symbolic_components(capsys):
    from fd2bec.tensor import BornCharges

    tensor = BornCharges(data=np.zeros((1, 3, 3)))
    symbolic = np.full((1, 3, 3), "a", dtype=object)

    tensor.print_components(symbolic)

    assert "[atom=0]" in capsys.readouterr().out


def test_tensor_print_components_displays_symmetric_strain_axes_in_voigt_notation(capsys):
    from fd2bec.tensor import ProperPiezoelectricTensor

    tensor = ProperPiezoelectricTensor(data=np.zeros((3, 3, 3)))
    tensor.print_components(np.full((3, 3, 3), "a", dtype=object))

    assert "Voigt notation:" in capsys.readouterr().out


def test_print_components_aligns_labels_with_numeric_columns(capsys):
    components = np.zeros((3, 6), dtype=int)
    axes = [
        {"name": "row", "type": "cartesian"},
        {"name": "voigt", "type": "voigt"},
    ]

    print_components(components, axes)

    header, first_row, *_ = capsys.readouterr().out.splitlines()
    assert header.index("xx") + len("xx") == first_row.index("0") + 1
    assert header.index("xy") + len("xy") == first_row.rindex("0") + 1


def test_print_components_displays_vector_labels_above_values(capsys):
    components = np.asarray(["a", "a", "b", "0", "0", "0"])
    axes = [{"name": "voigt", "type": "voigt"}]

    print_components(components, axes)

    header, values = capsys.readouterr().out.splitlines()
    assert header.index("xx") < values.index("a")
    assert header.index("xy") < values.rindex("0")


def test_force_constants_are_printed_as_a_flattened_nuclear_coordinate_matrix(capsys):
    from fd2bec.tensor import ForceConstants

    components = np.arange(36).reshape((2, 2, 3, 3))
    matrix, axes = flattened_nuclear_position_matrix(components, ForceConstants.template(2).axes)

    assert axes[0]["labels"] == ["0x", "0y", "0z", "1x", "1y", "1z"]
    np.testing.assert_array_equal(matrix[0], [0, 1, 2, 9, 10, 11])

    ForceConstants(data=np.zeros((2, 2, 3, 3))).print_components(components)
    output = capsys.readouterr().out
    assert "Flattened nuclear-coordinate matrix (atom-major):" in output
    assert "0x" in output and "1z" in output


def test_strain_pair_is_symmetrized_and_voigt_compressed():
    axes = [
        {"name": "dipole", "type": "cartesian", "role": "output"},
        {"name": "strain_i", "type": "cartesian", "role": "input"},
        {"name": "strain_j", "type": "cartesian", "role": "input"},
    ]
    basis = np.eye(27)
    pairs = symmetric_pairs(axes, (3, 3, 3))
    display_basis = _symmetric_basis(basis, (3, 3, 3), pairs)
    modes = basis.T.reshape((27, 3, 3, 3))
    symbolic, _ = symbolic_components(modes, axes=axes)
    voigt, _ = voigt_components(symbolic, axes, pairs)

    assert pairs == [(1, 2)]
    assert display_basis.shape == (27, 18)
    assert voigt.shape == (3, 6)
    np.testing.assert_array_equal(voigt[:, 0], symbolic[:, 0, 0])


def test_non_voigt_tensors_print_inequivalent_components(capsys):
    axes = [
        {"name": "atom", "type": "atomic"},
        {"name": "dipole", "type": "cartesian"},
        {"name": "position", "type": "cartesian"},
    ]

    print_independent_components([0, 4], (2, 3, 3), axes)

    output = capsys.readouterr().out
    assert "Symmetry-inequivalent components:" in output
    assert "a = atom=0, dipole=x, position=x" in output
    assert "b = atom=0, dipole=y, position=y" in output


def test_equal_atomic_blocks_are_printed_once(capsys):
    axes = [
        {"name": "atom", "type": "atomic"},
        {"name": "dipole", "type": "cartesian"},
        {"name": "position", "type": "cartesian"},
    ]
    block = np.asarray([["a", "0", "0"], ["0", "a", "0"], ["0", "0", "a"]])
    symbolic = np.stack([block, block])

    print_components(symbolic, axes)

    output = capsys.readouterr().out
    assert "[atom={0, 1}]" in output
    assert output.count("atom=") == 1
