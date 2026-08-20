import numpy as np

from fd2bec.cli.structures.tensor_symmetries import _physical_modes, _selected_basis
from fd2bec.tensor_components import (
    _symmetric_basis,
    add_affine_reference,
    common_symbolic_components,
    flattened_nuclear_position_matrix,
    print_components,
    print_independent_components,
    symbolic_affine_components,
    symbolic_components,
    symmetric_pairs,
    voigt_components,
)


def test_symbolic_components_use_independent_letters():
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    symbolic, pivots = symbolic_components(basis.T)

    assert pivots == [0, 1]
    assert symbolic.tolist() == ["a", "b", "a + b"]


def test_symbolic_components_with_no_modes_returns_zero_tensor():
    symbolic, pivots = symbolic_components(np.empty((0, 2, 3)))

    assert pivots == []
    assert symbolic.tolist() == [["0", "0", "0"], ["0", "0", "0"]]


def test_common_symbolic_components_keep_symbols_consistent_between_spaces():
    first = np.asarray([[1.0, 1.0, 0.0]])
    second = np.asarray([[0.0, 1.0, 1.0]])

    symbolic, common_pivots, parameter_indices = common_symbolic_components(
        [first, second]
    )

    assert common_pivots == [0, 1]
    assert parameter_indices == [[0], [1]]
    assert symbolic[0].tolist() == ["a", "a", "0"]
    assert symbolic[1].tolist() == ["0", "b", "b"]


def test_add_affine_reference_preserves_shared_parameter_names():
    reference = np.asarray([0.5, 0.982, 0.0])
    symbolic = np.asarray(["a", "-a", "0"])

    result = add_affine_reference(reference, symbolic, fractional=True)

    assert result.tolist() == ["0.5 + a", "-a", "0.0"]


def test_add_affine_reference_retains_fixed_cartesian_coordinates():
    reference = np.asarray([1.25, 2.5])
    symbolic = np.asarray(["a", "0"])

    result = add_affine_reference(reference, symbolic)

    assert result.tolist() == ["a", "2.5"]


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

    modes = _physical_modes(component_modes, (1, 3), affine=True)

    np.testing.assert_allclose(modes, [[[3.0, 4.0, 0.0]]])


def test_tensor_basis_defaults_to_fractional_only_for_positions():
    assert _selected_basis("positions", None) == "fractional"
    assert _selected_basis("forces", None) == "cartesian"
    assert _selected_basis("positions", "cartesian") == "cartesian"


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
