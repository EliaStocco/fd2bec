import numpy as np

from fd2bec.cli.general.tensor_symmetries import (
    _symmetric_basis,
    _symmetric_pairs,
    print_independent_components,
    print_symbolic_tensor,
    symbolic_components,
    voigt_components,
)


def test_symbolic_components_use_independent_letters():
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    symbolic, pivots = symbolic_components(basis, (3,))

    assert pivots == [0, 1]
    assert symbolic.tolist() == ["a", "b", "a + b"]


def test_strain_pair_is_symmetrized_and_voigt_compressed():
    axes = [
        {"name": "dipole", "type": "cartesian", "role": "output"},
        {"name": "strain_i", "type": "cartesian", "role": "input"},
        {"name": "strain_j", "type": "cartesian", "role": "input"},
    ]
    basis = np.eye(27)
    pairs = _symmetric_pairs(axes, (3, 3, 3))
    display_basis = _symmetric_basis(basis, (3, 3, 3), pairs)
    symbolic, _ = symbolic_components(display_basis, (3, 3, 3))
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

    print_symbolic_tensor(symbolic, axes)

    output = capsys.readouterr().out
    assert "[atom={0, 1}]" in output
    assert output.count("atom=") == 1
