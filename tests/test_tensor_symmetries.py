import numpy as np

from fd2bec.tensor_components import (
    _symmetric_basis,
    print_components,
    print_independent_components,
    symmetric_pairs,
    symbolic_components,
    voigt_components,
)


def test_symbolic_components_use_independent_letters():
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    symbolic, pivots = symbolic_components(basis.T)

    assert pivots == [0, 1]
    assert symbolic.tolist() == ["a", "b", "a + b"]


def test_tensor_print_components_accepts_symbolic_components(capsys):
    from fd2bec.tensor import BornCharges

    tensor = BornCharges(data=np.zeros((1, 3, 3)))
    symbolic = np.full((1, 3, 3), "a", dtype=object)

    tensor.print_components(symbolic)

    assert "[atom=0]" in capsys.readouterr().out


def test_tensor_print_components_can_include_voigt_notation(capsys):
    from fd2bec.tensor import ProperPiezoelectricTensor

    tensor = ProperPiezoelectricTensor(data=np.zeros((3, 3, 3)))
    tensor.print_components(np.full((3, 3, 3), "a", dtype=object), voigt=True)

    assert "Voigt notation (xx, yy, zz, yz, xz, xy):" in capsys.readouterr().out


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
