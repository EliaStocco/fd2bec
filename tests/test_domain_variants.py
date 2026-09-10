import numpy as np
from ase import Atoms
from ase.io import read, write

from fd2bec.cli.structures import generate_domain_variants as domain_variants_cli
from fd2bec.cli.structures import generate_inequivalent_pairs as pairs_cli
from fd2bec.domain_variants import (
    common_space_group_symbol,
    generate_domain_variants,
    inequivalent_structure_pairs,
    parent_point_operations,
    structures_match,
)
from fd2bec.show import print_compact_structure, print_structure


def _cubic_parent():
    return Atoms(
        "BaTiO3",
        cell=np.eye(3) * 4.0,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        pbc=True,
    )


def _tetragonal_variant():
    return Atoms(
        "BaTiO3",
        cell=np.diag([4.0, 4.0, 4.1]),
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.52],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        pbc=True,
    )


def _parent_with_roundoff_translations():
    parent = _cubic_parent()
    fractional = parent.get_scaled_positions(wrap=True)
    fractional[fractional > 0.0] -= 1e-9
    parent.set_scaled_positions(fractional)
    return parent


def test_cubic_parent_operations_and_tetragonal_variants():
    parent = _cubic_parent()
    variants = generate_domain_variants(_tetragonal_variant(), parent, symprec=1e-4)

    assert len(parent_point_operations(parent, symprec=1e-4)) == 48
    assert len(variants) == 6
    assert common_space_group_symbol(variants, symprec=1e-4) == "P4mm"

    cells = {tuple(np.round(np.diag(variant.cell.array), 8)) for variant in variants}
    assert cells == {(4.0, 4.0, 4.1), (4.0, 4.1, 4.0), (4.1, 4.0, 4.0)}
    assert all(
        not structures_match(variant, other, symprec=1e-4)
        for index, variant in enumerate(variants)
        for other in variants[index + 1 :]
    )


def test_roundoff_in_parent_translations_does_not_create_extra_variants():
    variants = generate_domain_variants(
        _tetragonal_variant(), _parent_with_roundoff_translations(), symprec=1e-4
    )

    assert len(variants) == 6


def test_variant_generation_removes_a_rigid_lattice_rotation():
    atoms = _tetragonal_variant()
    angle = np.deg2rad(7.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = atoms.copy()
    rotated.set_cell(atoms.cell.array @ rotation, scale_atoms=False)
    rotated.set_positions(atoms.get_positions() @ rotation, apply_constraint=False)

    assert len(generate_domain_variants(rotated, _cubic_parent(), symprec=1e-4)) == 6


def test_structure_match_ignores_atom_order():
    atoms = _tetragonal_variant()
    reordered = atoms[[1, 0, 4, 3, 2]]

    assert structures_match(atoms, reordered, symprec=1e-4)


def test_structure_match_can_ignore_a_global_translation():
    atoms = _tetragonal_variant()
    translated = atoms.copy()
    translated.set_scaled_positions(atoms.get_scaled_positions(wrap=True) + [0.25, 0.0, 0.0])

    assert not structures_match(atoms, translated, symprec=1e-4)
    assert structures_match(atoms, translated, symprec=1e-4, allow_global_translation=True)


def test_same_domain_file_has_two_inequivalent_tetragonal_pairs():
    variants = generate_domain_variants(_tetragonal_variant(), _cubic_parent(), symprec=1e-4)

    pairs = inequivalent_structure_pairs(
        variants,
        variants,
        _cubic_parent(),
        symprec=1e-4,
        treat_reversed_as_equivalent=True,
    )

    assert len(pairs) == 2
    assert sorted(len(pair.equivalent_pairs) for pair in pairs) == [3, 12]


def test_variants_preserve_per_atom_arrays():
    atoms = _tetragonal_variant()
    markers = np.arange(len(atoms) * 2, dtype=float).reshape((len(atoms), 2))
    atoms.set_array("markers", markers)

    variants = generate_domain_variants(atoms, _cubic_parent(), symprec=1e-4)

    for variant in variants:
        np.testing.assert_array_equal(variant.arrays["markers"], markers)


def test_compact_structure_output_normalizes_display_roundoff(capsys):
    atoms = _tetragonal_variant()
    fractional = atoms.get_scaled_positions(wrap=True)
    fractional[1, 0] = -1e-9
    atoms.set_scaled_positions(fractional)
    atoms.cell.array[0, 1] = -1e-12

    print_compact_structure(atoms, title="Variant 1", parent=_cubic_parent())

    output = capsys.readouterr().out
    assert "-0.000000" not in output
    assert "1.000000" not in output
    assert " 0.000000" in output
    assert "lattice difference from parent" in output
    assert "fractional difference from parent" in output


def test_print_structure_encloses_all_content_in_a_frame(capsys):
    print_structure(_tetragonal_variant(), title="Tetragonal structure")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("+") and lines[0].endswith("+")
    assert lines[-1] == lines[0]
    assert all(line.startswith("|") and line.endswith("|") for line in lines[1:-1])
    assert all(len(line) == len(lines[0]) for line in lines)
    assert "Tetragonal structure" in lines[1]


def test_cli_writes_all_domain_variants(tmp_path, capsys):
    input_path = tmp_path / "tetragonal.extxyz"
    parent_path = tmp_path / "cubic.extxyz"
    output_path = tmp_path / "variants.extxyz"
    write(input_path, _tetragonal_variant())
    write(parent_path, _cubic_parent())
    args = domain_variants_cli.prepare_args(domain_variants_cli.description).parse_args(
        [
            "-i",
            str(input_path),
            "-p",
            str(parent_path),
            "-o",
            str(output_path),
            "--show-differences",
        ]
    )

    domain_variants_cli.main.__wrapped__(args)

    assert len(read(output_path, index=":")) == 6
    output = capsys.readouterr().out
    assert "Variant 1 (P4mm):" in output
    assert "All generated structures have space group P4mm." in output
    assert "Distorted input structure" in output
    assert "Parent reference structure" in output
    assert "lattice [Angstrom]" in output
    assert "fractional:" in output
    assert "lattice difference from parent" in output


def test_cli_differences_are_opt_in():
    parser = domain_variants_cli.prepare_args(domain_variants_cli.description)

    defaults = parser.parse_args(["-i", "input.extxyz", "-p", "parent.extxyz", "-o", "out.extxyz"])
    enabled = parser.parse_args(
        ["-i", "input.extxyz", "-p", "parent.extxyz", "-o", "out.extxyz", "--show-differences"]
    )

    assert not defaults.show_differences
    assert enabled.show_differences


def test_pairs_cli_writes_two_frames_per_representative_pair(tmp_path, capsys):
    input_path = tmp_path / "tetragonal.extxyz"
    parent_path = tmp_path / "cubic.extxyz"
    output_path = tmp_path / "pairs.extxyz"
    variants = generate_domain_variants(_tetragonal_variant(), _cubic_parent(), symprec=1e-4)
    write(input_path, variants)
    write(parent_path, _cubic_parent())
    args = pairs_cli.prepare_args(pairs_cli.description).parse_args(
        [
            "-i",
            str(input_path),
            "-j",
            str(input_path),
            "-p",
            str(parent_path),
            "-o",
            str(output_path),
        ]
    )

    pairs_cli.main.__wrapped__(args)

    output = read(output_path, index=":")
    assert len(output) == 4
    assert [atoms.info["fd2bec_pair_index"] for atoms in output] == [1, 1, 2, 2]
    assert "Found 2 inequivalent pairs from 15 candidate pairs" in capsys.readouterr().out
