import numpy as np
from ase import Atoms
from ase.io import read, write

from fd2bec.cli.structures import generate_domain_variants as domain_variants_cli
from fd2bec.domain_variants import generate_domain_variants, parent_point_operations


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


def test_cubic_parent_operations_and_tetragonal_variants():
    parent = _cubic_parent()
    variants = generate_domain_variants(_tetragonal_variant(), parent, symprec=1e-4)

    assert len(parent_point_operations(parent, symprec=1e-4)) == 48
    assert len(variants) == 6

    cells = {tuple(np.round(np.diag(variant.cell.array), 8)) for variant in variants}
    assert cells == {(4.0, 4.0, 4.1), (4.0, 4.1, 4.0), (4.1, 4.0, 4.0)}


def test_cli_writes_all_domain_variants(tmp_path):
    input_path = tmp_path / "tetragonal.extxyz"
    parent_path = tmp_path / "cubic.extxyz"
    output_path = tmp_path / "variants.extxyz"
    write(input_path, _tetragonal_variant())
    write(parent_path, _cubic_parent())
    args = domain_variants_cli.prepare_args(domain_variants_cli.description).parse_args(
        ["-i", str(input_path), "-p", str(parent_path), "-o", str(output_path)]
    )

    domain_variants_cli.main.__wrapped__(args)

    assert len(read(output_path, index=":")) == 6
