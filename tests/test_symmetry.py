from pathlib import Path
import re
import numpy as np
import pytest
import spglib
from ase.utils import atoms_to_spglib_cell
from fd2bec.io import read

from fd2bec import ATOL, SYMPREC
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import wrap


def test_space_group(sg_case):

    dataset, filepath, n = sg_case

    atoms = read(filepath)

    cell = atoms_to_spglib_cell(atoms)

    dataset = spglib.get_symmetry_dataset(cell, symprec=SYMPREC)
    number = dataset.number             # e.g. 225

    assert number == n, f"Wrong space group: {n} != {number} on file {filepath}"

def test_symmetry(sg_case):

    dataset, filepath, n = sg_case

    atoms = read(filepath)

    atomic_structure = AtomicStructure.from_ase(atoms)
    # assert atomic_structure.space_group == n, f"Wrong space group: {n} != {atomic_structure.space_group} on file {filepath}"
    atomic_structure._test_symmetry()
    pass
    # new_atoms = atoms.copy()
    # N = atoms.get_global_number_of_atoms()
    # frac_pos = atoms.get_scaled_positions()
    # new_frac = frac_pos.copy()

    # for op_idx, (R, t) in enumerate(zip(atomic_structure.spglib_dataset.rotations, atomic_structure.spglib_dataset.translations)):
    #     new_frac = new_frac @ R + t[None, :]
    #     new_atoms.set_scaled_positions(new_frac)
    #     tmp = AtomicStructure.from_ase(new_atoms)
    #     atol = ATOL * len(atoms)

    #     if not atomic_structure.is_equal_to(tmp, atol=atol):
    #         # compute distances for debugging

    #         mapping = atomic_structure._get_atoms_mapping(tmp)
    #         diff = wrap(atomic_structure.frac_pos[mapping] - tmp.frac_pos)
    #         max_dev = np.max(np.abs(diff))

    #         raise AssertionError(
    #             f"{filepath}: Symmetry operation #{op_idx} failed\n"
    #             f"Rotation:\n{R}\n"
    #             f"Translation: {t}\n"
    #             f"Max deviation after wrapping: {max_dev:.3e}\n"
    #             f"Tolerance: {SYMPREC}\n"
    #             f"Number of atoms: {N}"
    #         )

    # diff = new_frac - frac_pos
    # diff = wrap(diff)
    # max_dev = np.max(np.abs(diff))

    # assert np.allclose(diff, 0, atol=ATOL * N), (
    #     f"{filepath}: Final symmetry mismatch\n"
    #     f"Max deviation: {max_dev:.3e}\n"
    #     f"Tolerance: {1e-5 * N:.3e}\n"
    #     f"Mean deviation: {np.mean(np.abs(diff)):.3e}"
    # )

    # if not np.allclose(atomic_structure.cell.array, atoms.cell.array, atol=ATOL):
    #     raise ValueError("There is a problem with the cell.")
    # try:
    #     atomic_structure._test_symmetry(atol=ATOL * len(atomic_structure))
    # except Exception as e:
    #     atomic_structure._test_symmetry(atol=ATOL * len(atomic_structure))
    #     raise e


if __name__ == "__main__":
    pytest.main([__file__])
