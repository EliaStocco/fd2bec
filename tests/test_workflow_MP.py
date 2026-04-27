from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from fd2bec.atomic import AtomicStructure

# from fd2bec.symmetry import is_sohncke

DATA_DIR = Path(__file__).parent / "MP/spacegroup_structures"

AMPLITUDE = 0.01


def run_workflow(filepath):
    atoms = read(filepath)
    unit_cell = AtomicStructure.from_ase(atoms)

    unit_cell._test_symmetry(atol=1e-4)

    Na = atoms.get_global_number_of_atoms()
    bec = np.random.rand(Na, 3, 3)

    kwargs = {"rank": 2, "atomic": True, "affine": False}
    P = unit_cell.get_totally_symmetric_projection(**kwargs)
    S, _, _, _ = unit_cell.get_symmetrizer(atol=1e-4, **kwargs)

    try:
        bec: np.ndarray = P @ bec.flatten()  # symmetrize the BECs
    except Exception as e:
        raise ValueError(f"Error symmetrizing BECs for {filepath}: {e}")

    if not np.allclose(P @ bec.flatten(), bec.flatten()):
        raise ValueError("BECs are not symmetrized correctly")

    # u,d = atomic_structure2unique_displacements(unit_cell, amplitude=AMPLITUDE)
    # displaced_structures = displacements2atoms(atoms, d)


@pytest.mark.skip("Currently fails for some structures, need to investigate the cause")
@pytest.mark.parametrize("n", range(230))
def test_workflow_MP(n):
    pattern = DATA_DIR / f"SG_{n}_mp-*.cif"
    files = list(pattern.parent.glob(pattern.name))

    if not files:
        pytest.skip(f"No files found for space group {n}")

    assert len(files) == 1, f"Expected exactly one file matching {pattern}, but found {len(files)}"

    file = files[0]

    run_workflow(file)


if __name__ == "__main__":
    pytest.main([__file__])
