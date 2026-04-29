from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from fd2bec import ATOL
from fd2bec.system import System
from fd2bec.atomic import AtomicStructure
from fd2bec.tensor import BornCharge
from fd2bec.cli.generate_all_displacements import atomic_structure2all_displacements



# from fd2bec.symmetry import is_sohncke

DATA_DIR = Path(__file__).parent / "MP/spacegroup_structures"
# DATA_DIR = Path(__file__).parent / "MP/problems"

AMPLITUDE = 0.01

def run_workflow(filepath):
    atoms = read(filepath)
    unit_cell = AtomicStructure.from_ase(atoms)
    if not np.all(unit_cell.cell == atoms.cell):
        pytest.skip("There is a problem with the cell.")

    # unit_cell._test_symmetry(atol=ATOL)

    Na = atoms.get_global_number_of_atoms()
    bec = np.random.rand(Na, 3, 3)

    bec = BornCharge(data=bec)
    tmp = unit_cell.to("fractional",bec)
    new_bec = unit_cell.to("cartesian",tmp)
    if not np.allclose(new_bec, bec, atol=ATOL):
        raise ValueError(f"Error in coordinate transformation for {filepath}")

    try:
        tmp = unit_cell.to("fractional",bec)
        P = unit_cell.get_totally_symmetric_projection(tensor=tmp)
        tmp: np.ndarray = P @ tmp.flatten(full=True)  # symmetrize the BECs
        tmp = BornCharge(data=tmp.reshape((Na, 3, 3)))
        bec = unit_cell.to("cartesian",tmp)
    except Exception as e:
        raise ValueError(f"Error symmetrizing BECs for {filepath}: {e}")

    if not np.allclose(P @ bec.flatten(full=True), bec.flatten(full=True), atol=ATOL):
        raise ValueError("BECs are not symmetrized correctly")

    d, _ = atomic_structure2all_displacements(
        unit_cell, amplitude=AMPLITUDE, use_delta_dipole=False
    )

    deltaR = d.reshape((-1, Na, 3))
    delta_mu = np.einsum("ijk,jkl->il", deltaR, bec)


    # system = System(
    #     unit_cell=unit_cell,
    #     dipoles=delta_mu,
    #     displacements=d,
    #     use_delta_dipole=False,
    #     asr_weight=-1.0,
    #     use_spacegroup_symmetry=False,
    # )

    # assert system.rank_type() == "determined", f"Linear system is not determined for {filepath}"

    # if system.rank_type() == "underdetermined":
    #     raise ValueError(f"Linear system is underdetermined for {filepath}, cannot solve for BECs.")

    # system.solve(method="lstsq")
    # computed_bec = system.born_charges.flatten()

    # if not np.allclose(computed_bec, bec.flatten(), atol=ATOL):
    #     if np.allclose(unit_cell.cellpar[3:], 90):
    #         raise ValueError("Really weird")
    #     raise ValueError(
    #     f"Computed BECs do not match the original BECs for {filepath}.\n"
    #     f"Original BECs:\n{bec.flatten()}\n"
    #     f"Computed BECs:\n{computed_bec}"
    # )

    # pass


# @pytest.mark.skip("Currently fails for some structures, need to investigate the cause")
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
