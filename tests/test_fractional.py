from pathlib import Path
from typing import Dict, Tuple, Type

import numpy as np
from ase import Atoms
import pytest
from fd2bec.io import read

from fd2bec import ATOL
from fd2bec.tensor import BornCharges, Dipole, Forces, Stress, Tensor, Vector
from fd2bec.tools import atoms2bec

FILE = Path(__file__).parent / "rotations/rotated.extxyz"

instructions: Dict[str, Tuple[str, type]] = {
    "positions": ("array", Vector),
    "MACE_BEC": ("array", BornCharges),
    "MACE_forces": ("array", Forces),
    "REF_dipole": ("info", Dipole),
    "MACE_stress": ("info", Stress),
}


def assert_allclose_debug(a: np.ndarray, b: np.ndarray, atol: float, msg: str):
    diff = np.abs(a - b)
    max_diff = np.max(diff)

    if not np.allclose(a, b, atol=atol):
        raise ValueError(f"\n{msg}" f"\nMax |Δ|: {max_diff:.3e}" f"\nShape: {a.shape}")


@pytest.mark.parametrize("method", ["recursive", "flat"])
@pytest.mark.parametrize("n", range(10))
def test_fractional(n, method):

    atoms = read(FILE, index=n)

    for keyword, (where, classname) in instructions.items():
        if keyword == "MACE_BEC":
            array = atoms2bec(atoms, keyword)
        elif where == "array":
            array = atoms.arrays[keyword]
        else:
            array = atoms.info[keyword]

        original: Tensor = classname(data=array, cell=atoms.cell)
        fractional = original.to(basis="fractional", method=method)
        test = fractional.to("cartesian")

        assert_allclose_debug(original.data, test.data, ATOL, f"[{keyword} ROUNDTRIP MISMATCH]")

        if classname in [Vector, Dipole]:
            frac_pos = atoms.cell.scaled_positions(array)
            original.to("fractional")
            assert_allclose_debug(
                fractional.data, frac_pos, ATOL, f"[{keyword} ROUNDTRIP MISMATCH]"
            )


def _apply_all_keywords(atoms:Atoms, keyword):
    if keyword == "MACE_BEC":
        return atoms2bec(atoms, keyword)
    elif keyword in atoms.arrays:
        return atoms.arrays[keyword]
    else:
        return atoms.info[keyword]


@pytest.mark.parametrize("n", range(10))
def test_recursive_vs_flat_fractional(n):
    atoms = read(FILE, index=n)

    for keyword, (_, classname) in instructions.items():

        array = _apply_all_keywords(atoms, keyword)
        classname:Type[Tensor]
        original = classname(data=array, cell=atoms.cell)

        # ------------------------------------------------------------
        # Apply both methods from SAME starting point
        # ------------------------------------------------------------
        frac_recursive = original.to(
            basis="fractional",
            method="recursive",
        )

        frac_flat = original.to(
            basis="fractional",
            method="flat",
        )

        # ------------------------------------------------------------
        # Core equivalence check
        # ------------------------------------------------------------
        assert_allclose_debug(
            frac_recursive.data,
            frac_flat.data,
            atol=ATOL,
            msg=f"[{keyword}] recursive != flat (fractional transform)",
        )

        # ------------------------------------------------------------
        # Optional: ensure both are consistent under roundtrip
        # ------------------------------------------------------------
        cart_recursive = frac_recursive.to("cartesian", method="recursive")
        cart_flat = frac_flat.to("cartesian", method="flat")

        assert_allclose_debug(
            cart_recursive.data,
            cart_flat.data,
            atol=ATOL,
            msg=f"[{keyword}] recursive != flat (cartesian back-transform)",
        )


if __name__ == "__main__":
    pytest.main([__file__])
