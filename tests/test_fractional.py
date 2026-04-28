import numpy as np
import pytest
from typing import Dict, Tuple
from ase.io import read
from pathlib import Path

from fd2bec import ATOL
from fd2bec.tools import atoms2bec
from fd2bec.tensor import Dipole, BornCharge, Force, Stress, Tensor, Vector


FILE = Path(__file__).parent / "rotations/rotated.extxyz"

instructions:Dict[str,Tuple[str,type]] = {
    "positions" : ( "array" , Vector ),
    "MACE_BEC" : ( "array" , BornCharge ),
    "MACE_forces" : ( "array" ,Force),
    "MACE_dipole" : ( "info" ,Dipole),
    "MACE_stress": ( "info" ,Stress)
}

def assert_allclose_debug(a:np.ndarray, b:np.ndarray, atol:float, msg:str):
    diff = np.abs(a - b)
    max_diff = np.max(diff)

    if not np.allclose(a, b, atol=atol):
        raise ValueError(
        f"\n{msg}"
        f"\nMax |Δ|: {max_diff:.3e}"
        f"\nShape: {a.shape}"
    )

@pytest.mark.parametrize("n", range(10))
def test_fractional(n):
    
    atoms = read(FILE, index=n)
    
    for keyword, (where,classname) in instructions.items():
        if keyword == "MACE_BEC":
            array = atoms2bec(atoms,keyword)
        elif where == "array":
            array = atoms.arrays[keyword]
        else:
            array = atoms.info[keyword]
        
        original:Tensor = classname(data=array,cell=atoms.cell)
        fractional = original.to("fractional")
        test = fractional.to("cartesian")
        
        assert_allclose_debug(original.data, test.data, ATOL, f"[{keyword} ROUNDTRIP MISMATCH]")
        
        if classname in [Vector,Dipole]:
            frac_pos = atoms.cell.scaled_positions(array)
            original.to("fractional")
            assert_allclose_debug(fractional.data, frac_pos, ATOL, f"[{keyword} ROUNDTRIP MISMATCH]")
        
        pass

if __name__ == "__main__":
    pytest.main([__file__])