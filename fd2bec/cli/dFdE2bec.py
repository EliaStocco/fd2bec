import argparse

import numpy as np
from typing import List
from ase import Atoms
from pathlib import Path
from fd2bec import SYMPREC
from fd2bec import float_format
from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.tensor import BornCharge
from fd2bec.atomic import AtomicStructure
from fd2bec.linear_system import LinearSystem

description = (
    "Compute the Born Effective Charges as derivative of the forces w.r.t. applied electric field."
)


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to extxyz file with all structures (e.g. structures.extxyz)",
    )
    parser.add_argument(
        "-e",
        "--efield_keyword",
        **argv,
        type=str,
        required=False,
        help="keyword for the electric field [V/ang] (default: %(default)s)",
        default="efield",
    )
    parser.add_argument(
        "-f",
        "--forces_keyword",
        **argv,
        type=str,
        required=False,
        help="keyword for the forces [eV/ang] (default: %(default)s)",
        default="REF_forces",
    )
    parser.add_argument(
        "-s",
        "--symprec",
        **argv,
        type=float,
        required=False,
        help="symmetry precision for spglib (default: %(default)s)",
        default=SYMPREC,
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=False,
        help="path to txt output file with the Born Charges (default: %(default)s)",
        default="bec.txt",
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structures from {args.input} ... ", end="")
    structures:List[Atoms] = read(args.input,index=":")
    print("done")
    
    Ns = len(structures) 
    Na = structures[0].get_global_number_of_atoms()
    print("n. structures: ", Ns)
    print("n. atoms: ", Na)
    
    pos = np.asarray([atoms.get_positions() for atoms in structures])
    assert np.all([np.allclose(pos_i,pos[0]) for pos_i in pos]), "You have provided different geometries."
    
    print(f"Extracting electric field from  {args.efield_keyword} ... ", end="")    
    efield = np.asarray([atoms.info["efield"] for atoms in structures])
    print("done")
    print("efield.shape: ",efield.shape)
    
    print(f"Extracting forces from  {args.forces_keyword} ... ", end="")    
    forces = np.asarray([atoms.arrays["REF_forces"] for atoms in structures])
    print("done")
    print("forces.shape: ",forces.shape)
    
    
    print(f"Preparing linear systems ... ", end="")   
    all_ls:List[LinearSystem] = [] 
    ones = np.full((Ns,1),1.)
    A = np.hstack((ones,efield))
    for n in range(Na):
        b = forces[:,n,:]
        all_ls.append(LinearSystem(A=A,b=b))
    print("done")
    
    print(f"Solving linear systems  ... ", end="")   
    for ls in all_ls:
        ls.solve()
    print("done")
    
    bec = np.zeros((Na,3,3))
    print(f"Extracting Born Charges  ... ", end="") 
    for n in range(Na):  
        bec[n,:,:] = all_ls[n].x[1:,:]
    print("done")

    print(f"Writing Born Charges to {args.output} ... ", end="")
    np.savetxt(args.output, bec.reshape((Na,9)), fmt=float_format)
    print("done")
    
    print(f"Symmetrizing Born Charges  ... ", end="") 
    aperiodic = structures[0].copy()
    aperiodic.set_cell([100,100,100])
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer

    unit_cell = AtomicStructure.from_ase(aperiodic)
    z = BornCharge(data=bec)
    P = unit_cell.get_totally_symmetric_projection(tensor=z)
    print("done")
    
    
    p = Path(args.output)
    new_filename = p.with_name(f"{p.stem}_sym{p.suffix}")
    

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
