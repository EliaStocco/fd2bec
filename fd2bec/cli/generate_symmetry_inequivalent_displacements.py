from fd2bec import float_format
from fd2bec.cli import cli
from ase.io import read
import numpy as np
from fd2bec.atomic import AtomicStructure

description = "Generate all symmetry inequivalent cartesian displacements."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input"    , **argv, type=str  , required=True, help="path to input structure (e.g. supercell.extxyz)")
    parser.add_argument("-a", "--amplitude", **argv, type=float, required=True, help="amplitude of the displacement")
    parser.add_argument("-o", "--output"   , **argv, type=str  , required=True, help="path to txt output file with cartesian displacements (e.g. displacement.txt)")
    return parser

@cli(prepare_args,description)
def main(args):
    
    print(f"Reading input structure from {args.input} ... ",end="")
    atoms = read(args.input, index=0)
    print("done")
    
    unit_cell = AtomicStructure.from_ase(atoms)
    S, theta, theta_real, shape = unit_cell.get_symmetrizer(rank=2,atomic=True,affine=False)
    
    theta_real = theta_real.reshape((-1,len(atoms),3,3))
    
    displacements = np.zeros((len(theta),3*len(atoms)))
    for n,t in enumerate(theta_real):
        displacements[n] = np.sum(t != 0,axis=2).flatten()
        
    displacements = displacements / np.linalg.norm(displacements,axis=1)[:,None]
    displacements *= args.amplitude
    
    print(f"Writing cartesian displacements to {args.output} ... ",end="")
    np.savetxt(args.output, displacements,fmt=float_format)
    print("done")

if __name__ == "__main__":
    main()