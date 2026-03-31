from fd2bec.cli import cli
from ase.io import read, write
import numpy as np

description = "Generate <N> (cartesian guassian) random displaced structures"

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input", **argv, type=str, required=True, help="path to input structure (e.g. supercell.extxyz)")
    parser.add_argument("-a", "--amplitude",**argv,  type=float, required=True, help="amplitude of the displacement")
    parser.add_argument("-n" , "--number"         , **argv, type=int, required=True,help="number of random displacements to generate" )
    parser.add_argument("-o", "--output",**argv,  type=str,required=True, help="path to output structure (e.g. supercell-displaced.extxyz)")
    return parser

@cli(prepare_args,description)
def main(args):
    
    print(f"Reading input structure from {args.input} ... ",end="")
    atoms = read(args.input, index=0)
    print("done")
    
    N = args.number * 3 * atoms.get_global_number_of_atoms()    
    print(f"Generating {N} random numbers for {args.number} displaced structures ... ",end="")
    shift = np.random.normal(scale=args.amplitude, size=(args.number,atoms.get_global_number_of_atoms(),3))
    print("done")
    
    displaced_structures = [None]*args.number
    for i in range(args.number):
        displaced = atoms.copy()
        displaced.set_positions(displaced.get_positions() + shift[i])
        displaced_structures[i] = displaced
    
    print(f"Writing displaced structures to {args.output} ... ",end="")
    write(args.output, displaced_structures)
    print("done")

if __name__ == "__main__":
    main()