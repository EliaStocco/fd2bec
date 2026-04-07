from fd2bec.cli import cli
from fd2bec.mathematics import pseudo_inverse
import numpy as np

description = "Solve a linear system."

choices = ['pseudo-inverse','lstsq']

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-b", "--coefficients" , **argv, type=str  , required=True, help="path to coefficients (e.g. dipole.txt)")
    parser.add_argument("-A", "--matrix"       , **argv, type=str  , required=True, help="path to displacement matrix (e.g. displacement.txt)" )
    parser.add_argument("-x", "--unknown"      , **argv, type=str  , required=True, help="path to output file (e.g. bec.txt)")
    parser.add_argument("-m", "--method"       , **argv, type=str  , required=True, help=f"method: {choices}"+" (default: %(default)s)", default="pseudo-inverse", choices=choices)
    return parser

@cli(prepare_args,description)
def main(args):
    
    print(f"Reading the coefficients b from file {args.coefficients} ... ",end="")
    b = np.loadtxt(args.coefficients)
    print("done")
    print("b.shape:", b.shape)
    
    print(f"Reading the matrix A from file {args.matrix} ... ",end="")
    A = np.loadtxt(args.matrix)
    print("done")
    print("A.shape:", A.shape)
    
    print(f"Solving the linear system Ax = b with '{args.method}' method ... ",end="")
    if args.method == "pseudo-inverse":
        Aplus = pseudo_inverse(A)
        x = Aplus @ b
    elif args.method == "lstsq":
        x, residuals, rank, singular_values = np.linalg.lstsq(A, b)
    else:
        raise ValueError(f"Unknown method: {args.method}\n"
                         f"Available methods: {choices}")
    print("done")
    print("x.shape:", x.shape)
    

if __name__ == "__main__":
    main()