import json
import numpy as np
from fd2bec.cli import cli

description = "Solve a linear system."

choices = ['pseudo-inverse','lstsq']

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input" , **argv, type=str  , required=True, help="JSON input file produced by 'prepare_linear_system.py'")
    parser.add_argument("-m", "--method"       , **argv, type=str  , required=True, help=f"method: {choices}"+" (default: %(default)s)", default="pseudo-inverse", choices=choices)
    parser.add_argument("-o", "--output" , **argv, type=str  , required=True, help="JSON output file")
    return parser

@cli(prepare_args,description)
def main(args):
    
    print(f"Reading linear system from {args.input} ... ",end="")
    with open(args.input, "r") as f:
        problem = json.load(f)
    print("done")
    
    A = np.array(problem["linear_system"]["A"])  
    b = np.array(problem["linear_system"]["b"])  
    
    print(f"A.shape: {A.shape}")
    print(f"b.shape: {b.shape}")
    
    rank = np.linalg.matrix_rank(A)
    
    if args.method == "pseudo-inverse":
        print("Solving linear system using pseudo-inverse ... ",end="")
        pinv = np.linalg.pinv(A)
        x = pinv @ b
        print("done")
        
        # SVD decomposition
        U, s, Vh = np.linalg.svd(A, full_matrices=False)


        # Pseudo-inverse solution
        S_inv = np.diag(1 / s)
        A_pinv = Vh.T @ S_inv @ U.T
        x = A_pinv @ b

    elif args.method == "lstsq":
        print("Solving linear system using least squares ... ",end="")
        x, residuals, rank_lstsq, singular_values = np.linalg.lstsq(A, b, rcond=None)
        assert rank == rank_lstsq, f"Rank mismatch: np.linalg.matrix_rank(A)={rank} vs np.linalg.lstsq(A,b)[2]={rank_lstsq}"
        print("done")
        
    output = {
        "results" : {
            "x.shape" : x.shape,
            "x" : x.tolist(),
            "A+" : A_pinv.tolist() if args.method == "pseudo-inverse" else None,
            "residuals" : residuals.tolist() if args.method == "lstsq" else None,
            "singular_values" : singular_values.tolist() if args.method == "lstsq" else None,
            "rank-A" : int(rank),
        },
        "equation" : problem
    }
    
    print(f"Saving results to {args.output} ... ",end="")
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)
    print("done")
    
    
if __name__ == "__main__":
    main()