import numpy as np
import json
from ase.io import read
from fd2bec import SYMPREC
from fd2bec.cli import cli, str2bool
from fd2bec.atomic import AtomicStructure

description = "Prepare the file to solve the linear system to get Born Effective Charges."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-uc", "--unit_cell"              , **argv, type=str     , required=True , help="path to unit cell structure (e.g. unitell.extxyz)")
    parser.add_argument("-sc", "--super_cell"              , **argv, type=str     , required=True , help="path to unit super structure (e.g. supercell.extxyz)")
    parser.add_argument("-b", "--coefficients"       , **argv, type=str     , required=True , help="path to coefficients (e.g. dipole.txt)")
    parser.add_argument("-A", "--matrix"             , **argv, type=str     , required=True , help="path to displacement matrix (e.g. displacement.txt)" )
    parser.add_argument("-asr", "--acoustic_sum_rule", **argv, type=float   , required=False, help="weight for the acoustic sum rule, -1: not used, positive number otherwise (default: %(default)s)", default=-1)
    parser.add_argument("-tran", "--translations"    , **argv, type=str2bool, required=False, help="apply translational symmetries (default: %(default)s)", default=True)
    parser.add_argument("-spg", "--space_group"      , **argv, type=str2bool, required=False, help="apply space group symmetries (default: %(default)s)", default=True)
    parser.add_argument("-s"  , "--symprec"          , **argv, type=float   , required=False, help="symmetry precision for spglib (default: %(default)s)", default=SYMPREC)
    parser.add_argument("-o"  , "--output"           , **argv, type=str     , required=True , help="JSON output file")
    return parser

@cli(prepare_args,description)
def main(args):
    
    #----------------------#
    # Structures
    #----------------------#
    
    # unitcell    
    print(f"Reading unit cell structure from {args.unit_cell} ... ",end="")
    unit_cell = read(args.unit_cell, index=0)
    print("done")
    Na = unit_cell.get_global_number_of_atoms()
    n_unknowns = Na * 3
    print(f"Number of atoms in the unit cell: {Na}")
    unit_cell = AtomicStructure.from_ase(unit_cell)
    x = np.zeros((Na*3, 3))
    print("x.shape:", x.shape)
    
    # supercell 
    print(f"Reading unit cell structure from {args.super_cell} ... ",end="")
    super_cell = read(args.super_cell, index=0)
    print("done")
    Nas = super_cell.get_global_number_of_atoms()
    print(f"Number of atoms in the super cell: {Nas}")
    if Nas % Na != 0:
        raise ValueError(f"The number of atoms in the super cell ({Nas}) must be a multiple of the number of atoms in the unit cell ({Na}).")
    if Nas < Na:
        raise ValueError(f"The number of atoms in the super cell ({Nas}) must be greater than or equal to the number of atoms in the unit cell ({Na}).")
    elif Nas > Na:
        use_supercell = True
    else:
        use_supercell = False
    super_cell = AtomicStructure.from_ase(super_cell)
        
    #----------------------#
    # Symmetries
    #----------------------#
    spg_uc = unit_cell.to_spglib_cell(symprec=args.symprec)
    spg_sc = super_cell.to_spglib_cell(symprec=args.symprec)
    
    assert spg_uc.number == spg_sc.number, \
        f"The unit cell and the super cell must have the same space group but they have space groups {spg_uc.number} and {spg_sc.number}."
    
    
    #----------------------#
    # Coefficients
    #----------------------#
    
    # b
    print(f"Reading the coefficients b from file {args.coefficients} ... ",end="")
    b = np.loadtxt(args.coefficients)
    print("done")
    print("b.shape:", b.shape)
    assert b.shape[1] == 3, f"'b' must have 3 columns but it has shape {b.shape}"
    
    # A
    print(f"Reading the matrix A from file {args.matrix} ... ",end="")
    A = np.loadtxt(args.matrix)
    print("done")
    print("A.shape:", A.shape)
    
    
    #----------------------#
    # Sanity checks
    #----------------------#
    
    if A.shape[1] > n_unknowns:
        if not use_supercell:
            raise ValueError(f"The number of columns in A ({A.shape[1]}) must be less than or equal to the number of unknowns ({n_unknowns}). If you want to use a supercell, please provide a super cell structure with more atoms than the unit cell.")
        # print("It seems that you are using a supercell.")
        
    if use_supercell:
        print("Using translations symmetries for supercells.")
        args.translations = True
    
    b_coeff = b.copy()
    A_coeff = A.copy()
    
    if args.acoustic_sum_rule >= 0:
        b_coeff = np.vstack([np.zeros((3,3)), b_coeff])
        id = np.eye(3)
        A_coeff = np.vstack([np.tile(id, A_coeff.shape[1]//3), A_coeff])
        
    assert b_coeff.shape[0] == A_coeff.shape[0], \
        f"'b_coeff' and 'A_coeff' must have the same number of rows but they have shapes {b_coeff.shape} and {A_coeff.shape}"
        
    assert A.shape[1] == A_coeff.shape[1], \
        f"A and A_coeff must have the same number of columns but they have shapes {A.shape} and {A_coeff.shape}"
        
    # assert A.shape[1] == x.shape[0], \
    #     f"The number of columns in A ({A.shape[1]}) must be equal to the number of unknowns ({x.shape[0]})."
    
    system_type = "overdetermined" if A_coeff.shape[0] > x.shape[0] else "underdetermined" if A_coeff.shape[0] < x.shape[0] else "determined"
    print(f"System type: {system_type}")
    
    #----------------------#
    # Save data
    #----------------------#
    
    data = {
        "asr_weight" : float(args.acoustic_sum_rule),
        "apply_translations" : args.translations,
        "apply_space_group" : args.space_group,
        "unitcell" : unit_cell.to_json(),
        "input" : {
            "b" : b.tolist(),
            "A" : A.tolist(),
        },
        "linear_system" : {
            "type" : system_type,
            "n_rows" : b_coeff.shape[0],
            "n_cols" : A_coeff.shape[1],
            "n_unknowns" : n_unknowns,
            "b" : b_coeff.tolist(),
            "A" : A_coeff.tolist(),
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)
    

if __name__ == "__main__":
    main()