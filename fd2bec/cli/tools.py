from warnings import warn

import numpy as np

from fd2bec import BEC_NORM_THRESHOLD
from fd2bec.atomic import AtomicStructure


def print_born_charges(reference: AtomicStructure, bec: np.ndarray):
    """
    Pretty-print Born effective charges and optionally warn about large norms.
    """

    Na = len(reference.symbols)

    for n in range(Na):
        symbol = reference.symbols[n]
        pos = reference.positions[n]

        zstar = bec[n]
        norm = np.linalg.norm(zstar)

        # --- atom header ---
        print(f"Atom {n:3d}, species {symbol}")
        print(f"Position: [{pos[0]:10.6f} {pos[1]:10.6f} {pos[2]:10.6f}]")
        print(f"||Z*|| = {norm:10.5f}")
        print("Born effective charge tensor (Z*):")

        for row in zstar:
            print("    " + " ".join(f"{x:10.5f}" for x in row))

        print()

        # --- optional warning ---
        if BEC_NORM_THRESHOLD is not None and norm > BEC_NORM_THRESHOLD:
            warn(
                f"Large Born effective charge detected for atom {n} ({symbol}): "
                f"||Z*|| = {norm:.3f} > {BEC_NORM_THRESHOLD:.3f}"
            )
