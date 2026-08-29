import numpy as np
from ase import Atoms

from fd2bec import SYMPREC

SOHNCKE_GROUPS = {
    1,
    3,
    4,
    5,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    75,
    76,
    77,
    78,
    79,
    80,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    143,
    144,
    145,
    146,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
}


def is_sohncke(sg_number):
    """Check if the space group number corresponds to a Sohncke group."""
    return sg_number in SOHNCKE_GROUPS


def symmetrize_bec(structure: Atoms, bec: np.ndarray, symprec: float = SYMPREC) -> np.ndarray:
    from fd2bec.atomic import AtomicStructure
    from fd2bec.tensor import BornCharges

    tensor = BornCharges(data=bec)
    atomic_structure = AtomicStructure.from_ase(structure, symprec=symprec)
    return atomic_structure.symmetrize(tensor=tensor).data
