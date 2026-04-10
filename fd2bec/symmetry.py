import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SymmetryRepresentationBuilder:
    natoms: int
    
    # -------------------------
    # permutation matrix
    # -------------------------
    def build_permutation(self, mapping):
        P = np.zeros((self.natoms, self.natoms))
        i = np.arange(self.natoms)
        P[i, mapping] = 1
        return P

    # -------------------------
    # Cartesian tensor part
    # -------------------------
    def build_cartesian_rep(self, R, rank):
        out = R
        for _ in range(rank - 1):
            out = np.kron(out, R)
        return out

    # -------------------------
    # full linear operator
    # -------------------------
    def build_R_flat(self, mapping, R, rank=1):
        P = self.build_permutation(mapping)
        R_cart = self.build_cartesian_rep(R, rank)
        return np.kron(P, R_cart)

    # -------------------------
    # translation term (ONLY for positions)
    # -------------------------
    def build_T_flat(self, mapping, t, use_translations=True):

        if not use_translations:
            return np.zeros((self.natoms , 3 ))

        # expand translation per atom
        t_atom = np.tile(t, self.natoms)

        # permute atoms
        P = self.build_permutation(mapping)
        t_atom = (P @ t_atom.reshape(self.natoms, 3)).reshape(-1)

        # if rank > 1 → no effect on internal tensor structure
        # (translations do NOT act on tensor indices)
        return t_atom