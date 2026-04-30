from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .atomic import AtomicStructure
from .linear_system import LinearSystem


@dataclass
class System:
    unit_cell: AtomicStructure
    dipoles: np.ndarray
    displacements: np.ndarray
    use_delta_dipole: bool = False
    asr_weight: float = -1.0
    use_spacegroup_symmetry: bool = True

    linear_system: LinearSystem = field(init=False)
    symmetrizer: np.ndarray = field(init=False)
    unknowns: np.ndarray = field(init=False)

    def __post_init__(self):
        # assert self.use_spacegroup_symmetry, "Only spgroup symmetry is supported for now."
        assert self.asr_weight == -1.0, "Only asr_weight=-1.0 is supported for now."
        nr = len(self.displacements)
        A = self.unit_cell.to_fractional(
            self.displacements.reshape((nr, len(self.unit_cell), 3)), rank=1
        ).reshape((nr, -1))
        b = self.unit_cell.to_fractional(self.dipoles, rank=1)
        # self.linear_system = LinearSystem(A = A, b = b)

        if self.use_spacegroup_symmetry:
            b = b.flatten()
            A = np.kron(A, np.eye(3))

            # if self.use_spacegroup_symmetry:
            S, theta, _ = self.unit_cell.get_symmetrizer(rank=2, atomic=True, affine=False)
            n_unknowns = len(theta)
            x = np.asarray([f"theta_{n}" for n in range(n_unknowns)], dtype=object)
            A = A @ S
        else:
            x = []
            for a in range(len(self.unit_cell)):
                for i in ["x", "y", "z"]:
                    for j in ["x", "y", "z"]:
                        x.append(f"mu_{j} / d R_{i}^{a}")
            x = np.asarray(x, dtype=object).reshape((-1, 3))

        if not self.use_delta_dipole:
            # this could be improved by using the space group
            # to understand which are the independent components of the dipole
            if self.use_spacegroup_symmetry:
                x = np.concatenate([np.asarray(["mu_x", "mu_y", "mu_z"], dtype=object), x])
                nr = int(len(b) / 3)
                tmp = np.tile(-np.eye(3), nr).T
            else:
                nr = len(b)
                tmp = np.full((nr, 1), 1, dtype=float)
                x = np.concatenate([np.asarray(["mu_x", "mu_y", "mu_z"], dtype=object)[None, :], x])

            A = np.hstack([tmp, A])

        self.symmetrizer = S if self.use_spacegroup_symmetry else None
        self.linear_system = LinearSystem(A=A, b=b)
        self.unknowns = x

    @property
    def min_num_displacements(self):
        return self.linear_system.n_unknowns

    def solve(self, **kwargs):
        self.linear_system.solve(**kwargs)

    @cached_property
    def born_charges(self):
        if self.linear_system.x is None:
            raise ValueError("The linear system has not been solved yet.")
        x = self.linear_system.x
        if not self.use_delta_dipole:
            x = x[3:]
        bec = self.symmetrizer @ x
        bec = bec.reshape((len(self.unit_cell), 3, 3))
        return bec

    def rank_type(self):
        if self.linear_system.n_cols < self.linear_system.n_unknowns:
            return "overdetermined"
        elif self.linear_system.n_cols == self.linear_system.n_rows:
            return "determined"
        else:
            return "underdetermined"
