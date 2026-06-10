from dataclasses import dataclass, field
from typing import List

import numpy as np

from fd2bec.mathematics import block_diag


@dataclass
class LinearSystem:
    A: np.ndarray
    b: np.ndarray
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.A.ndim != 2:
            raise ValueError(f"A must be a 2D array, but got shape {self.A.shape}")
        # if self.b.ndim != 1:
        #     raise ValueError(f"b must be a 1D array, but got shape {self.b.shape}")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(
                f"Number of rows in A ({self.A.shape[0]}) "
                + f"must match length of b ({self.b.shape[0]})"
            )
        self.x = None  # np.full((self.n_rows,self.n_cols), np.nan)

    @property
    def n_parallel(self):
        return self.b.shape[1]

    @property
    def n_rows(self):
        return self.A.shape[0]

    @property
    def n_cols(self):
        return self.A.shape[1]

    def solve(self, method="lstsq"):
        """Solve the linear system Ax = b using the specified method."""
        rank = np.linalg.matrix_rank(self.A)
        if method == "pseudo-inverse":
            pinv = np.linalg.pinv(self.A)
            x = pinv @ self.b

            # SVD decomposition
            U, s, Vh = np.linalg.svd(self.A, full_matrices=False)

            # Pseudo-inverse solution
            S_inv = np.diag(1 / s)
            A_pinv = Vh.T @ S_inv @ U.T
            x = A_pinv @ self.b

        elif method == "lstsq":
            x, residuals, rank_lstsq, singular_values = np.linalg.lstsq(self.A, self.b, rcond=None)
            assert rank == rank_lstsq, (
                f"Rank mismatch: np.linalg.matrix_rank(A)={rank}"
                + "vs np.linalg.lstsq(A,b)[2]={rank_lstsq}"
            )

        self.x = x


@dataclass
class StackedLinearSystem(LinearSystem):
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    list_n_rows: List[int] = field(init=False)
    list_n_cols: List[int] = field(init=False)
    systems: List[LinearSystem]

    def __post_init__(self):
        n_parallel = self.systems[0].n_parallel
        all_parallel = [sys.n_parallel for sys in self.systems]
        assert all([p == n_parallel for p in all_parallel]), (
            "Different number of parallel solutions."
        )
        self.list_n_rows = [sys.n_rows for sys in self.systems]
        self.list_n_cols = [sys.n_cols for sys in self.systems]
        self.A = block_diag([sys.A for sys in self.systems])
        self.b = np.vstack([sys.b for sys in self.systems])
        super().__post_init__()

    def __len__(self):
        return len(self.systems)

    def solve(self, *argv, **kwargs):
        super().solve(*argv, **kwargs)
        offset = 0
        for n, r in enumerate(self.list_n_cols):
            self.systems[n].x = self.x[offset : offset + r, :]
            offset += r
