import numpy as np
import pytest
from typing import List

from fd2bec import ATOL
from fd2bec.linear_system import LinearSystem, StackedLinearSystem


@pytest.mark.parametrize(
    "n_sys, m, n, k",
    [
        (1, 3, 3, 1),
        (2, 3, 3, 1),
        (3, 4, 4, 1),
        (5, 2, 2, 1),
        (2, 3, 3, 2),
        (4, 5, 3, 1),
    ]
)
def test_block_system_equivalence(n_sys, m, n, k):
    rng = np.random.default_rng(0)

    systems: List[LinearSystem] = []
    solutions_individual:List[np.ndarray] = []

    # IMPORTANT FIX: use n_sys
    for _ in range(n_sys):

        A = rng.normal(size=(m, n))
        b = rng.normal(size=(m, k))

        sys = LinearSystem(A, b)
        sys.solve()


        solutions_individual.append(sys.x.copy())
        sys.x = None

        systems.append(sys)

    big_sys = StackedLinearSystem(systems)
    big_sys.solve()

    for nn,sys in enumerate(systems):
        if sys.x.shape != solutions_individual[nn].shape:
            raise ValueError("error")
        if not np.allclose(sys.x,solutions_individual[nn], atol=ATOL):
            raise ValueError("error")


if __name__ == "__main__":
    pytest.main([__file__])