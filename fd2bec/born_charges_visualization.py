"""Portable atom-centred visualisation data for Born effective charges."""

from typing import Dict, Sequence, Tuple

import numpy as np
from ase import Atoms

from fd2bec.born_charges_plot import born_charges_from_atoms


def normalized_direction(direction: Sequence[float]) -> np.ndarray:
    """Return a normalized three-component displacement direction."""
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,):
        raise ValueError(f"A displacement direction must have shape (3,), got {direction.shape}.")
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("A displacement direction must be finite and non-zero.")
    return direction / norm


def born_charge_visualization_data(
    born_charges: np.ndarray, direction: Sequence[float]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return a normalized direction and OVITO-ready BEC response arrays.

    The response vector is ``Zstar @ direction``.  Its longitudinal and
    transverse parts are also supplied as scalar magnitudes for colouring and
    filtering in visualization programs.
    """
    born_charges = np.asarray(born_charges, dtype=float)
    if born_charges.ndim != 3 or born_charges.shape[1:] != (3, 3):
        raise ValueError(
            f"Born effective charges must have shape (atoms, 3, 3), got {born_charges.shape}."
        )
    if not np.all(np.isfinite(born_charges)):
        raise ValueError("Born effective charges contain non-finite values.")
    direction = normalized_direction(direction)
    response = np.einsum("nij,j->ni", born_charges, direction)
    longitudinal = np.einsum("i,ni->n", direction, response)
    transverse = response - longitudinal[:, np.newaxis] * direction
    return direction, {
        "tensor": born_charges.reshape((len(born_charges), 9)),
        "response": response,
        "longitudinal": longitudinal,
        "magnitude": np.linalg.norm(response, axis=1),
        "transverse": np.linalg.norm(transverse, axis=1),
    }


def attach_born_charge_visualization_data(
    atoms: Atoms,
    keyword: str,
    direction: Sequence[float],
    *,
    prefix: str = "BEC_",
) -> Tuple[Atoms, np.ndarray]:
    """Copy atoms and attach tensor and directional BEC response properties."""
    if not prefix:
        raise ValueError("The property prefix cannot be empty.")
    born_charges = born_charges_from_atoms(atoms, keyword)
    selected_direction, data = born_charge_visualization_data(born_charges, direction)
    result = atoms.copy()
    for name, values in data.items():
        result.set_array(f"{prefix}{name}", values)
    return result, selected_direction
