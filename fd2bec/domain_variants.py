"""Generate parent-symmetry-related orientation-domain structures."""

from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

from fd2bec.tools import ase2spglib_dataset


def _require_periodic_cell(atoms: Atoms, label: str) -> None:
    if not np.all(atoms.get_pbc()):
        raise ValueError(f"The {label} structure must be fully periodic.")
    if abs(np.linalg.det(atoms.cell.array)) < 1e-14:
        raise ValueError(f"The {label} structure must have a non-singular cell.")


def _rotation_key(rotation: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(value) for value in rotation.reshape(-1))


def parent_point_operations(
    parent: Atoms, *, symprec: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return one fractional affine operation for each parent point operation.

    The parent structure fixes both the parent space group and the Cartesian
    coordinate frame in which the variants are generated.  Operations sharing
    a rotation differ only by a parent translation and therefore do not create
    separate orientation variants.
    """
    _require_periodic_cell(parent, "parent")
    dataset = ase2spglib_dataset(parent, symprec=symprec)
    if dataset is None:
        raise ValueError("spglib could not determine the parent space group.")

    operations: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]] = {}
    for rotation, translation in zip(dataset.rotations, dataset.translations):
        rotation = np.asarray(rotation, dtype=int)
        operations.setdefault(
            _rotation_key(rotation), (rotation, np.asarray(translation, dtype=float))
        )

    identity = np.eye(3, dtype=int)
    return sorted(
        operations.values(),
        key=lambda operation: 0 if np.array_equal(operation[0], identity) else 1,
    )


def _minimum_image_distances(
    reference_fractional: np.ndarray, candidate_fractional: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Return all pair distances subject to the periodic minimum-image convention."""
    displacements = candidate_fractional[:, np.newaxis, :] - reference_fractional[np.newaxis, :, :]
    cartesian_displacements = displacements.reshape((-1, 3)) @ cell
    _, distances = find_mic(cartesian_displacements, cell, pbc=True)
    return distances.reshape(displacements.shape[:2])


def structures_match(reference: Atoms, candidate: Atoms, *, symprec: float) -> bool:
    """Return whether two periodic structures match in the same cell basis.

    Atom order is intentionally ignored, while a common translation is not:
    it can distinguish parent-related antiphase structures when present.
    """
    if len(reference) != len(candidate) or sorted(reference.numbers) != sorted(candidate.numbers):
        return False
    if not np.allclose(reference.cell.array, candidate.cell.array, atol=symprec, rtol=0.0):
        return False

    reference_fractional = reference.get_scaled_positions(wrap=True)
    candidate_fractional = candidate.get_scaled_positions(wrap=True)
    for number in np.unique(reference.numbers):
        reference_indices = np.flatnonzero(reference.numbers == number)
        candidate_indices = np.flatnonzero(candidate.numbers == number)
        if len(reference_indices) != len(candidate_indices):
            return False
        distances = _minimum_image_distances(
            reference_fractional[reference_indices],
            candidate_fractional[candidate_indices],
            reference.cell.array,
        )
        rows, columns = linear_sum_assignment(distances)
        if not np.all(distances[rows, columns] <= symprec):
            return False
    return True


def _transformed_variant(
    atoms: Atoms,
    parent_cell: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Atoms:
    """Apply one parent affine operation and express its cell in parent axes."""
    parent_cell_inverse = np.linalg.inv(parent_cell)
    deformation = atoms.cell.array @ parent_cell_inverse
    variant_cell = rotation @ deformation @ rotation.T @ parent_cell
    parent_fractional_positions = atoms.get_positions() @ parent_cell_inverse
    positions = (parent_fractional_positions @ rotation.T + translation) @ parent_cell

    variant = atoms.copy()
    variant.set_cell(variant_cell, scale_atoms=False)
    variant.set_positions(positions, apply_constraint=False)
    variant.set_scaled_positions(variant.get_scaled_positions(wrap=False) % 1.0)
    return variant


def generate_domain_variants(atoms: Atoms, parent: Atoms, *, symprec: float) -> List[Atoms]:
    """Generate the unique orientation variants of ``atoms`` under ``parent`` symmetry.

    ``atoms`` and ``parent`` must use corresponding lattice directions and the
    same origin.  The source structure need not have the parent lattice
    parameters, but its cell must represent the same primitive-cell setting.
    The output uses the parent Cartesian axes, so a tetragonal distortion along
    ``z`` of a cubic parent produces variants with its long axis along each of
    the three cubic axes and both polar directions.
    """
    _require_periodic_cell(atoms, "input")
    parent_cell = parent.cell.array
    variants: List[Atoms] = []
    for rotation, translation in parent_point_operations(parent, symprec=symprec):
        variant = _transformed_variant(atoms, parent_cell, rotation, translation)
        if not any(structures_match(existing, variant, symprec=symprec) for existing in variants):
            variants.append(variant)
    return variants
