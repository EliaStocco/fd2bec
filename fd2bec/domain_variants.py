"""Generate and classify parent-symmetry-related orientation-domain structures."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

from fd2bec.tools import ase2spglib_dataset


@dataclass(frozen=True)
class InequivalentStructurePair:
    """One representative pair and every source-index pair equivalent to it."""

    first_index: int
    second_index: int
    equivalent_pairs: Tuple[Tuple[int, int], ...]


def _require_periodic_cell(atoms: Atoms, label: str) -> None:
    if not np.all(atoms.get_pbc()):
        raise ValueError(f"The {label} structure must be fully periodic.")
    if abs(np.linalg.det(atoms.cell.array)) < 1e-14:
        raise ValueError(f"The {label} structure must have a non-singular cell.")


def canonicalize_lattice_orientation(atoms: Atoms, parent: Atoms) -> Atoms:
    """Express ``atoms`` in the parent frame without a rigid lattice rotation.

    Some relaxation codes store a strained cell in a triangular lattice basis.
    Its polar decomposition includes a rigid rotation that is a representation
    choice rather than a domain orientation. Removing it makes the remaining
    deformation tensor directly comparable under parent symmetry operations.
    """
    _require_periodic_cell(atoms, "input")
    _require_periodic_cell(parent, "parent")
    deformation = atoms.cell.array.T @ np.linalg.inv(parent.cell.array.T)
    left, _, right = np.linalg.svd(deformation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right

    canonical = atoms.copy()
    canonical.set_cell(atoms.cell.array @ rotation, scale_atoms=False)
    canonical.set_positions(atoms.get_positions() @ rotation, apply_constraint=False)
    return canonical


def _rotation_key(rotation: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(value) for value in rotation.reshape(-1))


def _canonical_translation(
    translation: np.ndarray, parent_cell: np.ndarray, symprec: float
) -> np.ndarray:
    """Wrap parent translations and remove round-off at a lattice vector."""
    shortest_vector = np.min(np.linalg.norm(parent_cell, axis=1))
    fractional_tolerance = symprec / shortest_vector
    translation = np.mod(np.asarray(translation, dtype=float), 1.0)
    translation[np.isclose(translation, 0.0, atol=fractional_tolerance, rtol=0.0)] = 0.0
    translation[np.isclose(translation, 1.0, atol=fractional_tolerance, rtol=0.0)] = 0.0
    return translation


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
        translation = _canonical_translation(translation, parent.cell.array, symprec)
        operations.setdefault(_rotation_key(rotation), (rotation, translation))

    identity = np.eye(3, dtype=int)
    return sorted(
        operations.values(),
        key=lambda operation: 0 if np.array_equal(operation[0], identity) else 1,
    )


def common_space_group_symbol(structures: Sequence[Atoms], *, symprec: float) -> str:
    """Return the common detected space-group symbol of generated variants."""
    if not structures:
        raise ValueError("At least one generated structure is required.")

    groups = set()
    for structure in structures:
        dataset = ase2spglib_dataset(structure, symprec=symprec)
        if dataset is None:
            raise ValueError("spglib could not determine a generated structure's space group.")
        symbol = (
            dataset.international.decode()
            if isinstance(dataset.international, bytes)
            else str(dataset.international)
        )
        groups.add((int(dataset.number), symbol))
    if len(groups) != 1:
        description = ", ".join(f"{symbol} (No. {number})" for number, symbol in sorted(groups))
        raise ValueError(f"Generated structures have different space groups: {description}.")
    return next(iter(groups))[1]


def validate_atom_arrays(atoms: Atoms) -> None:
    """Ensure every per-atom array has a writer-safe, atom-aligned shape."""
    for name, values in atoms.arrays.items():
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(atoms):
            raise ValueError(
                f"atoms.arrays[{name!r}] must have one leading entry for each of "
                f"the {len(atoms)} atoms; got shape {values.shape}."
            )
        if values.dtype.hasobject:
            raise ValueError(
                f"atoms.arrays[{name!r}] has object dtype, which ASE structure writers "
                "cannot serialize reliably."
            )


def _minimum_image_distances(
    reference_fractional: np.ndarray, candidate_fractional: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Return all pair distances subject to the periodic minimum-image convention."""
    displacements = candidate_fractional[:, np.newaxis, :] - reference_fractional[np.newaxis, :, :]
    cartesian_displacements = displacements.reshape((-1, 3)) @ cell
    _, distances = find_mic(cartesian_displacements, cell, pbc=True)
    return distances.reshape(displacements.shape[:2])


def _fractional_positions_match(
    reference_fractional: np.ndarray,
    candidate_fractional: np.ndarray,
    reference_numbers: np.ndarray,
    candidate_numbers: np.ndarray,
    cell: np.ndarray,
    symprec: float,
) -> bool:
    """Return whether two fractional-coordinate sets match by chemical species."""
    for number in np.unique(reference_numbers):
        reference_indices = np.flatnonzero(reference_numbers == number)
        candidate_indices = np.flatnonzero(candidate_numbers == number)
        distances = _minimum_image_distances(
            reference_fractional[reference_indices],
            candidate_fractional[candidate_indices],
            cell,
        )
        rows, columns = linear_sum_assignment(distances)
        if not np.all(distances[rows, columns] <= symprec):
            return False
    return True


def structures_match(
    reference: Atoms,
    candidate: Atoms,
    *,
    symprec: float,
    allow_global_translation: bool = False,
) -> bool:
    """Return whether two periodic structures match in the same cell basis.

    Atom order is ignored. Set ``allow_global_translation`` when only the
    orientation matters; it also treats structures with different choices of
    periodic origin as equal. The default retains such translations so callers
    can distinguish parent-related antiphase structures.
    """
    if len(reference) != len(candidate) or sorted(reference.numbers) != sorted(candidate.numbers):
        return False
    if not np.allclose(reference.cell.array, candidate.cell.array, atol=symprec, rtol=0.0):
        return False

    reference_fractional = reference.get_scaled_positions(wrap=True)
    candidate_fractional = candidate.get_scaled_positions(wrap=True)
    if _fractional_positions_match(
        reference_fractional,
        candidate_fractional,
        reference.numbers,
        candidate.numbers,
        reference.cell.array,
        symprec,
    ):
        return True
    if not allow_global_translation:
        return False

    anchor = 0
    matching_candidates = np.flatnonzero(candidate.numbers == reference.numbers[anchor])
    for candidate_index in matching_candidates:
        translation = reference_fractional[anchor] - candidate_fractional[candidate_index]
        translated = (candidate_fractional + translation) % 1.0
        if _fractional_positions_match(
            reference_fractional,
            translated,
            reference.numbers,
            candidate.numbers,
            reference.cell.array,
            symprec,
        ):
            return True
    return False


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


def _matching_structure_index(
    structure: Atoms, structures: Sequence[Atoms], *, symprec: float
) -> int:
    """Return the index of a symmetry-transformed structure in an input set."""
    for index, candidate in enumerate(structures):
        if structures_match(candidate, structure, symprec=symprec):
            return index
    raise ValueError(
        "Each input file must contain a complete set of parent-symmetry-related variants."
    )


def inequivalent_structure_pairs(
    first_structures: Sequence[Atoms],
    second_structures: Sequence[Atoms],
    parent: Atoms,
    *,
    symprec: float,
    treat_reversed_as_equivalent: bool = False,
    include_identical: bool = False,
) -> List[InequivalentStructurePair]:
    """Classify structure pairs under simultaneous parent symmetry operations.

    The same parent operation is applied to both endpoints. When
    ``treat_reversed_as_equivalent`` is true, pairs are unordered and therefore
    suitable for comparing two variants from the same input file. Identical
    endpoint pairs are omitted unless ``include_identical`` is true.
    """
    if not first_structures or not second_structures:
        raise ValueError("Both inputs must contain at least one structure.")

    first = [canonicalize_lattice_orientation(atoms, parent) for atoms in first_structures]
    second = [canonicalize_lattice_orientation(atoms, parent) for atoms in second_structures]
    reference_numbers = first[0].numbers
    for label, structures in (("first", first), ("second", second)):
        for structure in structures:
            if not np.array_equal(structure.numbers, reference_numbers):
                raise ValueError(
                    f"Every {label} endpoint must have the same atom ordering and composition."
                )

    if treat_reversed_as_equivalent:
        if len(first) != len(second) or any(
            not structures_match(left, right, symprec=symprec) for left, right in zip(first, second)
        ):
            raise ValueError(
                "Reversed-pair equivalence requires the two inputs to contain the same "
                "structures in the same order."
            )
        start = 0 if include_identical else 1
        pending = {
            (first_index, second_index)
            for first_index in range(len(first))
            for second_index in range(first_index + start, len(second))
        }
    else:
        pending = {
            (first_index, second_index)
            for first_index in range(len(first))
            for second_index in range(len(second))
        }

    operations = parent_point_operations(parent, symprec=symprec)
    pairs: List[InequivalentStructurePair] = []
    while pending:
        first_index, second_index = min(pending)
        equivalent = set()
        for rotation, translation in operations:
            transformed_first = _transformed_variant(
                first[first_index], parent.cell.array, rotation, translation
            )
            transformed_second = _transformed_variant(
                second[second_index], parent.cell.array, rotation, translation
            )
            mapped_first = _matching_structure_index(transformed_first, first, symprec=symprec)
            mapped_second = _matching_structure_index(transformed_second, second, symprec=symprec)
            if treat_reversed_as_equivalent:
                equivalent.add(tuple(sorted((mapped_first, mapped_second))))
            else:
                equivalent.add((mapped_first, mapped_second))
        pending.difference_update(equivalent)
        pairs.append(
            InequivalentStructurePair(
                first_index=first_index,
                second_index=second_index,
                equivalent_pairs=tuple(sorted(equivalent)),
            )
        )
    return pairs


def generate_domain_variants(atoms: Atoms, parent: Atoms, *, symprec: float) -> List[Atoms]:
    """Generate the unique orientation variants of ``atoms`` under ``parent`` symmetry.

    ``atoms`` and ``parent`` must use corresponding lattice directions and the
    same origin. The source structure need not have the parent lattice
    parameters, but its cell must represent the same primitive-cell setting.
    A rigid rotation introduced by a triangular lattice basis is removed before
    generating variants. The output uses the parent Cartesian axes, so a
    tetragonal distortion along ``z`` of a cubic parent produces variants with
    its long axis along each of the three cubic axes and both polar directions.
    """
    _require_periodic_cell(atoms, "input")
    validate_atom_arrays(atoms)
    parent_cell = parent.cell.array
    canonical_atoms = canonicalize_lattice_orientation(atoms, parent)
    variants: List[Atoms] = []
    for rotation, translation in parent_point_operations(parent, symprec=symprec):
        variant = _transformed_variant(canonical_atoms, parent_cell, rotation, translation)
        if not any(structures_match(existing, variant, symprec=symprec) for existing in variants):
            validate_atom_arrays(variant)
            variants.append(variant)
    return variants
