from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from numpy.typing import ArrayLike

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


@dataclass(frozen=True)
class GammaCharacterTable:
    """Finite character table of a crystallographic point group at Gamma."""

    class_labels: Tuple[str, ...]
    class_sizes: Tuple[int, ...]
    class_representatives: np.ndarray
    characters: np.ndarray
    dimensions: Tuple[int, ...]


def _rotation_key(rotation: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(value) for value in rotation.reshape(-1))


def _unique_rotations(rotations: ArrayLike) -> np.ndarray:
    values = np.asarray(rotations)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError(f"Rotations must have shape (N, 3, 3), got {values.shape}.")
    if not np.allclose(values, np.rint(values), atol=1e-8, rtol=0.0):
        raise ValueError("Crystallographic rotations must contain integer entries.")
    unique = {}
    for rotation in np.rint(values).astype(int):
        unique.setdefault(_rotation_key(rotation), rotation)
    if not unique:
        raise ValueError("At least one rotation is required.")
    return np.asarray(list(unique.values()), dtype=int)


def cayley_table(rotations: np.ndarray) -> np.ndarray:
    "Returns the Cayley table of a Point group given the rotation operations."
    indices = {_rotation_key(rotation): index for index, rotation in enumerate(rotations)}
    table = np.empty((len(rotations), len(rotations)), dtype=int)
    for left, left_rotation in enumerate(rotations):
        for right, right_rotation in enumerate(rotations):
            try:
                table[left, right] = indices[_rotation_key(left_rotation @ right_rotation)]
            except KeyError as error:
                raise ValueError(
                    "The supplied rotations are not closed under multiplication."
                ) from error
    return table


def _identity_and_inverses(table: np.ndarray) -> Tuple[int, np.ndarray]:
    """Find the identity and inverse of each element in a Cayley table.

    ``table[left, right]`` is assumed to contain the Cayley table of the point group,
    and it should have been negerated by the function ``cayley_table``.
    It return the index of the identiy element, and for each operation the index of its inverse.

    The identity is the unique index whose row and column both equal ``range(len(table))``.
    While for each element, its inverse is the unique element whose product with it,
    in either order, is the identity.

    Parameters
    ----------
    table
        Cayley table of a finite group.

    Returns
    -------
    identity
        Index of the identity element.
    inverses
        Integer array such that ``inverses[index]`` is the index of the inverse
        of ``index``.

    Raises
    ------
    ValueError
        If the table does not contain exactly one identity, or if any element
        does not have exactly one two-sided inverse.

    Examples
    --------
    The cyclic group of order three has identity 0, while elements 1 and 2 are
    inverses of each other:

    >>> table = np.array([[0, 1, 2],
    ...                   [1, 2, 0],
    ...                   [2, 0, 1]])
    >>> identity, inverses = _identity_and_inverses(table)
    >>> int(identity)
    0
    >>> inverses
    array([0, 2, 1])
    """
    indices = np.arange(len(table))
    identities = [
        index
        for index in indices
        if np.array_equal(table[index], indices) and np.array_equal(table[:, index], indices)
    ]
    if len(identities) != 1:
        raise ValueError("The rotation group must contain exactly one identity.")
    identity = identities[0]
    inverses = np.empty(len(table), dtype=int)
    for index in indices:
        candidates = np.flatnonzero((table[index] == identity) & (table[:, index] == identity))
        if len(candidates) != 1:
            raise ValueError("Every rotation must have exactly one inverse.")
        inverses[index] = candidates[0]
    return identity, inverses


def _conjugacy_classes(
    table: np.ndarray, inverses: np.ndarray
) -> Tuple[List[Tuple[int, ...]], int, np.ndarray]:
    """Partition a finite group into conjugacy classes.

    ``table[left, right]`` is assumed to contain the index of the product of
    the elements indexed by ``left`` and ``right``.  After finding the identity
    and inverse of every element, the function constructs the class of each
    representative ``r`` from all conjugates ``g r g^-1``.

    Parameters
    ----------
    table
        Cayley table of a finite group.

    Returns
    -------
    classes
        Conjugacy classes as sorted tuples of element indices.  Classes are
        ordered by the smallest element not assigned to an earlier class.
    identity
        Index of the identity element.
    inverses
        Integer array mapping each element index to its inverse index.
    """
    remaining = set(range(len(table)))
    classes = []
    while remaining:
        representative = min(remaining)
        conjugates: Tuple[int] = {
            int(table[table[element, representative], inverses[element]])  # convert np.int64 to int
            for element in range(len(table))
        }
        conjugacy_class = tuple(sorted(conjugates))
        classes.append(conjugacy_class)
        remaining.difference_update(conjugates)
    return classes


def _rotation_order(index: int, table: np.ndarray, identity: int) -> int:
    product = identity
    for order in range(1, len(table) + 1):
        product = int(table[product, index])
        if product == identity:
            return order
    raise ValueError("Could not determine the order of a rotation.")


def _class_base_label(rotation: np.ndarray, order: int) -> str:
    """Return a crystallographic base label for a rotation operation.

    The labels use standard point-group notation: ``E`` is the identity,
    ``i`` is inversion through the origin, ``C<n>`` is a proper ``n``-fold
    rotation, ``m`` is a mirror reflection, and ``S<n>`` is an improper
    ``n``-fold rotation (a rotation combined with a reflection).

    Parameters
    ----------
    rotation
        Integer 3-by-3 matrix representing the operation.
    order
        Smallest positive integer ``n`` for which the operation raised to
        ``n`` is the identity.

    Returns
    -------
    label
        Base label describing the type and order of the operation.

    Raises
    ------
    ValueError
        If ``rotation`` does not have determinant +1 or -1.
    """
    identity = np.eye(3, dtype=int)
    # E denotes the identity operation (from the German "Einheit").
    if np.array_equal(rotation, identity):
        return "E"
    # i denotes inversion: every position vector r is mapped to -r.
    if np.array_equal(rotation, -identity):
        return "i"
    determinant = int(round(np.linalg.det(rotation)))
    # C<n> denotes a proper n-fold rotation, which preserves handedness.
    if determinant == 1:
        return f"C{order}"
    # m denotes a mirror reflection; its eigenvalues are 1, 1, and -1.
    if determinant == -1 and order == 2 and int(np.trace(rotation)) == 1:
        return "m"
    # S<n> denotes any remaining improper n-fold rotation (roto-reflection).
    if determinant == -1:
        return f"S{order}"
    raise ValueError("A crystallographic rotation must have determinant +1 or -1.")


def _sort_and_label_classes(
    classes: Sequence[Tuple[int, ...]],
    rotations: np.ndarray,
    table: np.ndarray,
    identity: int,
) -> Tuple[List[Tuple[int, ...]], Tuple[str, ...]]:
    """Sort conjugacy classes and assign crystallographic operation labels.

    The identity class is placed first.  The remaining classes are sorted by
    their base label (such as ``C2``, ``m``, or ``i``) and then by the flattened
    entries of their representative rotation.  When several classes have the
    same base label, one-based suffixes such as ``C2(1)`` and ``C2(2)`` are
    added to distinguish them.

    Parameters
    ----------
    classes
        Conjugacy classes, each represented by a sequence of rotation indices.
    rotations
        Rotation matrices indexed by the entries in ``classes``.
    table
        Cayley table for ``rotations``.
    identity
        Index of the identity rotation.

    Returns
    -------
    sorted_classes
        Conjugacy classes in display order.
    labels
        Display label corresponding to each returned class.
    """
    decorated = []
    for conjugacy_class in classes:
        representative = conjugacy_class[0]
        order = _rotation_order(representative, table, identity)
        base = _class_base_label(rotations[representative], order)
        key = (
            0 if representative == identity else 1,
            base,
            _rotation_key(rotations[representative]),
        )
        decorated.append((key, conjugacy_class, base))
    decorated.sort(key=lambda item: item[0])

    totals = {}
    for _, _, base in decorated:
        totals[base] = totals.get(base, 0) + 1
    seen = {}
    labels = []
    sorted_classes = []
    for _, conjugacy_class, base in decorated:
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if totals[base] == 1 else f"{base}({seen[base]})")
        sorted_classes.append(conjugacy_class)
    return sorted_classes, tuple(labels)


def _left_regular_representation(table: np.ndarray) -> np.ndarray:
    """Construct the left regular representation from a Cayley table.
    https://en.wikipedia.org/wiki/Regular_representation

    In simple terms, this creates one permutation matrix for each group
    element.  Each matrix records how multiplying by that element rearranges
    all the group elements, so the complete collection of matrices encodes the
    multiplication table as linear algebra.

    For a group element ``g``, ``regular[g]`` is the permutation matrix that
    maps the basis vector for an element ``h`` to the basis vector for ``g h``.
    Equivalently, ``regular[g, table[g, h], h]`` equals one; all other entries
    in that column are zero.

    Parameters
    ----------
    table
        Cayley table of a finite group, with ``table[g, h]`` equal to the index
        of the product ``g h``.

    Returns
    -------
    regular
        Array of shape ``(N, N, N)``, where ``N`` is the group order and
        ``regular[g]`` is the representation matrix for element ``g``.

    Examples
    --------
    For the cyclic group of order three used above, element 0 is the identity
    and element 1 cyclically permutes the three group elements under left
    multiplication:

    >>> table = np.array([[0, 1, 2],
    ...                   [1, 2, 0],
    ...                   [2, 0, 1]])
    >>> regular = _left_regular_representation(table)
    >>> regular[0]
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> regular[1]
    array([[0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])
    >>> regular[2]
    array([[0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.]])
    """
    order = len(table)
    regular = np.zeros((order, order, order), dtype=float)
    columns = np.arange(order)
    for element in range(order):
        regular[element, table[element], columns] = 1.0
    return regular


def _eigenvalue_clusters(values: np.ndarray, tolerance: float) -> List[np.ndarray]:
    clusters = []
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or abs(values[stop] - values[start]) > tolerance:
            clusters.append(np.arange(start, stop))
            start = stop
    return clusters


def _isotypic_subspaces(
    class_sums: Sequence[np.ndarray], number_of_classes: int
) -> List[np.ndarray]:
    """Find one subspace containing all copies of each irreducible representation.

    Class-sum operators commute and therefore have common eigenspaces.  Starting
    with the full regular-representation space, this function uses each class
    sum in turn to split the current subspaces by eigenvalue.  The final common
    eigenspaces are the isotypic subspaces.

    Both Hermitian parts of every class sum are used so that complex-conjugate
    irreducible representations can be distinguished with ``numpy.linalg.eigh``.
    """
    order = class_sums[0].shape[0]
    subspaces = [np.eye(order, dtype=complex)]

    for class_sum in class_sums:
        adjoint = class_sum.conj().T
        hermitian_parts = (
            class_sum + adjoint,
            1j * (class_sum - adjoint),
        )
        for operator in hermitian_parts:
            refined_subspaces = []
            for subspace in subspaces:
                # Express this class-sum operator inside the current subspace.
                restricted = subspace.conj().T @ operator @ subspace
                eigenvalues, eigenvectors = np.linalg.eigh(restricted)
                scale = max(1.0, float(np.max(np.abs(eigenvalues))))
                clusters = _eigenvalue_clusters(eigenvalues, tolerance=1e-8 * scale)

                if len(clusters) == 1:
                    refined_subspaces.append(subspace)
                    continue

                # Each eigenvalue cluster is a smaller common eigenspace.
                refined_subspaces.extend(
                    subspace @ eigenvectors[:, cluster] for cluster in clusters
                )
            subspaces = refined_subspaces

    dimensions = [int(round(np.sqrt(subspace.shape[1]))) for subspace in subspaces]
    if len(subspaces) != number_of_classes or not all(
        dimension * dimension == subspace.shape[1]
        for dimension, subspace in zip(dimensions, subspaces)
    ):
        raise RuntimeError("Could not separate all point-group irreducible representations.")
    return subspaces


def _clean_characters(characters: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    values = np.asarray(characters, dtype=complex).copy()
    values.real[np.abs(values.real) < tolerance] = 0.0
    values.imag[np.abs(values.imag) < tolerance] = 0.0
    rounded_real = np.rint(values.real)
    rounded_imag = np.rint(values.imag)
    values.real[np.abs(values.real - rounded_real) < tolerance] = rounded_real[
        np.abs(values.real - rounded_real) < tolerance
    ]
    values.imag[np.abs(values.imag - rounded_imag) < tolerance] = rounded_imag[
        np.abs(values.imag - rounded_imag) < tolerance
    ]
    return values


def gamma_character_table(rotations: ArrayLike) -> GammaCharacterTable:
    """Compute the complex Γ-point character table from crystallographic rotations.

    Translational parts of space-group operations act trivially at Γ, so the
    finite table is the character table of the crystallographic point group.
    Duplicate rotations, such as those arising from centered cells, are removed.
    """
    # Remove duplicate point operations, then encode their multiplication using
    # integer indices in a Cayley table.
    unique_rotations = _unique_rotations(rotations)
    table = cayley_table(unique_rotations)

    # Characters are constant on conjugacy classes, which will become the
    # columns of the character table.
    identity, inverses = _identity_and_inverses(table)
    classes = _conjugacy_classes(table, inverses)
    classes, labels = _sort_and_label_classes(classes, unique_rotations, table, identity)
    class_sizes = tuple(len(conjugacy_class) for conjugacy_class in classes)

    # The regular representation is available directly from the Cayley table
    # and contains every d-dimensional irrep exactly d times. # ToDo: this is not really clear why to me
    regular = _left_regular_representation(table)
    # Summing over a conjugacy class gives a central operator.  Its common
    # eigenspaces with the other class sums collect all copies of one irrep.
    class_sums = [np.sum(regular[list(conjugacy_class)], axis=0) for conjugacy_class in classes]
    subspaces = _isotypic_subspaces(class_sums, len(classes))

    # Convert each isotypic subspace into one row of irreducible characters.
    rows = []
    for subspace in subspaces:
        # A d-dimensional irrep occurs d times in the regular representation,
        # so its isotypic subspace has dimension d squared.
        isotypic_dimension = subspace.shape[1]
        dimension = int(round(np.sqrt(isotypic_dimension)))
        characters = []
        for conjugacy_class, class_sum in zip(classes, class_sums):
            # On this subspace a class sum acts as one scalar lambda.  The
            # normalized trace extracts lambda, and
            # chi(C) = lambda * d / |C| recovers the character.
            restricted = subspace.conj().T @ class_sum @ subspace
            eigenvalue = np.trace(restricted) / isotypic_dimension
            characters.append(eigenvalue * dimension / len(conjugacy_class))
        rows.append(_clean_characters(np.asarray(characters)))

    character_rows = np.asarray(rows, dtype=complex)
    # The totally symmetric irrep has character 1 in every class.  Put it first
    # and use dimensions and character values to order the remaining rows.
    trivial = np.flatnonzero(np.all(np.isclose(character_rows, 1.0, atol=1e-8), axis=1))
    if len(trivial) != 1:
        raise RuntimeError("Could not identify the totally symmetric representation.")

    def row_key(index: int) -> tuple:
        row = character_rows[index]
        rounded = tuple(
            value
            for character in row
            for value in (round(float(character.real), 10), round(float(character.imag), 10))
        )
        return (0 if index == trivial[0] else 1, int(round(character_rows[index, 0].real)), rounded)

    order = sorted(range(len(character_rows)), key=row_key)
    character_rows = character_rows[order]
    # The character of the identity operation equals the irrep dimension.
    dimensions = tuple(int(round(character_rows[index, 0].real)) for index in range(len(order)))

    # Store one concrete rotation matrix to represent each character-table column.
    class_representatives = np.asarray(
        [unique_rotations[conjugacy_class[0]] for conjugacy_class in classes],
        dtype=int,
    )

    # Check character orthogonality and the regular-representation dimension
    # identity |G| = sum(d squared) before returning the numerical result.
    weights = np.asarray(class_sizes, dtype=float)
    gram = (character_rows * weights) @ character_rows.conj().T
    if not np.allclose(gram, len(unique_rotations) * np.eye(len(character_rows)), atol=1e-7):
        raise RuntimeError("Computed characters do not satisfy row orthogonality.")
    if sum(dimension * dimension for dimension in dimensions) != len(unique_rotations):
        raise RuntimeError("Irreducible-representation dimensions do not sum to the group order.")

    # Prevent callers from accidentally changing the computed table in place.
    character_rows.setflags(write=False)
    class_representatives.setflags(write=False)
    return GammaCharacterTable(
        class_labels=labels,
        class_sizes=class_sizes,
        class_representatives=class_representatives,
        characters=character_rows,
        dimensions=dimensions,
    )


def as_text(value: object) -> str:
    """Normalize text returned by different spglib versions."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def format_character(value: complex, tolerance: float = 1e-9) -> str:
    """Format a numerical character compactly without hiding complex values."""
    real = 0.0 if abs(value.real) < tolerance else float(value.real)
    imaginary = 0.0 if abs(value.imag) < tolerance else float(value.imag)
    if abs(real - round(real)) < tolerance:
        real = float(round(real))
    if abs(imaginary - round(imaginary)) < tolerance:
        imaginary = float(round(imaginary))
    if imaginary == 0.0:
        return f"{real:g}"
    if real == 0.0:
        return f"{imaginary:g}i"
    sign = "+" if imaginary > 0.0 else "-"
    return f"{real:g}{sign}{abs(imaginary):g}i"


def character_table_frame(table: GammaCharacterTable) -> pd.DataFrame:
    """Return a display-oriented character table."""
    columns = [f"{size} x {label}" for size, label in zip(table.class_sizes, table.class_labels)]
    rows = [
        f"Γ{number}  (d={dimension})" for number, dimension in enumerate(table.dimensions, start=1)
    ]
    values = [
        [format_character(value) for value in character_row] for character_row in table.characters
    ]
    return pd.DataFrame(values, index=rows, columns=columns)


def is_sohncke(sg_number):
    """Check if the space group number corresponds to a Sohncke group."""
    return sg_number in SOHNCKE_GROUPS


def symmetrize_bec(structure: Atoms, bec: np.ndarray, symprec: float = SYMPREC) -> np.ndarray:
    from fd2bec.atomic import AtomicStructure
    from fd2bec.tensor import BornCharges

    tensor = BornCharges(data=bec)
    atomic_structure = AtomicStructure.from_ase(structure, symprec=symprec)
    return atomic_structure.symmetrize(tensor=tensor).data
