"""Translation-invariant matching, reordering, and serialization of ASE structures."""

import json
from pathlib import Path

import numpy as np
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import wrap

CELL_ATOL = 1e-10
SORTING_MAP_FORMAT = "fd2bec sorting map"
SORTING_MAP_VERSION = 1


def is_ase_standard_cell(atoms: Atoms, atol: float = CELL_ATOL) -> bool:
    """Return whether a periodic cell is in ASE's lower-triangular standard form."""
    if not np.all(atoms.pbc):
        return True

    standard_cell, _ = atoms.cell.standard_form()
    return np.allclose(atoms.cell.array, standard_cell.array, rtol=0.0, atol=atol)


def require_ase_standard_cell(reference: Atoms) -> None:
    """Raise an actionable error when a periodic reference cell is rotated."""
    if not is_ase_standard_cell(reference):
        raise ValueError(
            "Reference cell is not in ASE's lower-triangular standard form. "
            "Run `rotate_cell -i <reference> -o <rotated-reference>` first."
        )


def _translated_to_anchor(
    reference: Atoms, candidate: Atoms, anchor: int, *, periodic: bool
) -> Atoms:
    """Return a copy with candidate ``anchor`` translated onto reference atom 0."""
    aligned = candidate.copy()
    if periodic:
        reference_positions = reference.get_scaled_positions(wrap=False)
        candidate_positions = candidate.get_scaled_positions(wrap=False)
        candidate_positions += reference_positions[0] - candidate_positions[anchor]
        aligned.set_scaled_positions(candidate_positions)
    else:
        positions = candidate.get_positions()
        positions += reference.positions[0] - positions[anchor]
        aligned.set_positions(positions, apply_constraint=False)
    return aligned


def _mapping_score(reference: Atoms, candidate: Atoms, mapping, *, periodic: bool) -> float:
    """Return the squared correspondence distance for an established mapping."""
    if periodic:
        reference_positions = reference.get_scaled_positions(wrap=False)
        candidate_positions = candidate.get_scaled_positions(wrap=False)
        displacement = wrap(candidate_positions - reference_positions[mapping])
    else:
        displacement = candidate.positions - reference.positions[mapping]
    return float(np.sum(displacement**2))


def sort_atoms_like_with_indices(reference: Atoms, candidate: Atoms, atol: float):
    """Align and reorder ``candidate`` to correspond to ``reference``.

    Every candidate atom having the same species as reference atom 0 is tried
    as the translation anchor. After aligning that atom to reference atom 0,
    atoms are matched by species and position. The valid alignment with the
    smallest total squared correspondence distance is retained.  Return both
    the aligned and ordered structure and the indices that select its atoms
    from the original candidate.
    """
    require_ase_standard_cell(reference)
    reference_structure = AtomicStructure.from_ase(reference)
    candidate_structure = AtomicStructure.from_ase(candidate)
    if len(reference_structure) != len(candidate_structure):
        raise ValueError(
            f"Structures have different numbers of atoms: "
            f"{len(reference_structure)} and {len(candidate_structure)}."
        )
    if reference_structure.pbc != candidate_structure.pbc:
        raise ValueError("Cannot match periodic and non-periodic structures.")
    if sorted(reference_structure.symbols) != sorted(candidate_structure.symbols):
        raise ValueError("Structures have different chemical compositions.")

    periodic = reference_structure.pbc
    anchor_symbol = reference.get_chemical_symbols()[0]
    anchors = np.flatnonzero(np.asarray(candidate.get_chemical_symbols()) == anchor_symbol)
    best = None
    failures = []
    for anchor in anchors:
        aligned = _translated_to_anchor(reference, candidate, int(anchor), periodic=periodic)
        aligned_structure = AtomicStructure.from_ase(aligned)
        try:
            mapping = reference_structure.get_atoms_mapping(aligned_structure, atol=atol)
        except ValueError as error:
            failures.append(str(error))
            continue
        if mapping[anchor] != 0:
            failures.append(
                f"Candidate anchor {anchor} mapped to reference atom {mapping[anchor]}, not 0."
            )
            continue
        score = _mapping_score(reference, aligned, mapping, periodic=periodic)
        if best is None or score < best[0]:
            best = score, mapping, aligned

    if best is None:
        detail = failures[0] if failures else "No candidate atom has the anchor species."
        raise ValueError(f"Could not align and match the structures. {detail}")

    _, mapping, aligned = best
    ordered = aligned[np.argsort(mapping)]
    if periodic:
        reference_positions = reference.get_scaled_positions(wrap=False)
        candidate_positions = ordered.get_scaled_positions(wrap=False)
        displacement = wrap(candidate_positions - reference_positions)
        ordered.set_scaled_positions(reference_positions + displacement)
    return ordered, np.argsort(mapping)


def sort_atoms_like(reference: Atoms, candidate: Atoms, atol: float) -> Atoms:
    """Return ``candidate`` aligned and reordered to correspond to ``reference``."""
    ordered, _ = sort_atoms_like_with_indices(reference, candidate, atol)
    return ordered


def _structure_record(atoms: Atoms):
    """Return the human-readable structural context stored in a sorting map."""
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions_angstrom": atoms.get_positions().tolist(),
        "cell_angstrom": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def write_sorting_map(filename, reference: Atoms, candidate: Atoms, sorting_indices) -> None:
    """Write a sorting map that can reorder data XYZ files without matching positions.

    ``sorting_indices[reference_index]`` is the atom index in the original
    candidate that belongs at that reference index.  The candidate's symbols
    are retained as context only: data XYZ files are allowed to use different
    labels, provided they have the same number and order of atoms.
    """
    indices = np.asarray(sorting_indices)
    expected = np.arange(len(candidate))
    if (
        indices.ndim != 1
        or len(indices) != len(candidate)
        or not np.array_equal(np.sort(indices), expected)
    ):
        raise ValueError("Sorting indices must be a permutation of the candidate atom indices.")

    record = {
        "format": SORTING_MAP_FORMAT,
        "version": SORTING_MAP_VERSION,
        "index_convention": (
            "sorting_indices[reference_index] is the original input atom index "
            "for that reference atom"
        ),
        "reference_structure": _structure_record(reference),
        "source_structure": {"symbols": candidate.get_chemical_symbols()},
        "sorting_indices": indices.tolist(),
    }
    with Path(filename).open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def read_sorting_map(filename):
    """Read and validate a sorting map written by :func:`write_sorting_map`."""
    with Path(filename).open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or record.get("format") != SORTING_MAP_FORMAT:
        raise ValueError(f"{filename} is not an fd2bec sorting map.")
    if record.get("version") != SORTING_MAP_VERSION:
        raise ValueError(f"Unsupported sorting map version: {record.get('version')!r}.")

    indices = np.asarray(record.get("sorting_indices"))
    expected = np.arange(len(indices))
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or not np.array_equal(np.sort(indices), expected)
    ):
        raise ValueError("Sorting map contains invalid sorting indices.")
    return indices


def apply_sorting_map(atoms: Atoms, sorting_indices) -> Atoms:
    """Reorder an ASE structure using indices read from a sorting map.

    This deliberately performs no alignment or coordinate matching, so XYZ
    files that encode velocities or momenta as positions retain their values.
    """
    indices = np.asarray(sorting_indices)
    if len(atoms) != len(indices):
        raise ValueError(
            f"Sorting map has {len(indices)} atoms but the input has {len(atoms)} atoms."
        )
    return atoms[indices]
