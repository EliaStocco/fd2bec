"""Translation-invariant matching and reordering of ASE structures."""

import numpy as np
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import wrap

CELL_ATOL = 1e-10


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


def sort_atoms_like(reference: Atoms, candidate: Atoms, atol: float) -> Atoms:
    """Align and reorder ``candidate`` to correspond to ``reference``.

    Every candidate atom having the same species as reference atom 0 is tried
    as the translation anchor. After aligning that atom to reference atom 0,
    atoms are matched by species and position. The valid alignment with the
    smallest total squared correspondence distance is retained.
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
    return ordered
