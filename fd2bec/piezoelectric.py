"""Finite-strain evaluation of proper and improper piezoelectric tensors."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from ase import Atoms

from fd2bec.linear_system import LinearSystem
from fd2bec.tensor import ImproperPiezoelectricTensor, ProperPiezoelectricTensor

# Voigt order and engineering-shear convention:
# (xx, yy, zz, yz, xz, xy) = (εxx, εyy, εzz, 2εyz, 2εxz, 2εxy)
VOIGT_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)

# One e/Å² expressed in C/m². The elementary charge is exact in SI.
E_PER_ANGSTROM2_TO_C_PER_M2 = 16.02176634


def strain_to_voigt(strain: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Convert symmetric strain tensors to engineering-strain Voigt vectors."""
    strain = np.asarray(strain, dtype=float)
    if strain.shape[-2:] != (3, 3):
        raise ValueError(f"Strain must end in shape (3, 3), got {strain.shape}.")
    if not np.allclose(strain, np.swapaxes(strain, -1, -2), atol=atol):
        raise ValueError("Only symmetric strain tensors are supported.")

    voigt = np.empty((*strain.shape[:-2], 6), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        factor = 1.0 if j == k else 2.0
        voigt[..., column] = factor * strain[..., j, k]
    return voigt


def voigt_to_strain(voigt: np.ndarray) -> np.ndarray:
    """Convert engineering-strain Voigt vectors to symmetric 3x3 tensors."""
    voigt = np.asarray(voigt, dtype=float)
    if voigt.shape[-1:] != (6,):
        raise ValueError(f"Voigt strain must end in shape (6,), got {voigt.shape}.")

    strain = np.zeros((*voigt.shape[:-1], 3, 3), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        value = voigt[..., column] if j == k else 0.5 * voigt[..., column]
        strain[..., j, k] = value
        strain[..., k, j] = value
    return strain


def piezoelectric_to_voigt(tensor: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Convert a piezoelectric tensor symmetric in its last indices to 3x6 form."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Piezoelectric tensor must end in shape (3, 3, 3), got {tensor.shape}.")
    if not np.allclose(tensor, np.swapaxes(tensor, -1, -2), atol=atol):
        raise ValueError("Piezoelectric tensor must be symmetric in its last two indices.")

    voigt = np.empty((*tensor.shape[:-3], 3, 6), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        voigt[..., :, column] = tensor[..., :, j, k]
    return voigt


def voigt_to_piezoelectric(voigt: np.ndarray) -> np.ndarray:
    """Convert a 3x6 piezoelectric matrix to a symmetric rank-3 tensor."""
    voigt = np.asarray(voigt, dtype=float)
    if voigt.shape[-2:] != (3, 6):
        raise ValueError(f"Piezoelectric Voigt data must end in shape (3, 6), got {voigt.shape}.")

    tensor = np.zeros((*voigt.shape[:-2], 3, 3, 3), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        tensor[..., :, j, k] = voigt[..., :, column]
        tensor[..., :, k, j] = voigt[..., :, column]
    return tensor


def generate_strains(amplitude: float) -> np.ndarray:
    """Return one reference and positive/negative versions of all six strain modes."""
    if amplitude <= 0:
        raise ValueError("The strain amplitude must be positive.")

    strains = [np.zeros((3, 3))]
    for column in range(6):
        mode = np.zeros(6)
        mode[column] = amplitude
        strain = voigt_to_strain(mode)
        strains.extend((strain, -strain))
    return np.asarray(strains)


def apply_strains(reference: Atoms, strains: np.ndarray) -> List[Atoms]:
    """Apply homogeneous strains while keeping fractional atomic positions fixed."""
    if not np.all(reference.get_pbc()):
        raise ValueError("Piezoelectric strains require a fully periodic structure.")

    strains = np.asarray(strains, dtype=float)
    strain_to_voigt(strains)  # shape and symmetry validation
    reference_cell = reference.cell.array

    structures = []
    for strain in strains:
        deformation = np.eye(3) + strain
        if np.linalg.det(deformation) <= 0:
            raise ValueError("A strain produced a non-positive cell volume.")

        atoms = reference.copy()
        # ASE stores lattice vectors as rows. For column-vector convention
        # r' = F r, the row-vector cell therefore transforms as C' = C F^T.
        atoms.set_cell(reference_cell @ deformation.T, scale_atoms=True)
        atoms.info["strain"] = strain.copy()
        structures.append(atoms)
    return structures


def build_strained_structures(reference: Atoms, amplitude: float) -> List[Atoms]:
    """Build the shared central-difference structure set for both piezo tensors."""
    return apply_strains(reference, generate_strains(amplitude))


def strains_from_cells(reference_cell: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Recover infinitesimal symmetric strains from deformed ASE cell matrices."""
    reference_cell = np.asarray(reference_cell, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if reference_cell.shape != (3, 3):
        raise ValueError(f"Reference cell must have shape (3, 3), got {reference_cell.shape}.")
    if cells.shape[-2:] != (3, 3):
        raise ValueError(f"Cells must end in shape (3, 3), got {cells.shape}.")

    deformation = np.einsum("ij,...jk->...ik", np.linalg.inv(reference_cell), cells)
    deformation = np.swapaxes(deformation, -1, -2)
    displacement_gradient = deformation - np.eye(3)
    return 0.5 * (displacement_gradient + np.swapaxes(displacement_gradient, -1, -2))


def unwrap_polarizations(
    polarizations: np.ndarray,
    cells: np.ndarray,
    reference_index: int = 0,
    quantum_scale: float = 1.0,
) -> np.ndarray:
    """Move strained-cell polarizations onto the branch nearest the reference.

    Polarization is converted to reduced coordinates using the polarization
    quanta ``a_i / volume`` of each strained cell. ``quantum_scale`` is the
    value of one e/Å² in the input polarization unit (1 for e/Å² and
    16.02176634 for C/m²). Integer jumps are removed in reduced coordinates
    before converting back to Cartesian polarization in the original unit.
    """
    polarizations = np.asarray(polarizations, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if polarizations.ndim != 2 or polarizations.shape[1] != 3:
        raise ValueError(f"Polarizations must have shape (N, 3), got {polarizations.shape}.")
    if cells.shape != (len(polarizations), 3, 3):
        raise ValueError(f"Cells must have shape ({len(polarizations)}, 3, 3), got {cells.shape}.")
    if not 0 <= reference_index < len(polarizations):
        raise IndexError("reference_index is outside the polarization array.")
    if quantum_scale <= 0:
        raise ValueError("quantum_scale must be positive.")

    volumes = np.abs(np.linalg.det(cells))
    if np.any(volumes == 0):
        raise ValueError("Polarization cells must have non-zero volume.")

    reduced = np.empty_like(polarizations)
    for n, (polarization, cell, volume) in enumerate(zip(polarizations, cells, volumes)):
        reduced[n] = polarization / quantum_scale * volume @ np.linalg.inv(cell)

    reference = reduced[reference_index]
    reduced -= np.rint(reduced - reference)

    unwrapped = np.empty_like(polarizations)
    for n, (polarization, cell, volume) in enumerate(zip(reduced, cells, volumes)):
        unwrapped[n] = polarization @ cell / volume * quantum_scale
    return unwrapped


def proper_piezoelectric_tensor(
    improper: np.ndarray, reference_polarization: np.ndarray
) -> np.ndarray:
    """Apply the proper-piezoelectric geometric correction for symmetric strain."""
    improper = np.asarray(improper, dtype=float)
    polarization = np.asarray(reference_polarization, dtype=float)
    if improper.shape != (3, 3, 3):
        raise ValueError(f"Improper tensor must have shape (3, 3, 3), got {improper.shape}.")
    if polarization.shape != (3,):
        raise ValueError(f"Reference polarization must have shape (3,), got {polarization.shape}.")

    delta = np.eye(3)
    correction = np.zeros_like(improper)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                correction[i, j, k] = delta[j, k] * polarization[i] - 0.5 * (
                    delta[i, j] * polarization[k] + delta[i, k] * polarization[j]
                )
    return improper + correction


@dataclass(frozen=True)
class PiezoelectricResult:
    """Both piezoelectric tensors obtained from one polarization/strain fit."""

    reference_polarization: np.ndarray
    improper: ImproperPiezoelectricTensor
    proper: ProperPiezoelectricTensor
    rank: int
    residual_rms: float

    @property
    def improper_voigt(self) -> np.ndarray:
        return piezoelectric_to_voigt(self.improper.data)

    @property
    def proper_voigt(self) -> np.ndarray:
        return piezoelectric_to_voigt(self.proper.data)


@dataclass(frozen=True)
class DipoleLatticeFit:
    """Dipole derivatives and piezoelectric tensors from one cell-response fit."""

    reference_dipole: np.ndarray
    dipole_strain_derivative: np.ndarray
    dipole_lattice_derivative: np.ndarray
    result: PiezoelectricResult
    linear_system: LinearSystem


def evaluate_dipole_lattice_derivative(
    dipoles: np.ndarray,
    cells: np.ndarray,
    reference_cell: np.ndarray,
    proper_symmetry_basis: np.ndarray = None,
) -> DipoleLatticeFit:
    """Fit d(dipole)/d(cell), imposing the exact three rotational responses.

    Six symmetric deformation modes and the reference dipole are fitted in one
    linear system. For the three antisymmetric modes covariance supplies
    ``delta_mu = omega @ mu``. The full 3x3x3 lattice derivative is then
    reconstructed and converted to the improper polarization response.
    """
    dipoles = np.asarray(dipoles, dtype=float)
    cells = np.asarray(cells, dtype=float)
    reference_cell = np.asarray(reference_cell, dtype=float)
    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError("Dipoles must have shape (N, 3).")
    if cells.shape != (len(dipoles), 3, 3):
        raise ValueError(f"Cells must have shape ({len(dipoles)}, 3, 3).")
    if reference_cell.shape != (3, 3):
        raise ValueError("The reference cell must have shape (3, 3).")

    inverse_cell = np.linalg.inv(reference_cell)
    deformation_gradients = np.asarray([(inverse_cell @ cell).T for cell in cells])
    displacement_gradients = deformation_gradients - np.eye(3)
    strains = 0.5 * (displacement_gradients + displacement_gradients.swapaxes(1, 2))
    rotations = 0.5 * (displacement_gradients - displacement_gradients.swapaxes(1, 2))
    strain_voigt = strain_to_voigt(strains)

    volume = abs(np.linalg.det(reference_cell))
    symmetry_basis = None
    if proper_symmetry_basis is not None:
        symmetry_basis = np.asarray(proper_symmetry_basis, dtype=float)
        if symmetry_basis.ndim != 2 or symmetry_basis.shape[0] != 27:
            raise ValueError("The proper-piezoelectric symmetry basis must have shape (27, M).")
        unit_vectors = np.eye(3)
        corrections = np.asarray(
            [proper_piezoelectric_tensor(np.zeros((3, 3, 3)), vector) for vector in unit_vectors]
        )
        geometric_modes = np.asarray(
            [
                np.einsum("jk,i->ijk", np.eye(3), vector) - correction
                for vector, correction in zip(unit_vectors, corrections)
            ]
        )

    blocks = []
    for strain_tensor, strain, rotation in zip(strains, strain_voigt, rotations):
        if symmetry_basis is None:
            derivative_block = np.kron(strain.reshape(1, 6), np.eye(3))
            reference_block = np.eye(3) + rotation
        else:
            tensor_design = np.zeros((3, 27))
            for component in range(3):
                tensor_design[component, 9 * component : 9 * component + 9] = strain_tensor.reshape(
                    9
                )
            derivative_block = tensor_design @ (volume * symmetry_basis)
            reference_block = (
                np.eye(3) + rotation + np.einsum("lijk,jk->il", geometric_modes, strain_tensor)
            )
        blocks.append(np.hstack((derivative_block, reference_block)))
    design = np.vstack(blocks)
    system = LinearSystem(A=design, b=dipoles.reshape(-1)).solve()
    if system.rank < design.shape[1]:
        raise ValueError(
            "The cell configurations do not determine all requested dipole/cell-response "
            f"parameters (rank {system.rank}, expected {design.shape[1]})."
        )

    coefficients = system.x[:, 0]
    if symmetry_basis is None:
        dipole_strain_voigt = coefficients[:18].reshape((6, 3)).T
        dipole_strain = voigt_to_piezoelectric(dipole_strain_voigt)
        reference_dipole = coefficients[18:]
    else:
        number_of_modes = symmetry_basis.shape[1]
        reference_dipole = coefficients[number_of_modes:]
        symmetry_proper = (symmetry_basis @ coefficients[:number_of_modes]).reshape((3, 3, 3))
        geometric_response = np.einsum("l,l...->...", reference_dipole, geometric_modes)
        dipole_strain = volume * symmetry_proper + geometric_response
    reference_polarization = reference_dipole / volume

    delta = np.eye(3)
    improper = dipole_strain / volume - np.einsum("jk,i->ijk", delta, reference_polarization)
    proper = proper_piezoelectric_tensor(improper, reference_polarization)

    # d(mu_i)/d(F_jk): symmetric fitted response plus exact vector rotation.
    rotation_response = 0.5 * (
        np.einsum("ij,k->ijk", delta, reference_dipole)
        - np.einsum("ik,j->ijk", delta, reference_dipole)
    )
    deformation_derivative = dipole_strain + rotation_response
    # F = C^T C0^{-T}; convert d(mu)/dF to d(mu)/dC (ASE row-cell convention).
    lattice_derivative = np.einsum("ibk,ka->iab", deformation_derivative, inverse_cell)

    residual_rms = float(system.rms_residual[0])
    result = PiezoelectricResult(
        reference_polarization=reference_polarization,
        improper=ImproperPiezoelectricTensor(data=improper, cell=reference_cell, basis="cartesian"),
        proper=ProperPiezoelectricTensor(data=proper, cell=reference_cell, basis="cartesian"),
        rank=system.rank,
        residual_rms=residual_rms,
    )
    return DipoleLatticeFit(
        reference_dipole=reference_dipole,
        dipole_strain_derivative=dipole_strain,
        dipole_lattice_derivative=lattice_derivative,
        result=result,
        linear_system=system,
    )


def evaluate_piezoelectric_tensors(
    polarizations: np.ndarray,
    strains: np.ndarray,
    *,
    cell: np.ndarray = None,
    basis: str = "cartesian",
) -> PiezoelectricResult:
    """Fit improper and proper tensors from the same strained configurations.

    Polarizations must already be placed on a common Berry-phase branch. The
    returned tensors have the same polarization units as the input because
    strain is dimensionless.
    """
    polarizations = np.asarray(polarizations, dtype=float)
    strains = np.asarray(strains, dtype=float)
    if polarizations.ndim != 2 or polarizations.shape[1] != 3:
        raise ValueError(f"Polarizations must have shape (N, 3), got {polarizations.shape}.")
    if strains.shape != (len(polarizations), 3, 3):
        raise ValueError(
            f"Strains must have shape ({len(polarizations)}, 3, 3), got {strains.shape}."
        )
    if not np.all(np.isfinite(polarizations)) or not np.all(np.isfinite(strains)):
        raise ValueError("Polarizations and strains must be finite.")

    strain_voigt = strain_to_voigt(strains)
    design = np.column_stack((np.ones(len(strain_voigt)), strain_voigt))
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise ValueError(
            "Strained configurations do not span the six symmetric strain modes "
            f"(design rank {rank}, expected {design.shape[1]})."
        )

    coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, polarizations, rcond=None)
    if fitted_rank != rank:
        raise RuntimeError(f"Inconsistent least-squares ranks: {fitted_rank} != {rank}.")

    reference_polarization = coefficients[0]
    improper_data = voigt_to_piezoelectric(coefficients[1:].T)
    proper_data = proper_piezoelectric_tensor(improper_data, reference_polarization)
    residual = design @ coefficients - polarizations
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    tensor_kwargs = {"cell": cell, "basis": basis}
    return PiezoelectricResult(
        reference_polarization=reference_polarization,
        improper=ImproperPiezoelectricTensor(data=improper_data, **tensor_kwargs),
        proper=ProperPiezoelectricTensor(data=proper_data, **tensor_kwargs),
        rank=rank,
        residual_rms=residual_rms,
    )


def proper_piezoelectric_symmetry_basis(unit_cell, atol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis for symmetry-allowed proper piezo tensors.

    The final two covariant indices are symmetrized before extracting the
    independent column space, consistently with infinitesimal strain.
    """
    template = ProperPiezoelectricTensor(
        data=np.zeros((3, 3, 3)), cell=unit_cell.cell, basis="cartesian"
    )
    symmetrizer, _, _ = unit_cell.get_symmetrizer(template)
    modes = symmetrizer.reshape((3, 3, 3, -1))
    modes = 0.5 * (modes + modes.swapaxes(1, 2))
    modes = modes.reshape((27, -1))
    if modes.shape[1] == 0:
        return np.empty((27, 0))

    left, singular_values, _ = np.linalg.svd(modes, full_matrices=False)
    if not len(singular_values):
        return np.empty((27, 0))
    threshold = atol * max(modes.shape) * singular_values[0]
    return left[:, singular_values > threshold]


def piezoelectric_symbolic_matrix(symmetry_basis: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    """Represent a symmetry-allowed piezoelectric subspace as a symbolic 3x6 matrix."""
    symmetry_basis = np.asarray(symmetry_basis, dtype=float)
    if symmetry_basis.ndim != 2 or symmetry_basis.shape[0] != 27:
        raise ValueError("The piezoelectric symmetry basis must have shape (27, M).")

    number_of_modes = symmetry_basis.shape[1]
    voigt_basis = (
        np.column_stack(
            [piezoelectric_to_voigt(mode.reshape(3, 3, 3)).reshape(-1) for mode in symmetry_basis.T]
        )
        if number_of_modes
        else np.empty((18, 0))
    )

    independent_rows = []
    rank = 0
    for row in range(18):
        candidate = independent_rows + [row]
        candidate_rank = np.linalg.matrix_rank(voigt_basis[candidate], tol=atol)
        if candidate_rank > rank:
            independent_rows.append(row)
            rank = candidate_rank
        if rank == number_of_modes:
            break
    if rank != number_of_modes:
        raise ValueError("Could not identify all independent piezoelectric components.")

    if number_of_modes:
        independent = voigt_basis[independent_rows]
        coefficients = voigt_basis @ np.linalg.inv(independent)
    else:
        coefficients = np.empty((18, 0))

    def parameter_name(index):
        return chr(ord("a") + index) if index < 26 else f"a{index + 1}"

    def expression(row):
        nonzero = np.flatnonzero(np.abs(row) > atol)
        if not len(nonzero):
            return "0"
        terms = []
        for index in nonzero:
            coefficient = row[index]
            name = parameter_name(index)
            if np.isclose(abs(coefficient), 1.0, atol=atol):
                term = name
            else:
                term = f"{abs(coefficient):.4g}{name}"
            if not terms:
                terms.append(f"-{term}" if coefficient < 0 else term)
            else:
                terms.append((" - " if coefficient < 0 else " + ") + term)
        return "".join(terms)

    return np.asarray([expression(row) for row in coefficients], dtype=object).reshape((3, 6))


def evaluate_proper_piezoelectric_direct(
    polarizations: np.ndarray,
    strains: np.ndarray,
    symmetry_basis: np.ndarray,
    *,
    cell: np.ndarray = None,
    basis: str = "cartesian",
):
    """Fit the proper tensor directly in a supplied symmetry-allowed basis.

    The measured polarization slope is related to the proper tensor through
    Vanderbilt's geometric correction. Both the proper-tensor coefficients
    and the reference polarization are therefore fitted in one linear system.
    """
    polarizations = np.asarray(polarizations, dtype=float)
    strains = np.asarray(strains, dtype=float)
    symmetry_basis = np.asarray(symmetry_basis, dtype=float)
    if polarizations.shape != (len(strains), 3):
        raise ValueError("Polarizations must have shape (N, 3).")
    if strains.shape != (len(polarizations), 3, 3):
        raise ValueError("Strains must have shape (N, 3, 3).")
    if symmetry_basis.ndim != 2 or symmetry_basis.shape[0] != 27:
        raise ValueError("The proper-piezoelectric symmetry basis must have shape (27, M).")

    number = len(strains)
    tensor_design = np.zeros((3 * number, 27))
    polarization_design = np.zeros((3 * number, 3))
    unit_vectors = np.eye(3)
    corrections = np.asarray(
        [proper_piezoelectric_tensor(np.zeros((3, 3, 3)), vector) for vector in unit_vectors]
    )
    for n, strain in enumerate(strains):
        for component in range(3):
            row = 3 * n + component
            start = 9 * component
            tensor_design[row, start : start + 9] = strain.reshape(9)
        polarization_design[3 * n : 3 * n + 3] = np.eye(3) - np.einsum(
            "lijk,jk->il", corrections, strain
        )

    design = np.column_stack((tensor_design @ symmetry_basis, polarization_design))
    expected_rank = design.shape[1]
    rank = int(np.linalg.matrix_rank(design))
    if rank < expected_rank:
        raise ValueError(
            "The strained configurations do not determine all symmetry-allowed proper "
            f"piezoelectric parameters (rank {rank}, expected {expected_rank})."
        )
    coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, polarizations.reshape(-1), rcond=None)
    if fitted_rank != rank:
        raise RuntimeError(f"Inconsistent least-squares ranks: {fitted_rank} != {rank}.")

    number_of_modes = symmetry_basis.shape[1]
    proper_data = (symmetry_basis @ coefficients[:number_of_modes]).reshape((3, 3, 3))
    reference_polarization = coefficients[number_of_modes:]
    residual = design @ coefficients - polarizations.reshape(-1)
    return (
        ProperPiezoelectricTensor(data=proper_data, cell=cell, basis=basis),
        reference_polarization,
        rank,
        float(np.sqrt(np.mean(residual**2))),
    )


def evaluate_piezoelectric_from_structures(
    structures: Sequence[Atoms],
    reference: Atoms,
    polarization_key: str = "REF_polarization",
    unwrap: bool = True,
    polarization_quantum_scale: float = 1.0,
) -> PiezoelectricResult:
    """Evaluate both tensors from one sequence of polarized strained structures."""
    if not structures:
        raise ValueError("At least one strained structure is required.")
    if not np.all(reference.get_pbc()):
        raise ValueError("The reference structure must be fully periodic.")

    cells = np.asarray([atoms.cell.array for atoms in structures])
    polarizations = np.asarray([atoms.info[polarization_key] for atoms in structures])
    strains = strains_from_cells(reference.cell.array, cells)
    if unwrap:
        reference_index = int(np.argmin(np.linalg.norm(strains.reshape(len(strains), -1), axis=1)))
        polarizations = unwrap_polarizations(
            polarizations,
            cells,
            reference_index,
            quantum_scale=polarization_quantum_scale,
        )

    return evaluate_piezoelectric_tensors(
        polarizations,
        strains,
        cell=reference.cell.array,
        basis="cartesian",
    )
