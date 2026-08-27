import numpy as np
import spglib
from ase import Atoms
from ase.data import atomic_numbers
from ase.utils import atoms_to_spglib_cell

from fd2bec.tensor_components import expand_voigt_data


def symbols2numbers(symbols):
    return [atomic_numbers[s] for s in symbols]


def numbers2symbols(numbers):
    return [list(atomic_numbers.keys())[list(atomic_numbers.values()).index(n)] for n in numbers]


def ase2spglib_dataset(atoms: Atoms, **kwargs) -> spglib.SpglibDataset:
    cell = atoms_to_spglib_cell(atoms)
    return spglib.get_symmetry_dataset(cell, **kwargs)


def invert_mapping_to_list(mapping: list[int]) -> list[list[int]]:
    """
    Invert a mapping from supercell atoms to primitive atoms
    into a list of lists grouped by primitive atom index.

    Parameters
    ----------
    mapping : array-like of int
        mapping_to_primitive from spglib (length N_super),
        where each entry gives the primitive atom index.

    Returns
    -------
    list[list[int]]
        reverse mapping such that:
        reverse_map[p] = list of supercell indices belonging to primitive atom p
    """
    mapping = np.asarray(mapping)

    n_prim = int(mapping.max()) + 1
    reverse = [[] for _ in range(n_prim)]

    for super_idx, prim_idx in enumerate(mapping):
        reverse[prim_idx].append(super_idx)

    return reverse


def allclose_chunked(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    for i in range(a.shape[0]):
        if not np.all(np.abs(a[i] - b[i]) <= atol):
            return False
    return True


def atoms2bec(atoms: Atoms, keyword: str) -> np.ndarray:
    ref_becx = atoms.arrays[f"{keyword}x"]
    ref_becy = atoms.arrays[f"{keyword}y"]
    ref_becz = atoms.arrays[f"{keyword}z"]
    bec = np.zeros((len(atoms), 3, 3))
    bec[:, :, 0] = ref_becx
    bec[:, :, 1] = ref_becy
    bec[:, :, 2] = ref_becz
    return bec  # .reshape((len(atoms), 3, 3))


def tensor_data_from_atoms(atoms: Atoms, keyword: str, tensor_name: str):
    """Return tensor data stored under an ASE info or array key."""
    if keyword in atoms.arrays:
        return np.asarray(atoms.arrays[keyword]), "atoms.arrays"
    if keyword in atoms.info:
        return np.asarray(atoms.info[keyword]), "atoms.info"

    # Some extended-XYZ writers store a Born-charge matrix as three
    # per-atom vector columns because ASE cannot represent a rank-three
    # per-atom array directly.
    split_keys = tuple(f"{keyword}{axis}" for axis in "xyz")
    if tensor_name == "bec" and all(key in atoms.arrays for key in split_keys):
        return atoms2bec(atoms, keyword), "atoms.arrays (split x/y/z fields)"

    available = sorted(set(atoms.arrays) | set(atoms.info))
    raise ValueError(
        f"Tensor keyword {keyword!r} was not found in atoms.arrays or atoms.info. "
        f"Available keys: {available}"
    )


def tensor_from_atoms(atoms: Atoms, keyword: str, tensor_name: str, tensor_class, template, basis):
    """Construct an fd2bec tensor from an ASE field in the requested basis.

    Standard Voigt data are expanded to the tensor's explicit Cartesian axes.
    Consequently, piezoelectric tensors can be read from either the current
    ``(3, 6)`` representation or the legacy ``(3, 3, 3)`` representation used
    internally by the tensor-symmetry machinery.
    """
    data, location = tensor_data_from_atoms(atoms, keyword, tensor_name)
    data = expand_voigt_data(data, template)
    tensor = tensor_class(data=np.asarray(data, dtype=float), cell=atoms.cell, basis="cartesian")
    if tensor.data.shape != template.core_shape():
        raise ValueError(
            f"Tensor keyword {keyword!r} has shape {tensor.data.shape}; "
            f"expected {template.core_shape()}."
        )
    if basis != "cartesian":
        tensor = tensor.to(basis=basis)
    return tensor, location


def shift_first_atom_to_origin(atoms: Atoms) -> Atoms:
    """Return a periodically equivalent copy with atom 0 at the fractional origin."""
    if len(atoms) == 0:
        raise ValueError("Cannot shift an empty structure.")
    if not np.all(atoms.get_pbc()):
        raise ValueError("Shifting to a fractional origin requires a fully periodic structure.")

    shifted = atoms.copy()
    fractional_positions = atoms.get_scaled_positions(wrap=False)
    fractional_positions -= fractional_positions[0]
    fractional_positions %= 1.0
    fractional_positions[np.isclose(fractional_positions, 1.0, atol=1e-12, rtol=0.0)] = 0.0
    fractional_positions[0] = 0.0
    shifted.set_scaled_positions(fractional_positions)
    return shifted


def symmetrize_bec(structure: Atoms, bec: np.ndarray) -> np.ndarray:
    from fd2bec.atomic import AtomicStructure
    from fd2bec.tensor import BornCharges

    tensor = BornCharges(data=bec)
    atomic_structure = AtomicStructure.from_ase(structure)
    return atomic_structure.symmetrize(tensor=tensor).data
