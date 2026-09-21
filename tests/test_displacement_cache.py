import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from fd2bec.displacement_cache import (
    born_charge_mode_cache_metadata,
    displacement_cache_metadata,
    restore_born_charge_mode_cache,
    restore_displacement_cache,
    save_born_charge_mode_cache,
    save_displacement_cache,
)


def test_displacement_cache_restores_structures_and_displacements(tmp_path):
    reference = tmp_path / "reference.extxyz"
    reference_atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    ase_write(str(reference), reference_atoms, format="extxyz")
    structures = tmp_path / "structures.extxyz"
    displaced_atoms = reference_atoms.copy()
    displaced_atoms.positions[0, 0] = 0.1
    ase_write(str(structures), [reference_atoms, displaced_atoms], format="extxyz")
    metadata = displacement_cache_metadata("bec", 0.001, 0.0001, False, None, None)
    cache_dir = tmp_path / ".fd2bec"
    save_displacement_cache(
        str(cache_dir), metadata, str(reference), str(structures), np.array([[0.0, 1.0]]), "%.8f"
    )
    assert (cache_dir / "structures.extxyz").is_file()
    assert (cache_dir / "displacements.txt").is_file()
    assert (cache_dir / "displacements.info").is_file()

    restored_structures = tmp_path / "restored.extxyz"
    restored_displacements = tmp_path / "restored.txt"
    assert restore_displacement_cache(
        str(cache_dir), metadata, str(reference), str(restored_structures), str(restored_displacements)
    )
    assert len(ase_read(str(restored_structures), index=":")) == 2
    np.testing.assert_allclose(np.loadtxt(restored_displacements), [0.0, 1.0])


def test_displacement_cache_metadata_records_symmetry_settings():
    first = displacement_cache_metadata("bec", 0.001, 0.0001, False, None, None)
    second = displacement_cache_metadata("bec", 0.001, 0.001, False, None, None)
    assert first != second


def test_born_charge_mode_cache_restores_matching_structure(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[2, 2, 2], pbc=True)
    metadata = born_charge_mode_cache_metadata(0.0001)
    component_modes = np.arange(18.0).reshape((2, 9))
    cache_dir = tmp_path / ".fd2bec"

    save_born_charge_mode_cache(str(cache_dir), metadata, atoms, component_modes)
    assert (cache_dir / "born-charge-component-modes.txt").is_file()
    assert (cache_dir / "born-charge-component-modes.info").is_file()

    np.testing.assert_allclose(
        restore_born_charge_mode_cache(str(cache_dir), metadata, atoms), component_modes
    )


def test_born_charge_mode_cache_writes_sparse_matrix_without_numerical_noise(tmp_path):
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    metadata = born_charge_mode_cache_metadata(0.0001)
    component_modes = np.array([[0.5, 1e-6, -1e-10], [0.0, -0.25, 1e-12]])
    cache_dir = tmp_path / ".fd2bec"

    save_born_charge_mode_cache(str(cache_dir), metadata, atoms, component_modes)

    contents = (cache_dir / "born-charge-component-modes.txt").read_text(encoding="utf-8")
    assert "0 0 5.000000000000e-01" in contents
    assert "1 1 -2.500000000000e-01" in contents
    assert "1.000000000000e-06" not in contents
    restored = restore_born_charge_mode_cache(str(cache_dir), metadata, atoms)
    np.testing.assert_allclose(restored, [[0.5, 0.0, 0.0], [0.0, -0.25, 0.0]])


def test_born_charge_mode_cache_metadata_records_symprec():
    atoms = Atoms("H", positions=[[0.123456789012, 0, 0]], cell=[2, 2, 2], pbc=True)
    assert born_charge_mode_cache_metadata(0.0001) != born_charge_mode_cache_metadata(0.001)
