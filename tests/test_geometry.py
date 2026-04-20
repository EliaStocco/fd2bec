import pytest
import numpy as np
from fd2bec import ATOL
from fd2bec.mathematics import affine2homogeneous, homogeneous2affine


def random_rotation(dim):
    """Generate a random orthogonal matrix (QR decomposition)."""
    A = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(A)
    # Ensure det = +1 (proper rotation)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def random_translations(Nops, dim):
    return np.random.randn(Nops, dim)


def test_round_trip_single():
    dim = 3
    R = random_rotation(dim)
    t = np.random.randn(dim)

    H = affine2homogeneous(np.array([R]), np.array([t]))[0]
    R_rec, t_rec = homogeneous2affine(H)

    assert np.allclose(R, R_rec, atol=ATOL)
    assert np.allclose(t, t_rec, atol=ATOL)


def test_round_trip_batch():
    Nops = 5
    dim = 3

    R = np.array([random_rotation(dim) for _ in range(Nops)])
    T = random_translations(Nops, dim)

    H = affine2homogeneous(R, T)
    R_rec, T_rec = homogeneous2affine(H)

    assert np.allclose(R, R_rec, atol=ATOL)
    assert np.allclose(T, T_rec, atol=ATOL)


def test_homogeneous_structure():
    dim = 3
    R = random_rotation(dim)
    t = np.random.randn(dim)

    H = affine2homogeneous(np.array([R]), np.array([t]))[0]

    # Check last row
    expected_last_row = np.zeros(dim + 1)
    expected_last_row[-1] = 1.0

    assert np.allclose(H[-1, :], expected_last_row, atol=ATOL)


def test_group_property_composition():
    """
    Test that composition of affine transforms corresponds to matrix multiplication
    in homogeneous form.
    """
    dim = 3

    R1 = random_rotation(dim)
    t1 = np.random.randn(dim)

    R2 = random_rotation(dim)
    t2 = np.random.randn(dim)

    H1 = affine2homogeneous(np.array([R1]), np.array([t1]))[0]
    H2 = affine2homogeneous(np.array([R2]), np.array([t2]))[0]

    # Compose via homogeneous matrices
    H_comp = H1 @ H2

    # Extract affine components
    R_comp, t_comp = homogeneous2affine(H_comp)

    # Expected affine composition
    R_expected = R1 @ R2
    t_expected = R1 @ t2 + t1

    assert np.allclose(R_comp, R_expected, atol=ATOL)
    assert np.allclose(t_comp, t_expected, atol=ATOL)


def test_inverse_property():
    """
    Check that inverse homogeneous transformation undoes the original.
    """
    dim = 3

    R = random_rotation(dim)
    t = np.random.randn(dim)

    H = affine2homogeneous(np.array([R]), np.array([t]))[0]

    # Compute inverse analytically
    R_inv = R.T
    t_inv = -R_inv @ t

    H_inv = affine2homogeneous(np.array([R_inv]), np.array([t_inv]))[0]

    # Should be identity
    H_id = H @ H_inv

    Id = np.eye(dim + 1)

    assert np.allclose(H_id, Id, atol=ATOL)


def test_batch_consistency():
    """
    Ensure batch affine2homogeneous behaves consistently with single calls.
    """
    Nops = 4
    dim = 3

    R = np.array([random_rotation(dim) for _ in range(Nops)])
    T = random_translations(Nops, dim)

    H_batch = affine2homogeneous(R, T)

    for i in range(Nops):
        H_single = affine2homogeneous(np.array([R[i]]), np.array([T[i]]))[0]
        assert np.allclose(H_batch[i], H_single, atol=ATOL)
        
if __name__ == "__main__":
    pytest.main([__file__])