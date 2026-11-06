import numpy as np
import pytest

from rl_reorientation import attitude as att


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_quat_mrp_roundtrip(rng):
    for _ in range(100):
        sigma = att.random_mrp(rng)
        assert np.linalg.norm(sigma) <= 1.0 + 1e-12
        np.testing.assert_allclose(att.quat_to_mrp(att.mrp_to_quat(sigma)), sigma, atol=1e-12)


def test_dcm_roundtrip_and_orthogonality(rng):
    for _ in range(100):
        sigma = att.random_mrp(rng)
        C = att.mrp_to_dcm(sigma)
        np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(C) == pytest.approx(1.0)
        np.testing.assert_allclose(att.dcm_to_mrp(C), sigma, atol=1e-10)


def test_principal_angle(rng):
    for angle in np.deg2rad([0.0, 1.0, 45.0, 90.0, 179.0]):
        assert att.principal_angle(att.random_mrp_with_angle(angle, rng)) == pytest.approx(angle)


def test_error_and_add_are_inverse(rng):
    for _ in range(50):
        sigma_RN, sigma_BR = att.random_mrp(rng), att.random_mrp(rng)
        sigma_BN = att.mrp_add(sigma_BR, sigma_RN)
        np.testing.assert_allclose(att.mrp_error(sigma_BN, sigma_RN), sigma_BR, atol=1e-10)
    np.testing.assert_allclose(att.mrp_error(sigma_RN, sigma_RN), np.zeros(3), atol=1e-12)


def test_shadow_set_same_rotation():
    sigma = np.array([0.9, 0.5, 0.3])  # |sigma| > 1
    shadow = att.mrp_shadow(sigma)
    assert np.linalg.norm(shadow) < 1.0
    np.testing.assert_allclose(att.mrp_to_dcm(shadow), att.mrp_to_dcm(sigma), atol=1e-12)
